"""Model loading and a frozen shared-backbone LoRA teacher registry."""

import contextlib
import inspect
import json
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def dtype_from_name(name):
    try:
        return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {name}") from exc


def adapter_path(path):
    """Accept PEFT dir, actor dir, or an Simulation global_step_N dir."""
    root = Path(path)
    for candidate in (root, root / "lora_adapter", root / "actor" / "lora_adapter"):
        if (candidate / "adapter_config.json").is_file():
            if not any((candidate / f).is_file() for f in ("adapter_model.safetensors", "adapter_model.bin")):
                raise ValueError(f"Adapter weights missing in {candidate}; Simulation export may have failed")
            return str(candidate.resolve())
    raise ValueError(f"No PEFT adapter in {root}. Expected adapter_config.json and adapter_model.safetensors. "
                     "Raw FSDP model_world_size_* shards are not PEFT adapters.")


def is_adapter(path):
    try:
        adapter_path(path)
        return True
    except ValueError:
        return False


def verify_adapter_weights(model, path, adapter_name="default"):
    path = Path(path)
    expected = get_peft_model_state_dict(model, adapter_name=adapter_name)
    if (path / "adapter_model.safetensors").is_file():
        with safe_open(path / "adapter_model.safetensors", framework="pt", device="cpu") as f:
            shapes = {k: tuple(f.get_slice(k).get_shape()) for k in f.keys()}
    else:
        weights = torch.load(path / "adapter_model.bin", map_location="cpu", weights_only=True)
        shapes = {k: tuple(v.shape) for k, v in weights.items()}
    missing, extra = set(expected)-set(shapes), set(shapes)-set(expected)
    mismatched = [k for k in set(expected) & set(shapes) if tuple(expected[k].shape) != shapes[k]]
    if missing or extra or mismatched:
        raise ValueError(f"Incomplete/incompatible adapter export {path}: "
                         f"missing={sorted(missing)}, unexpected={sorted(extra)}, shape_mismatch={mismatched}")


def disable_dropout(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0


def load_causal(path, dtype, trust=False):
    config = AutoConfig.from_pretrained(path, trust_remote_code=trust)
    # Old/new policy log-probs must describe the same dropout-free policy.
    for key in list(config.to_dict()):
        if "dropout" in key or key in ("resid_pdrop", "embd_pdrop", "attn_pdrop"):
            if isinstance(getattr(config, key, None), (float, int)):
                setattr(config, key, 0.0)
    model = AutoModelForCausalLM.from_pretrained(
        path, config=config, torch_dtype=dtype_from_name(dtype), trust_remote_code=trust
    )
    disable_dropout(model)
    return model


def load_tokenizer(cfg):
    model = cfg["model"]
    tokenizer = AutoTokenizer.from_pretrained(model.get("tokenizer", model["base_model"]),
                                               trust_remote_code=model["trust_remote_code"])
    if not tokenizer.chat_template:
        raise ValueError("Tokenizer needs the SAME chat template used to train the task experts")
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer needs eos_token_id")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_student(cfg):
    c = cfg["model"]
    initial = c.get("student_init", c["base_model"])
    initial_adapter = is_adapter(initial)
    if c["student_mode"] == "lora" and not initial_adapter and initial != c["base_model"]:
        raise ValueError("A LoRA student must train on the common base_model. For warmup continuation, "
                         "student_init must be the saved adapter, not a different full backbone.")
    model = load_causal(c["base_model"] if initial_adapter else initial, c["dtype"], c["trust_remote_code"])
    if initial_adapter:
        model = PeftModel.from_pretrained(model, adapter_path(initial), is_trainable=c["student_mode"] == "lora")
        verify_adapter_weights(model, adapter_path(initial))
        if c["student_mode"] == "full":
            model = model.merge_and_unload()
    elif c["student_mode"] == "lora":
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=c["lora_rank"],
                                                lora_alpha=c["lora_alpha"], lora_dropout=c["lora_dropout"],
                                                target_modules=c["lora_targets"]))
    if c["student_mode"] == "full":
        model.requires_grad_(True)
    if c["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if c["student_mode"] == "lora":
            model.enable_input_require_grads()
    model.config.use_cache = False
    disable_dropout(model)
    return model


def model_inputs(batch):
    return {key: batch[key] for key in ("input_ids", "attention_mask", "position_ids")}


def response_logits(model, batch, temperature=1.0):
    # Qwen3 (and recent Llama) can apply the LM head only to the requested
    # suffix. Keep one extra position for the final causal shift. Route the
    # actual forward through DDP/ZeRO/PEFT, never bypass distributed hooks.
    underlying = model
    while hasattr(underlying, "module"):
        underlying = underlying.module
    if isinstance(underlying, PeftModel):
        underlying = underlying.get_base_model()
    parameters = inspect.signature(underlying.forward).parameters
    keep = next((k for k in ("logits_to_keep", "num_logits_to_keep") if k in parameters), None)
    if keep:
        result = model(**model_inputs(batch), use_cache=False, **{keep: batch["responses"].shape[1] + 1})
        return result.logits[:, :-1, :].float() / temperature
    result = model(**model_inputs(batch), use_cache=False)
    return result.logits[:, batch["prompt_length"] - 1:-1, :].float() / temperature


@contextlib.contextmanager
def generation_model(model, accelerator):
    """Collectively materialize sharded students only for rollout, then reshard."""
    kind = str(accelerator.distributed_type)
    unwrapped = accelerator.unwrap_model(model)
    if "DEEPSPEED" in kind and accelerator.state.deepspeed_plugin.zero_stage == 3:
        import deepspeed
        with deepspeed.zero.GatheredParameters(list(model.parameters()), modifier_rank=None):
            yield unwrapped
    elif "FSDP" in kind:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        if not isinstance(model, FSDP):
            raise ValueError("This backend supports FSDP1, not FSDP2; use supplied DeepSpeed config")
        with FSDP.summon_full_params(model, recurse=True, writeback=False):
            yield unwrapped
    else:
        yield unwrapped


def teacher_load_context(accelerator):
    plugin = accelerator.state.deepspeed_plugin
    return plugin.zero3_init_context_manager(enable=False) if plugin else contextlib.nullcontext()


class TeacherBank:
    """Adapters share one frozen backbone; full-model teachers use a one-model CPU cache.

    Each process owns its teacher service. No student storage, optimizer, or EMA is shared.
    Switching PEFT adapters can enable gradients, so freeze again after every switch.
    """

    def __init__(self, cfg, tokenizer, device):
        self.cfg, self.tokenizer = cfg, tokenizer
        selected = cfg["teacher"]["device"]
        self.device = device if selected == "auto" else torch.device(selected)
        self.entries = {t["id"]: t for t in cfg.get("teachers", [])}
        self.shared = None
        self.full = None
        self.full_id = None
        self.adapter_names = {}
        self.calls = 0
        self.tokens = 0
        self._validate_full_tokenizers()
        adapters = [t for t in self.entries.values() if "adapter" in t]
        if adapters:
            backbone = load_causal(cfg["model"]["base_model"], cfg["teacher"]["dtype"],
                                   cfg["model"]["trust_remote_code"])
            for i, entry in enumerate(adapters):
                name = f"expert_{i}"
                path = adapter_path(entry["adapter"])
                config = json.loads((Path(path) / "adapter_config.json").read_text())
                if config.get("modules_to_save"):
                    # Supported by PEFT, but embedding/vocabulary changes require exact compatibility.
                    if any("embed" in n or "lm_head" in n for n in config["modules_to_save"]):
                        raise ValueError("Adapters with saved embeddings/lm_head require explicit conversion "
                                         "to a full model teacher for vocabulary verification")
                if i == 0:
                    self.shared = PeftModel.from_pretrained(backbone, path, adapter_name=name, is_trainable=False)
                else:
                    self.shared.load_adapter(path, adapter_name=name, is_trainable=False)
                verify_adapter_weights(self.shared, path, name)
                self.adapter_names[entry["id"]] = name
            self.shared.requires_grad_(False).eval().to(self.device)

    def _validate_full_tokenizers(self):
        for entry in self.entries.values():
            if "model" not in entry:
                continue
            other = AutoTokenizer.from_pretrained(entry["model"],
                                                   trust_remote_code=self.cfg["model"]["trust_remote_code"])
            if other.get_vocab() != self.tokenizer.get_vocab() or other.eos_token_id != self.tokenizer.eos_token_id:
                raise ValueError(f"Teacher {entry['id']} has an incompatible tokenizer; token KL is invalid")
            if other.chat_template != self.tokenizer.chat_template:
                raise ValueError(f"Teacher {entry['id']} has a different chat template")

    def activate(self, teacher_id):
        entry = self.entries[teacher_id]
        if "adapter" in entry:
            if self.full is not None:
                self.full.to("cpu")
            self.shared.to(self.device)
            self.shared.set_adapter(self.adapter_names[teacher_id])
            self.shared.requires_grad_(False).eval()
            return self.shared
        if self.shared is not None:
            self.shared.to("cpu")
        if self.full_id != teacher_id:
            if self.full is not None:
                self.full.to("cpu")
                self.full = None
            self.full = load_causal(entry["model"], self.cfg["teacher"]["dtype"],
                                    self.cfg["model"]["trust_remote_code"])
            self.full_id = teacher_id
        return self.full.requires_grad_(False).eval().to(self.device)

    @torch.no_grad()
    def score(self, teacher_id, batch, objective, top_k):
        model = self.activate(teacher_id)
        local = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        logits = response_logits(model, local, self.cfg["rollout"]["temperature"])
        log_probs = logits.log_softmax(-1)
        self.calls += len(batch["input_ids"])
        self.tokens += int(batch["attention_mask"].sum())
        if objective == "sampled_reverse_kl":
            return {"sampled": log_probs.gather(-1, local["responses"].unsqueeze(-1)).squeeze(-1).cpu()}
        k = min(top_k, log_probs.shape[-1])
        values, ids = log_probs.topk(k, dim=-1)
        return {"ids": ids.cpu(), "log_probs": values.cpu(), "vocab_size": log_probs.shape[-1]}

    @torch.no_grad()
    def anchor_score(self, batch):
        if self.shared is None:
            # No adapter teacher: load the common base as a temporary full entry.
            name = "__sft_anchor__"
            self.entries[name] = {"model": self.cfg["model"]["base_model"]}
            return self.score(name, batch, "sampled_reverse_kl", 1)["sampled"]
        if self.full is not None:
            self.full.to("cpu")
        self.shared.to(self.device).requires_grad_(False).eval()
        local = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        with self.shared.disable_adapter():
            logits = response_logits(self.shared, local, self.cfg["rollout"]["temperature"])
            return logits.log_softmax(-1).gather(-1, local["responses"].unsqueeze(-1)).squeeze(-1).cpu()
