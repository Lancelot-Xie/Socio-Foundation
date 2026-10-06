"""Entirely local tiny random fixtures. These are NOT trained task experts."""

import copy
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from .data import write_rows


def make_fixture(output):
    root = Path(output).resolve()
    if root.exists():
        raise FileExistsError(f"Fixture directory already exists: {root}")
    root.mkdir(parents=True)
    torch.manual_seed(31)
    torch.set_num_threads(1)
    words = ["<pad>", "<bos>", "<eos>", "<unk>", "system", "user", "assistant", ":",
             "You", "are", "Lin", "likes", "tea", "coffee", "key", "drawer", "bag", "believes",
             "choose", "find", "Answer", "A", "B", "C", "D", "hello", "yes", "no", "role",
             "lifechoices", "fantom", "coser", "the", "in", "please", "."]
    raw = Tokenizer(models.WordLevel({word: i for i, word in enumerate(words)}, unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="<pad>", bos_token="<bos>",
                                        eos_token="<eos>", unk_token="<unk>")
    tokenizer.chat_template = "{% for m in messages %}{{ m['role'] + ' : ' + m['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant : ' }}{% endif %}"
    base = root / "base"
    base.mkdir()
    model = GPT2LMHeadModel(GPT2Config(vocab_size=len(words), n_positions=256, n_embd=32, n_layer=2,
                                      n_head=2, bos_token_id=1, eos_token_id=2, pad_token_id=0,
                                      resid_pdrop=0., embd_pdrop=0., attn_pdrop=0.))
    model.save_pretrained(base)
    tokenizer.save_pretrained(base)
    teachers, rows = [], []
    for i, task in enumerate(("lifechoices", "fantom", "coser")):
        teacher = get_peft_model(GPT2LMHeadModel.from_pretrained(base), LoraConfig(
            task_type="CAUSAL_LM", r=4, lora_alpha=8, target_modules=["c_attn"], lora_dropout=0.))
        # Deliberately perturb adapters so distillation gradients are nonzero.
        with torch.no_grad():
            for name, p in teacher.named_parameters():
                if "lora_B" in name:
                    p.normal_(std=.12 + i*.03)
        adapter = root / "experts" / task / "actor" / "lora_adapter"
        teacher.save_pretrained(adapter)
        teachers.append({"id": task, "adapter": str(adapter.parent.parent)})
        dimensions = ["F"] if task == "lifechoices" else ["S"] if task == "fantom" else ["F", "S"]
        for j in range(8):
            rows.append({"id": f"{task}-{j}", "task_id": task, "dimensions": dimensions,
                         "messages": [{"role": "system", "content": "You are Lin ."},
                                      {"role": "user", "content": f"{task} Lin likes tea . choose A B ."}],
                         "evaluation": {"type": "nonempty_smoke_only", "dimension": dimensions[0]}})
    write_rows(root / "train.jsonl", rows)
    write_rows(root / "calibration.jsonl", [{**r, "id": r["id"] + "-cal"} for r in rows])
    write_rows(root / "eval.jsonl", [{**r, "id": r["id"] + "-eval"} for r in rows[:4]])
    cfg = {"seed": 17, "output_dir": str(root / "outputs" / "direct"),
           "model": {"base_model": str(base), "student_mode": "full", "dtype": "float32",
                     "gradient_checkpointing": False, "lora_rank": 4, "lora_alpha": 8,
                     "lora_targets": ["c_attn"]},
           "teacher": {"device": "cpu", "dtype": "float32"}, "teachers": teachers,
           "data": {"train_file": str(root / "train.jsonl"), "calibration_file": str(root / "calibration.jsonl"),
                    "eval_file": str(root / "eval.jsonl")},
           "routing": {"tasks": {t: {"teachers": {t: 1.0}, "strength": 1.0}
                                  for t in ("lifechoices", "fantom", "coser")}},
           "rollout": {"max_prompt_tokens": 96, "max_new_tokens": 6},
           "train": {"max_steps": 3, "batch_size": 2, "gradient_accumulation_steps": 1,
                     "learning_rate": .002, "warmup_steps": 0, "save_every": 3,
                     "teacher_top_k": 8, "advantage_clip": 0.0},
           "quality": {"min_score": 0.5}}
    def save(name, config):
        (root / f"{name}.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    save("direct", cfg)
    forward = copy.deepcopy(cfg)
    forward["train"]["objective"] = "forward_kl"
    forward["output_dir"] = str(root / "outputs" / "direct_forward")
    save("direct_forward", forward)
    for dimension in ("F", "S"):
        dim = copy.deepcopy(cfg)
        dim["model"]["student_mode"] = "lora"
        dim["data"]["dimension"] = dimension
        dim["output_dir"] = str(root / "outputs" / f"dimension_{dimension}")
        save(f"dimension_{dimension}", dim)
        warm = copy.deepcopy(dim)
        warm["train"]["stage"] = "warmup"
        warm["data"]["train_file"] = str(root / f"demos_{dimension}.jsonl")
        warm["output_dir"] = str(root / "outputs" / f"warmup_{dimension}")
        save(f"warmup_{dimension}", warm)
        dim["model"]["student_init"] = str(root / "outputs" / f"warmup_{dimension}" / "final")
        save(f"dimension_{dimension}_from_warmup", dim)
    final = copy.deepcopy(cfg)
    final["teachers"] = [{"id": d, "adapter": str(root / "outputs" / f"dimension_{d}" / "final")}
                         for d in ("F", "S")]
    final["routing"]["tasks"] = {"lifechoices": {"teachers": {"F": 1.}},
                                  "fantom": {"teachers": {"S": 1.}},
                                  "coser": {"teachers": {"F": .5, "S": .5}}}
    final["output_dir"] = str(root / "outputs" / "final_full")
    save("final", final)
    return {"fixture": str(root), "parameters": sum(p.numel() for p in model.parameters()),
            "warning": "Random synthetic teachers and nonempty scores validate mechanics only, not capability."}
