"""Exact weighted LoRA delta addition, accumulated in FP32 on CPU."""

import json
import math
import re
from pathlib import Path


def normalized_weights(teachers, supplied=None):
    ids = [t["id"] for t in teachers]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Need nonempty, unique teacher IDs")
    supplied = {name: 1.0 for name in ids} if supplied is None else supplied
    if set(supplied) != set(ids):
        raise ValueError("Merge weights must specify every teacher ID exactly once")
    values = {key: float(value) for key, value in supplied.items()}
    if any(not math.isfinite(v) or v < 0 for v in values.values()) or sum(values.values()) <= 0:
        raise ValueError("Merge weights must be finite, nonnegative and have positive sum")
    total = sum(values.values())
    return {key: value / total for key, value in values.items()}


def validate_adapters(cfg):
    from opd.models import adapter_path
    paths = []
    for teacher in cfg["teachers"]:
        if "adapter" not in teacher:
            raise ValueError("Model merge currently requires common-base LoRA experts, not full checkpoints")
        path = Path(adapter_path(teacher["adapter"]))
        meta = json.loads((path / "adapter_config.json").read_text())
        if (meta.get("peft_type") != "LORA" or meta.get("bias", "none") != "none"
                or meta.get("modules_to_save") or meta.get("use_dora") or meta.get("lora_bias")
                or meta.get("layer_replication") or meta.get("trainable_token_indices")
                or str(meta.get("init_lora_weights", "")).lower().startswith(("pissa", "olora", "corda"))):
            raise ValueError(f"Only additive, bias-free LoRA is supported: {path}")
        declared = meta.get("base_model_name_or_path")
        base = cfg["model"]["base_model"]
        # Different mounts are common. The run config declares the common base;
        # PEFT shapes are checked again against actual tensors before any delta is added.
        def base_name(value):
            name = re.sub(r"[^a-z0-9]", "", Path(value).name.lower())
            # Match the hierarchy loader's Qwen3-8B naming convention:
            # Qwen/Qwen3-8B and a local Qwen_Qwen3-8B mirror.
            return "qwen38b" if name in ("qwen38b", "qwenqwen38b") else name
        if declared and str(declared) != str(base) and base_name(declared) != base_name(base):
            raise ValueError(f"Adapter declares a different base: {path}: {declared} != {base}")
        paths.append(path)
    return paths


def merge(cfg, output, weights=None, scale=1.0, resume=False):
    import torch
    from peft import PeftModel
    from peft.tuners.lora.layer import LoraLayer
    from opd.checkpoints import archive_incomplete, atomic_json
    from opd.experiments import fingerprint
    from opd.full_export import complete_export
    from opd.fusion import file_hash
    from opd.models import dtype_from_name, load_causal, load_tokenizer, verify_adapter_weights

    if not math.isfinite(scale) or scale < 0:
        raise ValueError("merge scale must be finite and nonnegative")
    paths = validate_adapters(cfg)
    weights = normalized_weights(cfg["teachers"], weights)
    output = Path(output).resolve()
    for source in [Path(cfg["model"]["base_model"]).resolve(), *paths]:
        if output == source or output in source.parents or source in output.parents:
            raise ValueError("Merge output must be separate from base and adapters")
    identity = {"base": cfg["model"]["base_model"], "weights": weights, "scale": scale,
        "dtype": cfg["model"]["dtype"], "method": "weighted_lora_delta_mean",
        "adapter_hashes": {str(p): {f.name: file_hash(f) for f in sorted(p.iterdir())
            if f.name in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin")} for p in paths}}
    signature = fingerprint(identity)
    if output.exists():
        if not resume:
            raise FileExistsError(output)
        if complete_export(output):
            if json.loads((output / "export_manifest.json").read_text())["signature"] != signature:
                raise ValueError("Existing merge has different inputs/settings")
            return str(output)
        archive_incomplete(output)
    temporary = output.with_name("." + output.name + ".pending")
    if temporary.exists():
        if not resume:
            raise FileExistsError(temporary)
        archive_incomplete(temporary)
    output.parent.mkdir(parents=True, exist_ok=True)
    model = load_causal(cfg["model"]["base_model"], "float32", cfg["model"]["trust_remote_code"])
    # Ordinary LoRA deltas are independent of the backbone. Loading each adapter
    # on the running sum is exact and keeps only one base model resident in RAM.
    for teacher, path in zip(cfg["teachers"], paths, strict=True):
        coefficient = weights[teacher["id"]] * scale
        wrapped = PeftModel.from_pretrained(model, str(path), is_trainable=False)
        verify_adapter_weights(wrapped, path)
        with torch.no_grad():
            for module in wrapped.modules():
                if isinstance(module, LoraLayer):
                    delta = module.get_delta_weight("default")
                    if not torch.isfinite(delta).all():
                        raise ValueError(f"Nonfinite adapter delta: {path}")
                    module.get_base_layer().weight.add_(delta, alpha=coefficient)
        model = wrapped.unload()  # Do not merge a second time.
    model.to(dtype=dtype_from_name(cfg["model"]["dtype"]))
    for parameter in model.parameters():
        if not torch.isfinite(parameter).all():
            raise ValueError("Nonfinite merged weights after dtype conversion")
    model.config.use_cache = True
    model.save_pretrained(temporary, safe_serialization=True, max_shard_size="5GB")
    load_tokenizer(cfg).save_pretrained(temporary)
    atomic_json(temporary / "opd_metadata.json", {"student_mode": "full", "dimension": None,
        "base_model": cfg["model"]["base_model"], "standalone": True, **identity})
    files = {p.name: p.stat().st_size for p in temporary.iterdir() if p.is_file()}
    atomic_json(temporary / "export_manifest.json", {"signature": signature, **identity, "files": files})
    temporary.rename(output)
    return str(output)
