"""Atomically export a trained unified student as a self-contained HF model."""

import argparse
import json
import shutil
from pathlib import Path

from .checkpoints import archive_incomplete, atomic_json


def complete_export(output):
    output = Path(output)
    try:
        manifest = json.loads((output / "export_manifest.json").read_text())
        return (bool(manifest["files"]) and (output / "config.json").is_file()
                and not (output / "adapter_config.json").exists()
                and all((output / name).is_file() and (output / name).stat().st_size == size
                        for name, size in manifest["files"].items()))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def export_full(model, base, output, dtype="bfloat16", resume=False):
    from .fusion import file_hash
    from .experiments import fingerprint
    source, output = Path(model).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Export destination must be separate from the training checkpoint")
    base_path = Path(base).resolve()
    if base_path.is_dir() and (base_path == output or base_path in output.parents or output in base_path.parents):
        raise ValueError("Export destination must be separate from the base model")
    metadata = json.loads((source / "opd_metadata.json").read_text())
    if metadata.get("dimension") is not None:
        raise ValueError("Export the unified student; merging one dimension adapter does not fuse teachers")
    mode = metadata["student_mode"]
    if mode not in ("full", "lora"):
        raise ValueError(f"Unknown student mode: {mode}")
    files = sorted(p for p in source.iterdir() if p.is_file())
    identity = {"source": str(source), "base": str(base), "dtype": dtype,
                "source_hashes": {p.name: file_hash(p) for p in files}}
    signature = fingerprint(identity)
    if output.exists():
        if not resume:
            raise FileExistsError(f"Refusing to overwrite {output}")
        if complete_export(output):
            if json.loads((output / "export_manifest.json").read_text())["signature"] != signature:
                raise ValueError("Completed export belongs to a different source checkpoint")
            return {"output": str(output), "retained": True}
        archive_incomplete(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name("." + output.name + ".pending")
    if temporary.exists():
        if not resume:
            raise FileExistsError(f"Interrupted export at {temporary}; use --resume")
        archive_incomplete(temporary)
    print(f"[export] mode={mode} source={source} output={output}", flush=True)
    if mode == "lora":
        from peft import PeftModel
        from transformers import AutoTokenizer
        from .models import adapter_path, dtype_from_name, load_causal, verify_adapter_weights
        # Merge on CPU in FP32, then store at the deployment precision. This
        # process runs after training workers exit and never sums the F/S/N LoRAs.
        print("[export] Loading common base and unified student LoRA on CPU for FP32 merge", flush=True)
        merged = PeftModel.from_pretrained(load_causal(base, "float32"), adapter_path(source))
        verify_adapter_weights(merged, source)
        merged = merged.merge_and_unload(safe_merge=True).to(dtype=dtype_from_name(dtype))
        merged.config.use_cache = True
        merged.save_pretrained(temporary, safe_serialization=True, max_shard_size="5GB")
        tokenizer_source = source if (source / "tokenizer_config.json").is_file() else base
        AutoTokenizer.from_pretrained(tokenizer_source).save_pretrained(temporary)
    elif mode == "full":
        if (source / "adapter_config.json").exists() or not (source / "config.json").is_file():
            raise ValueError("Full student metadata does not match the saved model")
        config = json.loads((source / "config.json").read_text())
        stored_dtype = config.get("dtype", config.get("torch_dtype"))
        if stored_dtype == dtype:
            print("[export] Copying full student weights and tokenizer", flush=True)
            shutil.copytree(source, temporary, ignore=shutil.ignore_patterns("export_manifest.json"))
            config["use_cache"] = True
            atomic_json(temporary / "config.json", config)
        else:
            from transformers import AutoTokenizer
            from .models import load_causal
            converted = load_causal(source, dtype)
            converted.config.use_cache = True
            converted.save_pretrained(temporary, safe_serialization=True, max_shard_size="5GB")
            AutoTokenizer.from_pretrained(source).save_pretrained(temporary)
    weights = sorted(temporary.glob("model*.safetensors"))
    if not weights or not (temporary / "tokenizer_config.json").is_file():
        raise ValueError("Standalone export lacks HF weights or tokenizer")
    index = temporary / "model.safetensors.index.json"
    if index.exists():
        shards = set(json.loads(index.read_text())["weight_map"].values())
        if any(not (temporary / shard).is_file() for shard in shards):
            raise ValueError("Standalone export has a missing weight shard")
    atomic_json(temporary / "opd_metadata.json", {**metadata, "student_mode": "full",
        "trained_student_mode": mode, "standalone": True, "source_student": str(source)})
    exported = {p.name: p.stat().st_size for p in temporary.iterdir() if p.is_file()}
    atomic_json(temporary / "export_manifest.json", {"signature": signature, **identity, "files": exported,
        "method": "merge_unified_student_lora_into_base" if mode == "lora" else "copy_full_parameter_student"})
    temporary.rename(output)
    print(f"[export] Standalone model committed: {output}", flush=True)
    return {"output": str(output), "standalone": True, "trained_student_mode": mode}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--resume", action="store_true")
    print(json.dumps(export_full(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
