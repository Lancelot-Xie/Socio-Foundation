"""Fail early on missing production inputs; never launches training or a judge call."""

import argparse
import importlib.metadata
import json
import os
import platform
from pathlib import Path

import yaml

from .experiments import base_config, load_plan
from .source_data import FILES
from .upstream import VERIFIABLE, agent_task, canonical_task, source_hashes


def check(plan):
    import torch
    from .models import adapter_path, load_tokenizer

    cfg = base_config(plan, Path(plan["output_dir"]) / "data")
    report = {"python": platform.python_version(), "platform": platform.platform(),
              "packages": {name: importlib.metadata.version(name) for name in
                           ("torch", "transformers", "peft", "accelerate", "pyarrow", "pydantic")},
              "paths": {}, "gpus": [], "experts": [], "notes": []}
    for key in ("upstream_repo", "data_root"):
        path = Path(plan[key])
        if not path.is_dir():
            raise FileNotFoundError(f"Missing {key}: {path}")
        report["paths"][key] = str(path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use train_all.sh smoke for CPU validation")
    for index in range(torch.cuda.device_count()):
        device = torch.cuda.get_device_properties(index)
        report["gpus"].append({"index": index, "name": device.name,
                               "memory_gib": round(device.total_memory / 2**30, 1)})
    if cfg["model"]["dtype"] == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Selected GPU does not support bfloat16")
    processes = 1
    if plan.get("accelerate_config") and plan.get("workflow") != "expert_pool":
        distributed = yaml.safe_load(Path(plan["accelerate_config"]).read_text())
        processes = int(os.environ.get("OPD_NUM_PROCESSES", distributed.get("num_processes", 1)))
        if int(distributed.get("num_machines", 1)) != 1:
            raise ValueError("This one-click runner supports a single node; multi-node orchestration is not configured")
        kind = distributed.get("distributed_type")
        if kind not in ("DEEPSPEED", "MULTI_GPU", "NO"):
            raise ValueError("Use DDP or DeepSpeed ZeRO-2 with this rollout backend")
        if kind == "DEEPSPEED":
            report["packages"]["deepspeed"] = importlib.metadata.version("deepspeed")
            if distributed.get("deepspeed_config", {}).get("zero_stage") != 2:
                raise ValueError("The production runner requires ZeRO-2, not ZeRO-3")
    if not 1 <= processes <= len(report["gpus"]):
        raise ValueError(f"Requested {processes} processes but only {len(report['gpus'])} visible GPUs")
    report["processes"] = processes
    inference_processes = plan.get("inference_num_processes", 1)
    if type(inference_processes) is not int or not 1 <= inference_processes <= len(report["gpus"]):
        raise ValueError(f"Requested {inference_processes} inference processes but only {len(report['gpus'])} visible GPUs")
    report["inference_num_processes"] = inference_processes
    report["generation_batches"] = {"rollout": cfg["rollout"]["generation_batch_size"],
                                     "inference": cfg["inference"]["batch_size"]}
    if plan.get("workflow") == "dimension_offline":
        from .objective import validate_objective_config
        report.update(workflow="dimension_offline", quality_basis=validate_objective_config(cfg),
                      requires_llm_judge=False, dimension_generation_experts=True,
                      excluded_dimensions=["T", "C"], pending_dimensions=["U"], automatic_fusion=False)
        report["notes"].append("F/S/N independent teacher-trajectory forward-KL LoRAs; N is a first-turn lexical proxy only.")
        if "mirrorbench" in cfg["routing"]["tasks"]:
            from .objective_metrics import lexical_encoding
            report["metric_tokenizer"] = lexical_encoding().name
    elif cfg["qgpi"]["enabled"]:
        report["workflow"] = plan.get("workflow", "qgpi")
        from .objective import validate_objective_config
        from .quality_basis import VERIFIABLE_AXIS
        if cfg["qgpi"]["evaluator"] == "opd.objective:objective_candidates":
            report["quality_basis"] = validate_objective_config(cfg)
            report["requires_llm_judge"] = False
            if "mirrorbench" in cfg["routing"]["tasks"]:
                from .objective_metrics import lexical_encoding
                report["metric_tokenizer"] = lexical_encoding().name
        else:
            report["quality_basis"] = ["F", "S", "U", "T", "N"]
        report["notes"].append("C is disabled. Dimensions are evidence-dependent, not five generation experts. Static CoSER T measures reference-history consistency, not future simulation.")
        if cfg["qgpi"]["evaluator"] == "opd.quality_basis:hybrid_candidates" and any(
            task not in VERIFIABLE_AXIS or cfg["qgpi"]["profiles"].get(task, {}).get("dimensions", [VERIFIABLE_AXIS.get(task)]) != [VERIFIABLE_AXIS.get(task)]
            for task in cfg["routing"]["tasks"]
        ):
            if not all(os.environ.get(v) for v in ("OPD_JUDGE_BASE_URL", "OPD_JUDGE_MODEL")):
                raise ValueError("QGPI requires OPD_JUDGE_BASE_URL and OPD_JUDGE_MODEL for independent rubric axes")
        if cfg["qgpi"]["evaluator"].startswith("examples."):
            raise ValueError("Synthetic quality evaluators are debug-only; do not use them for production")
    report["effective_prompt_batch"] = processes * cfg["train"]["batch_size"] * cfg["train"]["gradient_accumulation_steps"]
    if cfg["train"]["global_prompt_batch"] not in (None, report["effective_prompt_batch"]):
        raise ValueError("Resolved global prompt batch differs from the actual launcher")
    if plan.get("recipe_alignment"):
        report["recipe_alignment"] = plan["recipe_alignment"]
        budgets = ({d: plan.get("dimension_steps", {}).get(d, plan["steps"]["distill"]) for d in plan["dimensions"]}
                   if plan.get("workflow") == "dimension_offline" else plan["steps"]["final"])
        report["recipe_parameters"] = {"train": cfg["train"], "rollout": cfg["rollout"],
                                        "steps": budgets,
                                        "candidates_per_teacher": plan.get("demo_candidates", cfg["qgpi"]["candidates_per_teacher"])}
        report["notes"].append("No automatic checkpoint deletion. LoRA final/ exports adapters, but resumable Accelerate/ZeRO state may still contain frozen base weights; reserve tens of GB or more for 8B checkpoints. Full-parameter runs cost substantially more.")
    # Tokenizer loading also rejects a missing or invalid shared base path.
    tokenizer = load_tokenizer(cfg)
    report["paths"]["base_model"] = cfg["model"]["base_model"]
    report["tokenizer_vocabulary"] = len(tokenizer)
    for expert in plan["experts"]:
        task = canonical_task(expert["task"])
        source_hashes(plan["upstream_repo"], task)
        train, evaluation = FILES[task]
        defaults = ([f"test/alignx_{s}_val.parquet" for s in ("demo", "pair", "ugc", "arbitrary", "history16")]
                    if task == "alignx" else [f"test/{evaluation}.parquet"])
        files = [f"train/{train}.parquet", *expert.get("eval_files", defaults)]
        for relative in files:
            if not (Path(plan["data_root"]) / relative).is_file():
                raise FileNotFoundError(f"Missing original data: {relative}")
        from .expert_manifest import checkpoint_candidates
        for candidate in checkpoint_candidates(expert):
            path = adapter_path(candidate["adapter"]) if "adapter" in candidate else candidate["model"]
            from .objective import TASK_AXES
            dimensions = ([TASK_AXES[task]] if cfg["quality"]["evaluator"] == "opd.objective:objective_response"
                          else expert["dimensions"])
            report["experts"].append({"task": task, "id": candidate["id"], "checkpoint": path,
                                      "dimensions": dimensions, "registered_dimensions": expert["dimensions"]})
            if "adapter" in candidate:
                metadata = json.loads((Path(path) / "adapter_config.json").read_text())
                declared = metadata.get("base_model_name_or_path")
                report["experts"][-1]["declared_base"] = declared
                report["experts"][-1].update(lora_rank=metadata.get("r"), lora_alpha=metadata.get("lora_alpha"))
                if declared != cfg["model"]["base_model"]:
                    report["notes"].append(f"{candidate['id']}: adapter declares {declared!r}; verify it is the same weights as SFT_BASE after relocation")
        for candidate in expert.get("checkpoints", []):
            if not candidate.get("enabled", True):
                report["notes"].append(f"Excluded checkpoint {candidate['id']}: {candidate.get('note', 'disabled')}")
        if agent_task(task) not in VERIFIABLE and (plan.get("evaluate", True) or cfg["quality"]["filter_demos"]):
            if cfg["quality"]["evaluator"] == "opd.upstream:task_or_rubric":
                if not expert.get("rubrics"):
                    raise ValueError(f"{task}: configure explicit rubrics or a task-specific evaluator")
                if not all(os.environ.get(v) for v in ("OPD_JUDGE_BASE_URL", "OPD_JUDGE_MODEL")):
                    raise ValueError(f"{task}: set OPD_JUDGE_BASE_URL and OPD_JUDGE_MODEL before filtering/evaluation")
    report["notes"].append("Each rank holds a complete student and frozen teacher backbone; ZeRO-2 shards optimizer/gradients only. Full 8B training needs large-memory GPUs.")
    report["notes"].append("Paths and metadata checked; exact historical base weights and judge service availability require the saved expert provenance and target server validation.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(check(load_plan(args.config)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
