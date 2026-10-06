"""Convert task RL rows using the bundled agent prompt builders."""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from .data import normalize_row, read_rows, write_rows
from .upstream import ExternalRequired, canonical_task, render_prefix, source_hashes

EXPORTER_VERSION = 4


FILES = {
    "lifechoices": ("lifechoices_hard_rl", "lifechoices_val"),
    "fantom": ("fantom_rl_train", "fantom_val"),
    "coser": ("coser_rl_train", "coser_val"),
    "sotopia": ("sotopia_clean_rl", "sotopia_hard_val"),
    "social_r1": ("social_r1_rl", "social_r1_val"),
    "behavior_chain": ("behaviorchain_rl_train", "behaviorchain_val"),
    "alignx": ("alignx_rl_8k", "alignx_demo_val"),
    "socsci210": ("socsci210_rl_2k", "socsci210_val"),
    "sim_math": ("sim_math_rl", "sim_math_val"),
    "sim_doc": ("sim_doc_rl", "sim_doc_val"),
}
FILES.update({t: (f"{t}_rl_train", f"{t}_val") for t in (
    "userllm", "mirrorbench", "humanllm", "hitom", "paratomi", "mistakes", "twinvoice")})
FILES.update({f"humanual_{t}": (f"humanual_rl_{t}", f"humanual_{t}_val")
              for t in ("book", "chat", "email", "news", "opinion", "politics")})


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def nested(value, field):
    for key in field.split("."):
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                return None
        value = value.get(key) if isinstance(value, dict) else None
    return value


def group_id(raw, task, fields=None):
    info = raw["extra_info"]
    if fields:
        values = [nested(info, f) for f in fields]
        if any(v is None or v == "" for v in values):
            raise ValueError(f"{task}: missing explicit group_fields {fields}")
        return digest([task, values]), "configured"
    preferred = {"lifechoices": ["book"], "coser": ["circumstance.book"],
                 "fantom": ["part_id"], "hitom": ["set_id"], "sotopia": ["environment_id"],
                 "socsci210": ["study_id"]}.get(task, [])
    for field in preferred + ["user_id", "conversation_id", "post_id", "story", "raw"]:
        value = nested(info, field)
        if value is not None and value != "":
            return digest([task, field, value]), field
    # This is explicitly reported as a row-level fallback, never claimed to be a persona split.
    identity = {k: v for k, v in info.items() if k != "index"}
    return digest([task, identity]), "row_fallback"


def prepare_sources(plan, output, limit=0, sample_jsonl=False):
    from .quality_basis import visible_quality_evidence
    root, source = Path(plan["data_root"]), Path(plan["upstream_repo"])
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Data output is nonempty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    splits = defaultdict(list)
    audit = {"tasks": {}, "source_repo": str(source), "data_root": str(root),
             "exporter_version": EXPORTER_VERSION, "seed": plan.get("seed", 42),
             "coser_fallback": "Original uniform speaker fallback, sorted candidates and deterministic per-prefix RNG",
             "scope": "static policy prefixes; multi-agent trajectory metrics require the original evaluator",
             "holdout_note": "validation/calibration carved from expert training data; not unseen to original experts",
             "sample_jsonl": sample_jsonl, "limit_per_task_per_split": limit}
    validation = float(plan.get("validation_fraction", .05))
    calibration = float(plan.get("calibration_fraction", .05))
    if validation < 0 or calibration < 0 or validation + calibration >= 1:
        raise ValueError("Invalid held-out fractions")
    policy = plan.get("eval_group_overlap", "report")
    if policy not in ("report", "exclude", "error"):
        raise ValueError("eval_group_overlap must be report, exclude, or error")
    for spec in plan["experts"]:
        task = canonical_task(spec["task"])
        if task not in FILES or not spec.get("dimensions"):
            raise ValueError(f"Register a known task with explicit dimensions: {task}")
        hashes = source_hashes(source, task)
        train_stem, eval_stem = FILES[task]
        suffix = ".sample20.jsonl" if sample_jsonl else ".parquet"
        training = root / "train" / (train_stem + suffix)
        evaluations = spec.get("eval_files") or ["test/" + eval_stem + suffix]
        if task == "alignx" and "eval_files" not in spec:
            evaluations = [f"test/alignx_{s}_val{suffix}" for s in ("demo", "pair", "ugc", "arbitrary", "history16")]
        counts, grouping = Counter(), Counter()
        eval_groups, eval_prompts, used = set(), set(), set()
        task_rows = defaultdict(list)

        def convert(path, split):
            for index, raw in enumerate(read_rows(path)):
                if canonical_task(raw["data_source"]) != task:
                    # AlignX split suffixes remain one task expert.
                    if not (task == "alignx" and str(raw["data_source"]).startswith("alignx")):
                        raise ValueError(f"Unexpected data_source in {path}: {raw['data_source']}")
                counts[split + "_source_rows"] += 1
                if plan.get("objective_only"):
                    from .objective_metrics import eligible
                    usable, reason = eligible(task, raw)
                    if not usable:
                        counts[split + "_excluded_objective_metric"] += 1
                        counts["excluded_reason: " + reason] += 1
                        continue
                group, field = group_id(raw, task, spec.get("group_fields"))
                grouping[field] += 1
                # CoSER reference-history prefixes follow the original continue_from logic.
                starts = [None]
                if task == "coser" and plan.get("coser_reference_prefixes", True):
                    circumstance = nested(raw["extra_info"], "circumstance")
                    if isinstance(circumstance, str):
                        circumstance = json.loads(circumstance)
                    # The audited original agent caps its simulation at 10 rounds.
                    starts = list(range(min(10, len(circumstance.get("dialogues", []))))) or [0]
                for turn in starts:
                    entry = {**raw, "extra_info": dict(raw["extra_info"])}
                    if turn is not None:
                        entry["extra_info"]["continue_from"] = turn
                    try:
                        prefix_seed = int(digest([plan.get("seed", 42), task, str(path.relative_to(root)), index, turn])[:16], 16)
                        messages = render_prefix(entry, task, source, seed=prefix_seed)
                    except ExternalRequired:
                        counts[split + "_external_before_policy"] += 1
                        continue
                    prompt_hash = digest(messages)
                    if split == "eval":
                        eval_groups.add(group)
                        eval_prompts.add(prompt_hash)
                    else:
                        if prompt_hash in eval_prompts:
                            counts["excluded_eval_prompt_duplicate"] += 1
                            continue
                        if group in eval_groups:
                            counts["eval_group_overlap_prefixes"] += 1
                            if policy == "error":
                                raise ValueError(f"{task}: original train/test have overlapping groups")
                            if policy == "exclude":
                                continue
                    if (split, prompt_hash) in used:
                        counts[split + "_duplicates"] += 1
                        continue
                    used.add((split, prompt_hash))
                    dimensions = spec["dimensions"]
                    if plan.get("objective_only"):
                        from .objective import TASK_AXES
                        dimensions = [TASK_AXES[task]]
                    row = {"id": f"{task}:{prompt_hash[:24]}", "source_id": f"{task}:{group}:{index}",
                           "task_id": task, "dimensions": dimensions, "messages": messages,
                           "group_id": group, "source_split": split, "source_file": str(path.relative_to(root)),
                           "source_index": index, "source_turn": turn,
                           "evaluator_context": {"original_row": entry, "source_hashes": hashes,
                                                 "rubrics": spec.get("rubrics", {}),
                                                 "quality_evidence": visible_quality_evidence(task, messages, spec.get("quality_evidence")),
                                                 "hard_constraints": spec.get("hard_constraints", [])}}
                    destination = split
                    if split == "train":
                        bucket = int(digest([plan.get("seed", 42), group])[:16], 16) / 2**64
                        destination = "validation" if bucket < validation else (
                            "calibration" if bucket < validation + calibration else "train")
                    counts[destination + "_available"] += 1
                    if not limit or len(task_rows[destination]) < limit:
                        task_rows[destination].append(normalize_row(row, index))

        for path in evaluations:
            convert(root / path, "eval")
        convert(training, "train")
        if not task_rows["train"]:
            raise ValueError(f"No training rows remain for {task}; inspect grouping/overlap policy")
        for split, rows in task_rows.items():
            splits[split].extend(rows)
        audit["tasks"][task] = {"counts": dict(counts), "grouping": dict(grouping),
                                "exported": {s: len(r) for s, r in task_rows.items()}, "source_hashes": hashes}
    for split in ("train", "validation", "calibration", "eval"):
        write_rows(output / f"{split}.jsonl", splits[split])
    group_sets = {s: {r["group_id"] for r in rows} for s, rows in splits.items()}
    for a, b in (("train", "validation"), ("train", "calibration"), ("validation", "calibration")):
        if group_sets[a] & group_sets[b]:
            raise AssertionError(f"Group leakage between {a} and {b}")
    audit["eval_group_overlap_policy"] = policy
    audit["exported"] = {s: len(r) for s, r in splits.items()}
    (output / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    return audit
