"""Tokenize and audit source data in a separate process, releasing inference dependencies."""

import argparse
import json
from pathlib import Path

from .config import load_config
from .data import read_rows, write_rows


def run(base, data, destination, policy):
    """Explicitly report/exclude overlong prefixes instead of truncating persona context."""
    from .models import load_tokenizer
    from .data import encode_prompt, tokenizer_signature
    tokenizer = load_tokenizer(base)
    if policy not in ("exclude", "error"):
        raise ValueError("data.overlong_policy must be exclude or error")
    report = {"policy": policy, "tokenizer_signature": tokenizer_signature(tokenizer, base),
              "max_prompt_tokens": base["rollout"]["max_prompt_tokens"], "splits": {}}
    for split in ("train", "validation", "calibration", "eval"):
        path = data / (split + ".jsonl")
        if not path.exists():
            continue
        retained, counts = [], {}
        for row in read_rows(path):
            task = row["task_id"]
            count = counts.setdefault(task, {"retained": 0, "overlong": 0})
            try:
                encode_prompt(tokenizer, row["messages"], base)
            except ValueError as error:
                if "exceeding max_prompt_tokens" not in str(error) or policy == "error":
                    raise
                count["overlong"] += 1
                continue
            retained.append(row)
            count["retained"] += 1
        write_rows(destination / (split + ".jsonl"), retained)
        report["splits"][split] = counts
        if base.get("qgpi", {}).get("enabled"):
            from .quality_basis import quality_spec
            for row in retained:
                spec = quality_spec(base, row, row["messages"], [])
                coverage = counts[row["task_id"]].setdefault("static_quality_coverage", {})
                for dim in spec["dimensions"]:
                    coverage[dim] = coverage.get(dim, 0) + 1
    (destination / "length_audit.json").write_text(json.dumps(report, indent=2))
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--policy", choices=["exclude", "error"], default="error")
    args = parser.parse_args()
    run(load_config(args.config), Path(args.data), Path(args.output), args.policy)


if __name__ == "__main__":
    main()
