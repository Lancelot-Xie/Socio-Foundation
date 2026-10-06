#!/usr/bin/env python3
"""Write detailed and headline F/S/U/T/N comparison tables."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim_eval.capability_comparison import DEFAULT_MAPPING, write_capability_comparison
from sim_eval.errors import SimEvalError


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path, help="baseline full suite directory or summary JSON")
    parser.add_argument("--candidate", required=True, type=Path, help="Candidate full suite directory or summary JSON")
    parser.add_argument("--baseline-label", default="Baseline")
    parser.add_argument("--candidate-label", default="Candidate")
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--digits", type=int, default=4, choices=range(1, 13))
    parser.add_argument("--allow-missing-summary", action="store_true",
                        help="if a root summary is absent, compare planned child summaries without modifying the run")
    args = parser.parse_args(argv)
    try:
        outputs = write_capability_comparison(baseline_path=args.baseline, candidate_path=args.candidate,
            baseline_label=args.baseline_label, candidate_label=args.candidate_label,
            mapping_path=args.mapping, output_dir=args.output_dir, digits=args.digits,
            allow_missing_summary=args.allow_missing_summary)
    except (SimEvalError, OSError, ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
