#!/usr/bin/env python3
"""Batch headline metrics by F/S/U/T/N for available formal-run metrics."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim_eval.capability_batch import DEFAULT_MAPPING, write_batch_report
from sim_eval.errors import SimEvalError


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "artifacts/formal")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/comparisons/formal_capability_summary")
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--selection", choices=("by-type", "one-per-dataset"), default="by-type")
    parser.add_argument("--digits", type=int, choices=range(1, 13), default=4)
    args = parser.parse_args(argv)
    try:
        report, paths = write_batch_report(root=args.root, output_dir=args.output_dir,
            mapping_path=args.mapping, selection=args.selection, digits=args.digits)
    except (SimEvalError, OSError, ValueError, TypeError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"纳入 {len(report['runs'])} 个 run；跳过 {len(report['skipped'])} 个目录。")
    for run in report["skipped"]:
        print(f"[跳过] {run['directory']}: {run['reason']}")
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0 if report["runs"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
