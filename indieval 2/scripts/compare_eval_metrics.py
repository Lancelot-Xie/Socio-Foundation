#!/usr/bin/env python3
"""Compare exact benchmark entries across one baseline and one or more runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim_eval.errors import SimEvalError  # noqa: E402
from sim_eval.metric_comparison import (  # noqa: E402
    discover_latest_candidate_paths,
    write_metric_comparison,
    write_three_way_metric_comparison,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Match candidate entries to a baseline suite by exact entry ID and write "
            "a Markdown comparison containing primary and all aggregate metrics. "
            "Optionally add one full-suite third run for a three-system comparison; "
            "three-way reports cover the union of all entries and mark an entirely "
            "missing model entry with a backslash."
        )
    )
    parser.add_argument("--baseline", type=Path, required=True)
    candidate_source = parser.add_mutually_exclusive_group(required=True)
    candidate_source.add_argument(
        "--candidate",
        type=Path,
        action="append",
        help="candidate result directory; repeat once per single- or multi-entry run",
    )
    candidate_source.add_argument(
        "--candidate-root",
        type=Path,
        help="scan immediate child directories and select the newest artifact per entry",
    )
    parser.add_argument(
        "--candidate-glob",
        default="*",
        help="directory glob used with --candidate-root",
    )
    parser.add_argument("--baseline-label", default="Baseline")
    parser.add_argument("--candidate-label", default="Candidate")
    parser.add_argument(
        "--baseline-entry",
        action="append",
        default=[],
        metavar="ENTRY=PATH",
        help=(
            "replace one baseline entry from another result artifact; repeatable"
        ),
    )
    parser.add_argument(
        "--candidate-entry",
        action="append",
        default=[],
        metavar="ENTRY=PATH",
        help=(
            "replace one auto-discovered/explicit candidate entry from another "
            "result artifact; repeatable"
        ),
    )
    parser.add_argument(
        "--third-run",
        type=Path,
        help=(
            "optional full-suite result directory to compare beside the baseline "
            "and candidate collection"
        ),
    )
    parser.add_argument("--third-label", default="Unified-8B")
    parser.add_argument("-o", "--output", type=Path, required=True)
    return parser


def _entry_path_arguments(values: list[str], option: str) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"{option} must use ENTRY=PATH, got {raw!r}")
        entry_id, path = raw.split("=", 1)
        entry_id = entry_id.strip()
        path = path.strip()
        if not entry_id or not path:
            raise ValueError(f"{option} must use non-empty ENTRY=PATH, got {raw!r}")
        if entry_id in parsed:
            raise ValueError(f"{option} repeats entry {entry_id!r}")
        parsed[entry_id] = Path(path)
    return parsed


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        baseline_entry_overrides = _entry_path_arguments(
            args.baseline_entry,
            "--baseline-entry",
        )
        candidate_entry_overrides = _entry_path_arguments(
            args.candidate_entry,
            "--candidate-entry",
        )
        candidate_paths = args.candidate
        if args.candidate_root is not None:
            candidate_paths = discover_latest_candidate_paths(
                args.candidate_root,
                glob_pattern=args.candidate_glob,
            )
            for path in candidate_paths:
                print(f"selected candidate: {path}", file=sys.stderr)
        if args.third_run is None:
            output = write_metric_comparison(
                baseline_path=args.baseline,
                candidate_paths=candidate_paths,
                output_path=args.output,
                baseline_label=args.baseline_label,
                candidate_label=args.candidate_label,
                baseline_entry_overrides=baseline_entry_overrides,
                candidate_entry_overrides=candidate_entry_overrides,
            )
        else:
            output = write_three_way_metric_comparison(
                baseline_path=args.baseline,
                candidate_paths=candidate_paths,
                third_path=args.third_run,
                output_path=args.output,
                baseline_label=args.baseline_label,
                candidate_label=args.candidate_label,
                third_label=args.third_label,
                baseline_entry_overrides=baseline_entry_overrides,
                candidate_entry_overrides=candidate_entry_overrides,
            )
    except (SimEvalError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
