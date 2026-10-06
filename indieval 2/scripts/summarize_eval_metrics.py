#!/usr/bin/env python3
"""Generate a tiered performance Markdown report from evaluation artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim_eval.errors import SimEvalError  # noqa: E402
from sim_eval.metric_markdown import write_metric_markdown  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read a formal/smoke evaluation result directory (or its summary JSON) "
            "and write a tiered Markdown report with primary scores, components, "
            "health checks, and a collapsed diagnostic appendix."
        )
    )
    parser.add_argument(
        "result_path",
        type=Path,
        help="evaluation result directory, suite_summary.json, or smoke_summary.json",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output Markdown path; default: <result_path>/metric_summary.md",
    )
    parser.add_argument(
        "--corrected-metrics",
        action="append",
        default=[],
        metavar="ENTRY=PATH",
        help=(
            "replace only one entry's aggregate metrics with an audited "
            "AgentSense/MirrorBench post-hoc corrected_metrics.json; repeatable"
        ),
    )
    parser.add_argument(
        "--replace-entry",
        action="append",
        default=[],
        metavar="ENTRY=PATH",
        help=(
            "replace one complete entry (records, status, and metrics) from another "
            "suite/result directory or summary JSON; repeatable"
        ),
    )
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
        corrected_metrics = _entry_path_arguments(
            args.corrected_metrics,
            "--corrected-metrics",
        )
        replacement_entries = _entry_path_arguments(
            args.replace_entry,
            "--replace-entry",
        )
        output = write_metric_markdown(
            input_path=args.result_path,
            output_path=args.output,
            corrected_metrics=corrected_metrics,
            replacement_entries=replacement_entries,
        )
    except (SimEvalError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
