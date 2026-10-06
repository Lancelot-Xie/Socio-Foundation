import argparse
import json

from .config import load_config


def main():
    parser = argparse.ArgumentParser(description="Simulation standalone multi-teacher OPD")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("train", "validate-config", "build-demos", "build-corrections", "calibrate", "evaluate"):
        p = sub.add_parser(command)
        p.add_argument("--config", required=True)
        if command in ("build-demos", "build-corrections", "calibrate", "evaluate"):
            p.add_argument("--output", required=True)
        if command in ("build-demos", "build-corrections"):
            p.add_argument("--candidates", type=int, default=1)
        if command == "calibrate":
            p.add_argument("--repeats", type=int, default=2)
            p.add_argument("--min-samples", type=int, default=8)
        if command == "evaluate":
            p.add_argument("--teacher")
            p.add_argument("--model", help="Explicit checkpoint; otherwise evaluate output_dir/final when available")
        if command == "train":
            p.add_argument("--resume")
    p = sub.add_parser("prepare")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--task-id")
    p.add_argument("--dimensions", nargs="+")
    p.add_argument("--messages-field", default="messages")
    p = sub.add_parser("merge-adapter")
    p.add_argument("--base", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--dtype", default="float32")
    p = sub.add_parser("make-smoke-fixture")
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "train":
        from .trainer import train
        cfg = load_config(args.config)
        if args.resume:
            from pathlib import Path
            cfg["train"]["resume_from"] = str(Path(args.resume).resolve())
        result = train(cfg)
    elif args.command == "make-smoke-fixture":
        from .smoke import make_fixture
        result = make_fixture(args.output)
    else:
        from . import workflows
        if args.command == "prepare":
            result = workflows.prepare_data(args.input, args.output, args.task_id, args.dimensions, args.messages_field)
        elif args.command == "merge-adapter":
            result = workflows.merge_adapter(args.base, args.adapter, args.output, args.dtype)
        else:
            cfg = load_config(args.config)
            if args.command == "validate-config":
                result = workflows.inspect_config(cfg)
            elif args.command == "build-demos":
                result = workflows.build_demos(cfg, args.output, args.candidates)
            elif args.command == "build-corrections":
                from .corrections import build_corrections
                result = build_corrections(cfg, args.output, args.candidates)
            elif args.command == "calibrate":
                result = workflows.calibrate(cfg, args.output, args.repeats, args.min_samples)
            else:
                result = workflows.evaluate_model(cfg, args.output, args.teacher, args.model)
    print(json.dumps(result, indent=2, ensure_ascii=False))
