"""Unified command-line entrypoint for offline evaluation workflows."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from .artifacts import load_json
from .backends import registered_backends
from .benchmarks import register_builtin_adapters
from .catalog import REQUESTED_BENCHMARK_IDS, BenchmarkCatalog, load_and_validate, load_catalog
from .data.commands import _safe_output, probe_import, sample_fixture_suite, source_status
from .data.loaders import HUMANUAL_DOMAINS, load_fixture_suite, load_import_spec
from .errors import ConfigurationError, SimEvalError
from .external_runner import run_tau_usi_import, run_userlm_import
from .generic_runner import GENERIC_LIVE_BENCHMARK_IDS, run_generic_import
from .local_env import configure_resource_cache_defaults, load_local_env
from .registry import get_adapter, registered_adapters
from .reporting import write_suite_report
from .runtime_config import GLOBAL_EVAL_MODEL_CHOICES, load_global_eval_model_presets
from .runner import run_fixture_suite
from .smoke import DEFAULT_FORMAL_CONFIG, DEFAULT_SMOKE_CONFIG, build_smoke_plan, run_smoke_suite


DEFAULT_CATALOG = str(Path(__file__).resolve().parent / "resources/benchmarks.json")
DEFAULT_SAMPLING = str(Path(__file__).resolve().parent / "resources/sampling_profiles.json")
DEFAULT_FIXTURES = str(Path(__file__).resolve().parent / "resources/fixtures")
DEFAULT_FORMAL_OUTPUT_ROOT = Path("artifacts/formal")


def _catalog_validate(args: argparse.Namespace) -> int:
    catalog, sampling = load_and_validate(args.catalog, args.sampling)
    payload = {
        "status": "valid",
        "catalog_revision": catalog.catalog_revision,
        "benchmark_count": len(catalog.benchmarks),
        "benchmark_ids": sorted(catalog.benchmarks),
        "sampling_profiles": sorted(sampling.profiles),
        "backend_names": list(registered_backends()),
        "catalog_path": str(Path(args.catalog)),
        "sampling_path": str(Path(args.sampling)),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _data_status(args: argparse.Namespace) -> int:
    print(json.dumps(source_status(args.catalog), ensure_ascii=False, indent=2))
    return 0


def _data_probe(args: argparse.Namespace) -> int:
    print(json.dumps(probe_import(args.manifest), ensure_ascii=False, indent=2))
    return 0


def _sample(args: argparse.Namespace) -> int:
    result = sample_fixture_suite(
        profile=args.profile,
        fixture_directory=args.fixtures,
        output_directory=args.output,
        catalog_path=args.catalog,
        sampling_path=args.sampling,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _resolve_benchmark_ids(raw: str | None, catalog: BenchmarkCatalog) -> list[str]:
    if raw is None or not raw.strip():
        return sorted(catalog.benchmarks)
    values = list(dict.fromkeys(value.strip() for value in raw.split(",") if value.strip()))
    if not values:
        raise ConfigurationError("--benchmarks must name at least one benchmark")
    unknown = set(values) - set(catalog.benchmarks)
    if unknown:
        raise ConfigurationError(f"unknown benchmark IDs: {sorted(unknown)}")
    return values


def _run(args: argparse.Namespace) -> int:
    catalog = load_catalog(args.catalog)
    benchmarks = _resolve_benchmark_ids(args.benchmarks, catalog)
    result = run_fixture_suite(
        benchmark_ids=benchmarks,
        profile=args.profile,
        backend_name=args.backend,
        output_directory=args.output,
        fixture_directory=args.fixtures,
        catalog_path=args.catalog,
        sampling_path=args.sampling,
        model=args.model,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    imported_benchmark = load_import_spec(args.manifest).benchmark_id
    if imported_benchmark != args.benchmark:
        raise ConfigurationError(
            f"--benchmark={args.benchmark!r} does not match import manifest benchmark_id={imported_benchmark!r}"
        )
    if args.benchmark == "tau_usi":
        runner = run_tau_usi_import
    elif args.benchmark == "userlm":
        runner = run_userlm_import
    else:
        runner = run_generic_import
    direct = {
        name: getattr(args, name) for name in ("model", "model_revision", "base_url", "max_workers")
        if getattr(args, name, None) is not None
    }
    extra: dict[str, Any] = {"evaluate_overrides": direct} if direct else {}
    if args.humanual_domain is not None:
        if args.benchmark != "humanual":
            raise ConfigurationError("--humanual-domain is only valid for --benchmark humanual")
        extra["humanual_domain"] = args.humanual_domain
    result = runner(
        **extra,
        manifest_path=args.manifest,
        runtime_config_path=args.runtime_config,
        output_directory=args.output,
        catalog_path=args.catalog,
        seed=args.seed,
        limit=args.limit,
        validate_only=args.validate_only,
        global_eval_model=args.global_eval_model,
        evaluated_model_config=args.evaluated_model_config,
        progress=(
            (lambda message: print(message, file=sys.stderr, flush=True))
            if not args.validate_only
            else None
        ),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.require_live_ready:
        if not args.validate_only:
            raise ConfigurationError("--require-live-ready is only valid with --validate-only")
        return 0 if result.get("live_ready") is True else 1
    return 0


def _smoke(args: argparse.Namespace) -> int:
    selected_entries = None
    if args.entries:
        selected_entries = tuple(
            dict.fromkeys(value.strip() for value in args.entries.split(",") if value.strip())
        )
        if not selected_entries:
            raise ConfigurationError("--entries must name at least one smoke entry")
    result = run_smoke_suite(
        config_path=args.config,
        output_directory=args.output,
        plan_only=args.plan_only,
        selected_entries=selected_entries,
        global_eval_model=args.global_eval_model,
        progress=(lambda message: print(message, file=sys.stderr)) if not args.plan_only else None,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") in {"valid", "valid_with_partial_coverage", "completed"} else 1


def _artifact_slug(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "unknown").strip())
    return text.strip("-._") or "unknown"


def _formal_output_path(
    *,
    config: str | Path,
    output_root: str | Path,
    selected_entries: tuple[str, ...] | None,
    global_eval_model: str | None,
) -> Path:
    plan = build_smoke_plan(
        config,
        selected_entries=selected_entries,
        global_eval_model=global_eval_model,
    )
    if plan.get("suite_mode") != "formal":
        raise ConfigurationError("the suite command requires a config with suite_mode: formal")
    candidate = _artifact_slug(plan.get("model", {}).get("model"))
    selected_support = plan.get("global_eval_model")
    if selected_support is None:
        support = "same-as-candidate"
    else:
        support = load_global_eval_model_presets()[str(selected_support)].get("model")
    date = datetime.now().astimezone().strftime("%Y-%m-%d")
    return Path(output_root) / f"{candidate}__eval-{_artifact_slug(support)}__{date}"


def _suite(args: argparse.Namespace) -> int:
    selected_entries = None
    if args.entries:
        selected_entries = tuple(
            dict.fromkeys(value.strip() for value in args.entries.split(",") if value.strip())
        )
        if not selected_entries:
            raise ConfigurationError("--entries must name at least one formal-suite entry")

    output = Path(args.output) if args.output else _formal_output_path(
        config=args.config,
        output_root=args.output_root,
        selected_entries=selected_entries,
        global_eval_model=args.global_eval_model,
    )
    if not args.plan_only:
        print(f"formal suite output: {output}", file=sys.stderr, flush=True)

    log_lock = threading.Lock()
    log_path = output / "suite.log"

    def progress(message: str) -> None:
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        line = f"{timestamp} | {message}"
        with log_lock:
            print(line, file=sys.stderr, flush=True)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")

    result = run_smoke_suite(
        config_path=args.config,
        output_directory=None if args.plan_only else output,
        plan_only=args.plan_only,
        selected_entries=selected_entries,
        global_eval_model=args.global_eval_model,
        progress=None if args.plan_only else progress,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") in {"valid", "valid_with_partial_coverage", "completed"} else 1


def _resume(args: argparse.Namespace) -> int:
    output = _safe_output(args.output)
    summary_path = output / "suite_summary.json"
    if not summary_path.is_file():
        raise ConfigurationError(
            f"cannot resume because suite summary is missing: {summary_path}; use `sim_eval run` first"
        )
    existing = load_json(summary_path)
    if not isinstance(existing, Mapping):
        raise ConfigurationError("existing suite_summary.json must contain an object")
    catalog = load_catalog(args.catalog)
    stored_ids = existing.get("requested_benchmark_ids")
    if not isinstance(stored_ids, list) or not all(isinstance(value, str) for value in stored_ids):
        entries = existing.get("benchmarks")
        if not isinstance(entries, Mapping):
            raise ConfigurationError("existing suite summary has no valid benchmark selection")
        stored_ids = list(entries)
    requested_ids = _resolve_benchmark_ids(args.benchmarks, catalog) if args.benchmarks else list(stored_ids)
    mismatches: list[str] = []
    for field, requested in (("profile", args.profile), ("backend", args.backend), ("model", args.model)):
        if existing.get(field) != requested:
            mismatches.append(f"{field}: existing={existing.get(field)!r}, requested={requested!r}")
    if set(requested_ids) != set(stored_ids):
        mismatches.append(
            f"benchmarks: existing={sorted(stored_ids)!r}, requested={sorted(requested_ids)!r}"
        )
    if mismatches:
        raise ConfigurationError(
            "refusing incompatible resume; " + "; ".join(mismatches) + ". Use a new --output directory."
        )
    result = run_fixture_suite(
        benchmark_ids=list(stored_ids),
        profile=args.profile,
        backend_name=args.backend,
        output_directory=output,
        fixture_directory=args.fixtures,
        catalog_path=args.catalog,
        sampling_path=args.sampling,
        model=args.model,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _report(args: argparse.Namespace) -> int:
    result = write_suite_report(
        run_directory=args.run,
        output_path=args.output,
        catalog_path=args.catalog,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _list_benchmarks(args: argparse.Namespace) -> int:
    catalog = load_catalog(args.catalog)
    register_builtin_adapters()
    adapters = set(registered_adapters())
    rows = []
    for benchmark_id in sorted(catalog.benchmarks):
        spec = catalog.get(benchmark_id)
        rows.append(
            {
                "id": benchmark_id,
                "display_name": spec.display_name,
                "aliases": list(spec.raw.get("aliases") or ()),
                "protocol_family": spec.protocol_family,
                "access_status": spec.access.get("status"),
                "primary_metric": spec.official_protocol.get("primary_metric"),
                "adapter_registered": benchmark_id in adapters,
            }
        )
    print(
        json.dumps(
            {
                "schema_version": "1.0",
                "benchmark_count": len(rows),
                "benchmarks": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _doctor_check(check_id: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {
        "id": check_id,
        "status": "passed" if passed else "failed",
        "detail": detail,
    }


def _doctor(args: argparse.Namespace) -> int:
    checks: list[dict[str, Any]] = []
    catalog = None
    try:
        catalog, sampling = load_and_validate(args.catalog, args.sampling)
        checks.append(
            _doctor_check(
                "catalog_and_sampling",
                set(catalog.benchmarks) == set(REQUESTED_BENCHMARK_IDS),
                {
                    "catalog_revision": catalog.catalog_revision,
                    "benchmark_count": len(catalog.benchmarks),
                    "profiles": sorted(sampling.profiles),
                },
            )
        )
    except SimEvalError as exc:
        checks.append(_doctor_check("catalog_and_sampling", False, str(exc)))

    register_builtin_adapters()
    adapter_names = set(registered_adapters())
    checks.append(
        _doctor_check(
            "adapter_registry",
            adapter_names == set(REQUESTED_BENCHMARK_IDS),
            {"registered": sorted(adapter_names)},
        )
    )
    try:
        fixtures = load_fixture_suite(args.fixtures)
        errors: list[str] = []
        for benchmark_id in sorted(REQUESTED_BENCHMARK_IDS):
            cases = fixtures.get(benchmark_id, ([], None))[0]
            if not cases:
                errors.append(f"{benchmark_id}: no fixture cases")
                continue
            adapter = get_adapter(benchmark_id)
            for case in cases:
                try:
                    adapter.validate_case(case)
                except SimEvalError as exc:
                    errors.append(f"{benchmark_id}/{case.case_id}: {exc}")
        checks.append(
            _doctor_check(
                "fixture_contracts",
                not errors,
                {"benchmark_count": len(fixtures), "errors": errors},
            )
        )
    except SimEvalError as exc:
        checks.append(_doctor_check("fixture_contracts", False, str(exc)))

    backends = set(registered_backends())
    required_backends = {
        "replay",
        "openai",
        "openai_compatible",
        "chat_completions",
        "openai_responses",
        "vllm",
        "huggingface",
        "hf",
    }
    checks.append(
        _doctor_check(
            "backend_registry",
            required_backends <= backends,
            {"registered": sorted(backends)},
        )
    )
    checks.append(
        {
            "id": "optional_huggingface_dependencies",
            "status": "available" if importlib.util.find_spec("transformers") else "optional_missing",
            "detail": "Install `.[huggingface]` only for local Hugging Face execution; offline replay is unaffected.",
        }
    )
    failed = [item for item in checks if item["status"] == "failed"]
    payload = {
        "schema_version": "1.0",
        "status": "passed" if not failed else "failed",
        "offline": True,
        "network_calls": 0,
        "checks": checks,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if not failed else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sim_eval", description="Human-simulation benchmark evaluation framework")
    commands = parser.add_subparsers(dest="command", required=True)
    catalog = commands.add_parser("catalog", help="inspect or validate benchmark metadata")
    catalog_commands = catalog.add_subparsers(dest="catalog_command", required=True)
    validate = catalog_commands.add_parser("validate", help="validate catalog and sampling profiles")
    validate.add_argument("--catalog", required=True)
    validate.add_argument("--sampling", required=True)
    validate.set_defaults(handler=_catalog_validate)

    data = commands.add_parser("data", help="inspect acquisition state or probe a local import")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    status = data_commands.add_parser("status", help="show fail-closed acquisition status for every benchmark")
    status.add_argument("--catalog", default=DEFAULT_CATALOG)
    status.set_defaults(handler=_data_status)
    probe = data_commands.add_parser("probe", help="validate a local import manifest and normalized schema")
    probe.add_argument("--manifest", required=True)
    probe.set_defaults(handler=_data_probe)

    sample = commands.add_parser("sample", help="materialize deterministic sample manifests")
    sample.add_argument("--profile", required=True)
    sample.add_argument("--fixtures", required=True, help="repository-owned synthetic fixture directory")
    sample.add_argument("--output", required=True)
    sample.add_argument("--catalog", default=DEFAULT_CATALOG)
    sample.add_argument("--sampling", default=DEFAULT_SAMPLING)
    sample.set_defaults(handler=_sample)

    list_command = commands.add_parser("list", help="list benchmark protocols and availability")
    list_command.add_argument("--catalog", default=DEFAULT_CATALOG)
    list_command.set_defaults(handler=_list_benchmarks)

    doctor = commands.add_parser("doctor", help="run offline configuration and fixture diagnostics")
    doctor.add_argument("--catalog", default=DEFAULT_CATALOG)
    doctor.add_argument("--sampling", default=DEFAULT_SAMPLING)
    doctor.add_argument("--fixtures", default=DEFAULT_FIXTURES)
    doctor.set_defaults(handler=_doctor)

    run = commands.add_parser("run", help="execute a replay-backed fixture suite")
    run.add_argument(
        "--benchmarks",
        help="comma-separated canonical benchmark IDs; omit to run all catalog benchmarks",
    )
    run.add_argument("--profile", required=True)
    run.add_argument("--backend", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--fixtures", default=DEFAULT_FIXTURES)
    run.add_argument("--catalog", default=DEFAULT_CATALOG)
    run.add_argument("--sampling", default=DEFAULT_SAMPLING)
    run.add_argument("--model", default="offline-replay")
    run.set_defaults(handler=_run)

    evaluate = commands.add_parser(
        "evaluate",
        help="execute a normalized official/local benchmark import with a real backend",
    )
    evaluate.add_argument(
        "--benchmark",
        choices=tuple(sorted((*GENERIC_LIVE_BENCHMARK_IDS, "tau_usi", "userlm"))),
        required=True,
    )
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--runtime-config", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--catalog", default=DEFAULT_CATALOG)
    evaluate.add_argument("--seed", type=int, default=20260817)
    evaluate.add_argument(
        "--limit",
        type=int,
        help=(
            "deterministic diagnostic subset; for the generic runner this is a dependency-group limit, "
            "while tau-USI/UserLM retain their case-limit semantics; omit for the full import"
        ),
    )
    evaluate.add_argument(
        "--validate-only",
        action="store_true",
        help="validate all data, reference, schema, and runtime locks without calling a model endpoint",
    )
    evaluate.add_argument(
        "--require-live-ready",
        action="store_true",
        help=(
            "with --validate-only, also require resolved model routing and every mandatory "
            "local metric dependency; still performs no API or model call"
        ),
    )
    evaluate.add_argument(
        "--global-eval-model",
        "--global_eval_model",
        dest="global_eval_model",
        choices=GLOBAL_EVAL_MODEL_CHOICES,
        default=None,
        help=(
            "override every helper/partner/judge role with the selected preset while leaving "
            "the evaluated model role unchanged"
        ),
    )
    evaluate.add_argument(
        "--evaluated-model-config",
        help=(
            "reusable endpoint/model routing file for the evaluated role; benchmark-specific "
            "generation limits and prompt protocol remain in the runtime YAML"
        ),
    )
    evaluate.add_argument("--model", help="override only the evaluated model/LoRA name; defaults its revision to this name")
    evaluate.add_argument("--model-revision", help="explicit immutable evaluated checkpoint revision")
    evaluate.add_argument("--base-url", help="override only the evaluated model API base URL")
    evaluate.add_argument("--max-workers", type=int, help="override case concurrency without changing scoring or repetitions")
    evaluate.add_argument("--humanual-domain", choices=HUMANUAL_DOMAINS, help="evaluate only this domain of the original HUMANUAL collection")
    evaluate.set_defaults(handler=_evaluate)

    smoke = commands.add_parser(
        "smoke",
        help="run the sequential DeepSeek CPU metric-coverage diagnostic suite",
    )
    smoke.add_argument("--config", required=True)
    smoke.add_argument(
        "--output",
        help="artifact directory under indieval; required unless --plan-only is used",
    )
    smoke.add_argument(
        "--entries",
        help="optional comma-separated smoke entry IDs; omit to run all 14 entries",
    )
    smoke.add_argument(
        "--plan-only",
        action="store_true",
        help="validate data, coverage selection, runtime overlays, and local resources without API calls",
    )
    smoke.add_argument(
        "--global-eval-model",
        "--global_eval_model",
        dest="global_eval_model",
        choices=GLOBAL_EVAL_MODEL_CHOICES,
        default=None,
        help=(
            "preserve each base runtime's evaluated role and override every support role with "
            "Qwen, Deepseek, GPT, or DeepseekRelay; omit to keep the original DeepSeek-only smoke behavior"
        ),
    )
    smoke.set_defaults(handler=_smoke)

    suite = commands.add_parser(
        "suite",
        help="run a live-ready full-import benchmark suite with shared concurrency limits",
    )
    suite.add_argument("--config", required=True)
    suite.add_argument(
        "--output",
        help=(
            "explicit artifact directory; by default uses "
            "artifacts/formal/<candidate>__eval-<support-model>__<date>"
        ),
    )
    suite.add_argument("--output-root", default=str(DEFAULT_FORMAL_OUTPUT_ROOT))
    suite.add_argument(
        "--entries",
        help="optional comma-separated formal-suite entry IDs; omit to run all entries",
    )
    suite.add_argument(
        "--plan-only",
        action="store_true",
        help="validate the full data/config plan and report live-readiness without API calls",
    )
    suite.add_argument(
        "--global-eval-model",
        "--global_eval_model",
        dest="global_eval_model",
        choices=GLOBAL_EVAL_MODEL_CHOICES,
        default=None,
        help="override the formal config's support-model preset",
    )
    suite.set_defaults(handler=_suite)

    resume = commands.add_parser("resume", help="resume an identity-compatible checkpointed suite")
    resume.add_argument(
        "--benchmarks",
        help="optional comma-separated IDs; must match the existing suite selection",
    )
    resume.add_argument("--profile", default="offline_smoke")
    resume.add_argument("--backend", default="replay")
    resume.add_argument("--output", required=True)
    resume.add_argument("--fixtures", default=DEFAULT_FIXTURES)
    resume.add_argument("--catalog", default=DEFAULT_CATALOG)
    resume.add_argument("--sampling", default=DEFAULT_SAMPLING)
    resume.add_argument("--model", default="offline-replay")
    resume.set_defaults(handler=_resume)

    report = commands.add_parser("report", help="render benchmark-native metrics from suite artifacts")
    report.add_argument("--run", required=True, help="suite artifact directory")
    report.add_argument("--output", required=True, help="Markdown report path")
    report.add_argument("--catalog", default=DEFAULT_CATALOG)
    report.set_defaults(handler=_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    load_local_env()
    configure_resource_cache_defaults()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except SimEvalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
