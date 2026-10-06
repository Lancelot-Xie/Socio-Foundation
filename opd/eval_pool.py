"""Single-node persistent GPU pool for independent base/task-LoRA baselines.

The coordinator alone schedules work and commits reports. Workers never touch
stages.json, never share a mutable PEFT model, and write only assigned chunks.
No training models, optimizer or distributed process groups live in this pool.
"""

import argparse
import contextlib
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import signal
import sys
import time
import traceback
from collections import defaultdict, deque

from .checkpoints import archive_incomplete, atomic_json
from .data import load_data, read_rows, write_rows

SCHEMA = "opd-baseline-task-pool-v2"


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def artifact_config(cfg):
    """Configuration that can change generated responses or their evaluation.

    Optimizer, training microbatch and checkpoint cadence are deliberately not
    trajectory dependencies.
    """
    model_fields = ("base_model", "tokenizer", "dtype", "trust_remote_code", "chat_template_kwargs")
    return {
        "seed": cfg["seed"],
        "model": {k: cfg["model"].get(k) for k in model_fields},
        "teacher": cfg["teacher"], "teachers": cfg["teachers"],
        "routing": cfg["routing"], "rollout": cfg["rollout"],
        "inference": cfg["inference"], "quality": cfg["quality"],
        "qgpi": cfg["qgpi"], "dimension": cfg["data"].get("dimension"),
    }


def validate_pool(cfg, workers, chunk_size, gpus=None):
    if type(workers) is not int or workers < 1 or type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("Pool workers and chunk size must be positive integers")
    if gpus is not None and (type(gpus) is not int or not 1 <= gpus <= workers):
        raise ValueError("Pool GPU count must be a positive integer no greater than total workers")
    if cfg["rollout"]["environment_factory"] or cfg["rollout"]["max_turns"] != 1:
        raise ValueError("Task pool supports independent single-turn baselines only; use baseline_scheduler=distributed")
    if cfg["model"]["dtype"] != cfg["teacher"]["dtype"]:
        raise ValueError("Shared baseline backbone requires identical student/teacher dtype; use distributed scheduler")
    if any("adapter" not in t for t in cfg["teachers"]):
        raise ValueError("Task pool requires common-base LoRA teachers; full-model teachers use distributed scheduler")
    if cfg["teacher"]["device"] not in ("auto", "cpu"):
        raise ValueError("Task pool requires teacher.device=auto (or cpu for CPU checks)")


def prepare_jobs(cfg, requests, chunk_size):
    """Round-robin task/version chunks; teacher jobs precede base jobs initially."""
    if not requests or len({r["name"] for r in requests}) != len(requests):
        raise ValueError("Baseline requests must be nonempty with unique names")
    if len({str(Path(r["output"]).resolve()) for r in requests}) != len(requests):
        raise ValueError("Baseline requests must have different output files")
    teachers = {t["id"] for t in cfg["teachers"]}
    evaluations, groups = [], []
    for i, request in enumerate(requests):
        teacher = request.get("teacher")
        mode = request.get("mode", "evaluation")
        if mode not in ("evaluation", "demonstrations"):
            raise ValueError("Unknown pool request mode")
        candidates = request.get("candidates", 1)
        if mode == "demonstrations" and (teacher is None or type(candidates) is not int or candidates < 1):
            raise ValueError("Demonstrations require a teacher and positive candidates")
        if teacher is not None and teacher not in teachers:
            raise ValueError("Unknown pool teacher: " + str(teacher))
        if request.get("model") not in (None, cfg["model"]["base_model"]) or (teacher and request.get("model")):
            raise ValueError("Pool baselines evaluate only the common base or its registered task adapters")
        rows = load_data(request["eval_file"], cfg["data"]["dimension"])
        task_rows = defaultdict(list)
        for row in rows:
            task_rows[row["task_id"]].append(row)
        evaluation = {"key": f"eval_{i:04d}", "request": request, "rows": rows, "jobs": []}
        evaluation["signature"] = signature({"schema": SCHEMA, "config": artifact_config(cfg), "request": request, "rows": rows,
                                             "chunk_size": chunk_size})
        for task_index, (task, subset) in enumerate(sorted(task_rows.items())):
            chunks = []
            for start in range(0, len(subset), chunk_size):
                key = f"eval_{i:04d}_task_{task_index:04d}_chunk_{start // chunk_size:06d}"
                job = {"key": key, "evaluation": evaluation["key"], "name": request["name"], "task": task,
                       "teacher": teacher, "rows": subset[start:start + chunk_size]}
                if mode == "demonstrations":
                    if teacher not in cfg["routing"]["tasks"][task]["teachers"]:
                        raise ValueError("Demonstration teacher is not compatible with task")
                    job.update(mode=mode, candidates=candidates)
                job["signature"] = signature({"evaluation": evaluation["signature"], "key": key, "rows": job["rows"]})
                chunks.append(job)
                evaluation["jobs"].append(job)
            groups.append((teacher is None, chunks))
        evaluations.append(evaluation)
    ordered = []
    for base in (False, True):
        by_task = defaultdict(deque)
        for is_base, chunks in groups:
            if is_base == base:
                by_task[chunks[0]["task"]].append(chunks)
        # Give distinct tasks their first worker before queueing second versions.
        while any(by_task.values()):
            for versions in by_task.values():
                if versions:
                    ordered.append(versions.popleft())
    jobs = [chunks[index] for index in range(max(map(len, ordered), default=0))
            for chunks in ordered if index < len(chunks)]
    return evaluations, jobs


class BaselineCache:
    """One frozen backbone per worker; all requested LoRAs share its storage."""
    def __init__(self, cfg):
        from .models import TeacherBank, load_causal, load_tokenizer
        from .workflows import inference_device
        self.device, self.tokenizer = inference_device(), load_tokenizer(cfg)
        self.bank = TeacherBank(cfg, self.tokenizer, self.device)
        if self.bank.shared is None:
            self.base = load_causal(cfg["model"]["base_model"], cfg["model"]["dtype"],
                                    cfg["model"]["trust_remote_code"]).to(self.device).requires_grad_(False).eval()
        else:
            self.base = self.bank.shared
            self.device = self.bank.device
        self.stats = {"backbone_loads": 1, "adapters": len(self.bank.adapter_names), "pid": os.getpid(),
                      "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "device": str(self.device)}

    @contextlib.contextmanager
    def policy(self, teacher):
        if teacher is not None:
            yield self.bank.activate(teacher)
        elif self.bank.shared is not None:
            # This context restores the previously active adapter after base evaluation.
            with self.bank.shared.disable_adapter():
                yield self.bank.shared
        else:
            yield self.base


def chunk_paths(root, job):
    folder = root / "chunks" / job["key"]
    return folder, folder / "responses.jsonl", folder / "complete.json"


def valid_chunk(root, job):
    _, records, marker = chunk_paths(root, job)
    try:
        saved = json.loads(marker.read_text())
        return (saved["signature"] == job["signature"] and saved["rows"] == len(job["rows"]) and
                saved["sha256"] == file_hash(records) and
                [r["id"] for r in read_rows(records)] == [r["id"] for r in job["rows"]])
    except (OSError, ValueError, KeyError, TypeError):
        return False


def compute_chunk(cache, cfg, job, root, worker):
    from accelerate.utils import set_seed
    from .workflows import evaluate_rows
    folder, _, _ = chunk_paths(root, job)
    temporary = folder.with_name("." + folder.name + ".pending")
    for path in (folder, temporary):
        if path.exists():
            archive_incomplete(path)
    temporary.mkdir(parents=True)
    started, started_at = time.monotonic(), time.time()
    set_seed(cfg["seed"])  # Greedy generation; no dependence on worker assignment order.
    with cache.policy(job["teacher"]) as model:
        if job.get("mode") == "demonstrations":
            from .dimension_workflow import generate_candidates
            set_seed((cfg["seed"] + int(job["signature"][:8], 16)) % 2**32)
            records = generate_candidates(cfg, model, cache.tokenizer, job, cache.device)
        else:
            records = evaluate_rows(cfg, model, cache.tokenizer, job["rows"], cache.device)
    if [r["id"] for r in records] != [r["id"] for r in job["rows"]]:
        raise ValueError("Pool worker returned missing/reordered rows")
    write_rows(temporary / "responses.jsonl", records)
    atomic_json(temporary / "complete.json", {"signature": job["signature"], "rows": len(records),
        "sha256": file_hash(temporary / "responses.jsonl"), "worker": worker,
        "elapsed_seconds": time.monotonic() - started, "teacher": job["teacher"], "task": job["task"],
        "pid": cache.stats["pid"], "visible_devices": cache.stats["visible_devices"],
        "started_at_unix": started_at, "finished_at_unix": time.time()})
    temporary.rename(folder)


def report_paths(evaluation):
    output = Path(evaluation["request"]["output"])
    return [output, Path(str(output) + ".summary.json"), Path(str(output) + ".model.json")]


def valid_report(root, evaluation):
    try:
        marker = json.loads((root / "evaluations" / (evaluation["key"] + ".json")).read_text())
        return marker["signature"] == evaluation["signature"] and marker["hashes"] == [
            file_hash(p) for p in report_paths(evaluation)]
    except (OSError, ValueError, KeyError, TypeError):
        return False


def finish_evaluation(cfg, root, evaluation, workers, gpus, runtime_workers=None):
    from .workflows import summarize_evaluation
    records, chunks = [], []
    for job in evaluation["jobs"]:
        if not valid_chunk(root, job):
            raise ValueError("Cannot finalize an incomplete/corrupt chunk: " + job["key"])
        _, path, marker = chunk_paths(root, job)
        records.extend(read_rows(path))
        chunks.append(json.loads(marker.read_text()))
    order = {r["id"]: i for i, r in enumerate(evaluation["rows"])}
    if len(records) != len(order) or {r["id"] for r in records} != set(order):
        raise ValueError("Merged baseline has missing/duplicate records")
    records.sort(key=lambda r: order[r["id"]])
    paths = report_paths(evaluation)
    temporary = root / "report_staging" / evaluation["key"]
    if temporary.exists():
        archive_incomplete(temporary)
    temporary.mkdir(parents=True)
    staged = [temporary / name for name in ("responses.jsonl", "summary.json", "model.json")]
    write_rows(staged[0], records)
    summary = ({"rows": len(records), "mode": "demonstrations", "candidates": sum(len(r["candidates"]) for r in records)}
               if evaluation["request"].get("mode") == "demonstrations" else summarize_evaluation(cfg, records))
    atomic_json(staged[1], summary)  # Global F1 for evaluations, never mean chunk/rank F1.
    atomic_json(staged[2], {"teacher_id": evaluation["request"].get("teacher"),
        "student_checkpoint": None if evaluation["request"].get("teacher") else cfg["model"]["base_model"],
        "scheduler": "task_pool", "pool_workers": workers, "worker_ids": sorted({c["worker"] for c in chunks}),
        "pool_gpus": gpus, "max_workers_per_gpu": (workers + gpus - 1) // gpus,
        # A resumed report can contain chunks from older, wider pools. Preserve
        # configured counts above and record the current invocation's limit here.
        "runtime_worker_limit": workers if runtime_workers is None else runtime_workers,
        "runtime_max_workers_per_gpu": ((workers if runtime_workers is None else runtime_workers) + gpus - 1) // gpus,
        "batch_size_per_worker": cfg["inference"]["batch_size"], "chunks": len(chunks),
        "worker_seconds": sum(c["elapsed_seconds"] for c in chunks)})
    for source, destination in zip(staged, paths, strict=True):
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            archive_incomplete(destination)
        source.replace(destination)
    atomic_json(root / "evaluations" / (evaluation["key"] + ".json"), {
        "signature": evaluation["signature"], "hashes": [file_hash(p) for p in paths]})
    print(f"[pool eval-done] {evaluation['request']['name']} rows={len(records)}", flush=True)


def _worker(cfg, root, worker, visible, inbox, events, parent_pid):
    # Spawn, never fork an initialized CUDA process. Bind before importing Torch.
    if visible is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        os.environ.pop(name, None)
    if sys.platform.startswith("linux"):
        import ctypes
        if ctypes.CDLL(None).prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
            raise RuntimeError("Could not set worker parent-death signal")
        if os.getppid() != parent_pid:
            return
    try:
        cache = BaselineCache(cfg)
        events.put({"event": "ready", "worker": worker, "stats": cache.stats})
        while True:
            job = inbox.get()
            if job is None:
                return
            compute_chunk(cache, cfg, job, Path(root), worker)
            events.put({"event": "done", "worker": worker, "key": job["key"]})
    except BaseException:
        events.put({"event": "error", "worker": worker, "traceback": traceback.format_exc()})
        raise


def worker_devices(count, gpus=None):
    """Spread across selected GPUs, then colocate independent processes evenly.

    Eight GPUs with sixteen jobs map to 0,0,1,1,...,7,7. With only ten
    jobs they map to 0,0,1,1,2,3,4,5,6,7, so no selected GPU is stranded.
    """
    gpus = count if gpus is None else gpus
    if type(count) is not int or count < 1 or type(gpus) is not int or gpus < 1:
        raise ValueError("Worker and GPU counts must be positive integers")
    cpu = os.environ.get("ACCELERATE_USE_CPU", "").lower() in ("true", "1", "yes")
    if cpu:
        return [None] * count
    import torch
    available = torch.cuda.device_count()
    if gpus > available:
        raise ValueError(f"Requested {gpus} pool GPUs but only {available} visible GPUs")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = visible.split(",") if visible else list(map(str, range(available)))
    active = min(count, gpus)
    return [devices[index].strip() for index in range(active)
            for _ in range(count // active + int(index < count % active))]


def runtime_worker_limit(workers, gpus):
    """Reduce colocated model processes without changing saved recipe identity.

    Job RNG is seeded from its original signature, independently of worker ID.
    Keep generation batches, chunks and configured counts intact on resume.
    """
    value = os.environ.get("OPD_POOL_MAX_WORKERS_PER_GPU")
    if value is None:
        return workers
    try:
        cap = int(value)
    except ValueError:
        raise ValueError("OPD_POOL_MAX_WORKERS_PER_GPU must be a positive integer") from None
    if cap < 1:
        raise ValueError("OPD_POOL_MAX_WORKERS_PER_GPU must be a positive integer")
    return min(workers, gpus * cap)


def run_pool(cfg, requests, output, workers=8, chunk_size=32, resume=False, in_process=False, gpus=None):
    gpus = workers if gpus is None else gpus
    validate_pool(cfg, workers, chunk_size, gpus)
    runtime_workers = runtime_worker_limit(workers, gpus)
    root = Path(output)
    evaluations, jobs = prepare_jobs(cfg, requests, chunk_size)
    key = signature({"schema": SCHEMA, "evaluations": [e["signature"] for e in evaluations], "workers": workers, "gpus": gpus})
    manifest = root / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text())["signature"] != key:
            raise ValueError("Pool config/data changed; use a new output directory")
        if not resume:
            raise FileExistsError("Pool exists; use --resume")
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(manifest, {"signature": key, "schema": SCHEMA, "evaluations": [e["request"] for e in evaluations]})
    finished = {e["key"] for e in evaluations if valid_report(root, e)}
    completed = {j["key"] for j in jobs if j["evaluation"] in finished or valid_chunk(root, j)}
    todo = deque(j for j in jobs if j["key"] not in completed)
    reused = len(completed)
    launched = min(runtime_workers, len(todo))
    started = time.monotonic()
    stats = {}
    print(f"[pool] evaluations={len(evaluations)} chunks={len(jobs)} retained_chunks={reused} "
          f"gpus={gpus} workers={launched}/{workers} runtime_worker_limit={runtime_workers} "
          f"max_workers_per_gpu={(runtime_workers + gpus - 1) // gpus} "
          f"batch_per_worker={cfg['inference']['batch_size']}", flush=True)

    def finalize_ready():
        for evaluation in evaluations:
            if evaluation["key"] not in finished and all(j["key"] in completed for j in evaluation["jobs"]):
                finish_evaluation(cfg, root, evaluation, workers, gpus, runtime_workers)
                finished.add(evaluation["key"])

    processes, inboxes, events = [], [], None
    try:
        finalize_ready()
        if todo and in_process:
            if workers != 1 or os.environ.get("ACCELERATE_USE_CPU", "").lower() not in ("true", "1", "yes"):
                raise ValueError("In-process pool is restricted to one CPU smoke worker")
            cache = BaselineCache(cfg)
            stats[0] = cache.stats
            for job in todo:
                compute_chunk(cache, cfg, job, root, 0)
                completed.add(job["key"])
                finalize_ready()
        elif todo:
            devices = worker_devices(launched, gpus)
            ctx = mp.get_context("spawn")
            events = ctx.Queue()
            # Each worker gets only one in-flight chunk, not the entire data queue.
            for worker, visible in enumerate(devices):
                inbox = ctx.Queue()
                process = ctx.Process(target=_worker, args=(cfg, str(root), worker, visible, inbox, events, os.getpid()))
                process.start()
                inboxes.append(inbox)
                processes.append(process)
            busy, ready = {}, set()

            def dispatch(worker):
                if todo:
                    job = todo.popleft()
                    busy[worker] = job
                    print(f"[pool run] worker={worker} visible_gpu={devices[worker]} "
                          f"eval={job['name']} task={job['task']} rows={len(job['rows'])}", flush=True)
                    inboxes[worker].put(job)
                else:
                    busy.pop(worker, None)
                    inboxes[worker].put(None)

            while len(completed) < len(jobs):
                try:
                    event = events.get(timeout=.25)
                except queue.Empty:
                    for worker, process in enumerate(processes):
                        if process.exitcode is not None and (process.exitcode != 0 or worker in busy or worker not in ready):
                            raise RuntimeError(f"Pool worker {worker} exited unexpectedly ({process.exitcode}); completed chunks retained")
                    continue
                worker = event["worker"]
                if event["event"] == "error":
                    raise RuntimeError(f"Pool worker {worker} failed:\n{event['traceback']}")
                if event["event"] == "ready":
                    ready.add(worker)
                    stats[worker] = event["stats"]
                    stats[worker]["gpu_slot"] = devices[:worker].count(devices[worker])
                    print(f"[pool ready] worker={worker} {event['stats']}", flush=True)
                    if len(ready) == launched:
                        for worker_id in range(launched):
                            dispatch(worker_id)
                elif event["event"] == "done":
                    job = busy[worker]
                    if event["key"] != job["key"] or not valid_chunk(root, job):
                        raise RuntimeError("Worker completion does not match its assigned chunk")
                    completed.add(job["key"])
                    print(f"[pool done] worker={worker} eval={job['name']} chunks={len(completed)}/{len(jobs)}", flush=True)
                    finalize_ready()
                    dispatch(worker)
        finalize_ready()
        report = {"schema": SCHEMA, "status": "complete", "scheduler": "task_pool", "evaluations": len(evaluations),
                  "chunks": len(jobs), "retained_chunks": reused, "executed_chunks": len(jobs) - reused,
                  "requested_workers": workers, "launched_workers": launched, "worker_stats": stats,
                  "requested_gpus": gpus, "runtime_worker_limit": runtime_workers,
                  "configured_max_workers_per_gpu": (workers + gpus - 1) // gpus,
                  "max_workers_per_gpu": (runtime_workers + gpus - 1) // gpus,
                  "elapsed_seconds": time.monotonic() - started}
        atomic_json(root / "summary.json", report)
        return report
    except BaseException as error:
        atomic_json(root / "summary.json", {"schema": SCHEMA, "status": "failed", "error": str(error),
                                            "completed_chunks": len(completed), "chunks": len(jobs),
                                            "requested_workers": workers, "runtime_worker_limit": runtime_workers,
                                            "launched_workers": launched})
        raise
    finally:
        for process in processes:
            process.join(timeout=1)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
        for q in [*inboxes, *([events] if events is not None else [])]:
            q.cancel_join_thread()
            q.close()


def main():
    from .config import load_config
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=8, help="Total independent model processes, not GPU count")
    parser.add_argument("--gpus", type=int, help="Number of visible GPUs to use (default: one per worker)")
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_pool(load_config(args.config), json.loads(Path(args.jobs).read_text()), args.output,
                              args.workers, args.chunk_size, args.resume, gpus=args.gpus), indent=2))


if __name__ == "__main__":
    main()
