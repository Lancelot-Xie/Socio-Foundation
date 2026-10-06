"""Audited partial metrics; never execute a mixed task's agent_loop or a judge.

UserLM: PRISM boundary accuracy (T); decomposition/diversity are diagnostics.
MirrorBench: first-user-turn lexical similarity (N proxy), NOT full GTEval.
"""

import argparse
import hashlib
import os
from functools import lru_cache
from pathlib import Path

from .upstream import MODULES, load_source, repository, source_hashes

BPE_URL = "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
BPE_SHA256 = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"


def tokenizer_cache():
    directory = os.environ.get("TIKTOKEN_CACHE_DIR")
    if directory == "":
        raise ValueError("TIKTOKEN_CACHE_DIR must not disable the local cache")
    directory = Path(directory or Path(__file__).resolve().parents[1] / ".cache/tiktoken").resolve()
    return directory / hashlib.sha1(BPE_URL.encode()).hexdigest()


@lru_cache(maxsize=4)
def _encoding(cache_path):
    import tiktoken
    path = Path(cache_path)
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != BPE_SHA256:
        raise FileNotFoundError("MirrorBench needs the verified local o200k_base tokenizer. "
                                "Run bash train_objective.sh metrics-setup first (asset download only).")
    os.environ["TIKTOKEN_CACHE_DIR"] = str(path.parent)
    # Cache is verified BEFORE invoking tiktoken. No download on this path.
    return tiktoken.get_encoding("o200k_base")


def lexical_encoding():
    return _encoding(str(tokenizer_cache()))


def prepare_tokenizer(url=BPE_URL):
    """Explicit setup operation, never invoked by a scorer, doctor or exporter."""
    from urllib.request import urlopen
    path = tokenizer_cache()
    if path.exists():
        lexical_encoding()  # reject corrupt assets, never silently replace them
        return str(path)
    with urlopen(url, timeout=30) as response:
        content = response.read()
    if hashlib.sha256(content).hexdigest() != BPE_SHA256:
        raise ValueError("Downloaded tokenizer checksum mismatch")
    path.parent.mkdir(parents=True, exist_ok=True)
    from tempfile import NamedTemporaryFile
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        stream.write(content)
        temporary = Path(stream.name)
    temporary.replace(path)
    lexical_encoding()
    return str(path)


def eligible(task, raw):
    info = raw.get("extra_info", {})
    if task == "userllm":
        if "prism" not in str(info.get("source", "")).lower():
            return False, "UserLM non-PRISM source: not the audited termination task"
        if type(info.get("is_last_turn")) is not bool:
            return False, "UserLM needs a boolean gold termination label"
    if task == "mirrorbench":
        turns = info.get("turns") or []
        if not turns or turns[0].get("role") != "user" or not str(turns[0].get("content", "")).strip():
            return False, "MirrorBench first-user-turn metric needs a matching human first turn"
    return True, None


def module_for(row):
    task = row["task_id"]
    root = repository()
    expected = row.get("evaluator_context", {}).get("source_hashes")
    if expected and expected != source_hashes(str(root), task):
        raise ValueError("Original metric sources changed; re-export to a new output directory")
    return load_source(str(root), MODULES[task])


def user_output(module, response):
    output = module.remove_think(response)
    if "<|endconversation|>" in output:
        output = output.split("<|endconversation|>")[0].strip() + "<|endconversation|>"
    return output


def partial_response(row, response):
    task = row["task_id"]
    raw = row.get("evaluator_context", {}).get("original_row", {})
    valid, reason = eligible(task, raw)
    if not valid:
        return {"valid": False, "reason": reason}
    module = module_for(row)
    info = raw["extra_info"]
    if task == "userllm":
        output = user_output(module, response)
        pred = "<|endconversation|>" in output
        true = info["is_last_turn"]
        # Empty responses are model failures, not successful nontermination.
        score = float(bool(output.strip()) and pred == true)
        metrics = {"termination_accuracy": score, "termination_tp": float(pred and true),
                   "termination_fp": float(pred and not true), "termination_fn": float(not pred and true),
                   "termination_gold_positive": float(true)}
        intent = module._extract_intent(module._as_test_case(info))
        if intent and output.strip():
            metrics["intent_decomposition_diagnostic"] = 1.0 - module._intent_1gram_overlap_compatible(intent, output)
        return {"valid": True, "constraint_pass": True, "scores": {"T": score}, "task_metrics": metrics}
    if task != "mirrorbench":
        raise ValueError(f"No audited partial metric for {task}")
    output = response.strip()
    if output.lower().startswith("user:"):
        output = output[5:].strip()
    output = module.split_think(output)[1].strip()
    encoding = lexical_encoding()
    # Same tokenization and lexical kernels as upstream, but a matched static
    # first turn on BOTH sides. Full-dialogue lexical metrics are not claimed.
    reference = encoding.encode(info["turns"][0]["content"], disallowed_special=())
    generated = encoding.encode(output, disallowed_special=())
    metrics = {}
    for name in ("mattr", "hdd", "yules_k"):
        compute = getattr(module, "compute_" + name)
        metrics["human_first_" + name] = compute(reference)
        metrics["generated_first_" + name] = compute(generated)
    gap = sum(abs(metrics["human_first_" + k] - metrics["generated_first_" + k]) for k in ("mattr", "hdd")) / 2
    score = max(0.0, 1.0 - gap) if generated else 0.0
    metrics["first_turn_lexical_similarity"] = score
    return {"valid": True, "constraint_pass": True, "scores": {"N": score}, "task_metrics": metrics}


def aggregate_partial_metrics(task, records):
    if task != "userllm":
        return {}
    valid = [r for r in records if r["evaluation"].get("valid")]
    sums = {k: sum(r["evaluation"].get("task_metrics", {}).get("termination_" + k, 0.0) for r in valid)
            for k in ("tp", "fp", "fn")}
    denominator = 2 * sums["tp"] + sums["fp"] + sums["fn"]
    module = load_source(str(repository()), MODULES["userllm"])
    first = [user_output(module, r["response"]) for r in valid if r.get("metric_context", {}).get("is_first_turn")]
    first = [text for text in first if text.strip()]
    return {"termination_tp": sums["tp"], "termination_fp": sums["fp"], "termination_fn": sums["fn"],
            "termination_f1": 2*sums["tp"]/denominator if denominator else 0.0,
            "termination_f1_denominator": denominator,
            "first_turn_samples": len(first),
            "first_turn_diversity_diagnostic": module._first_turn_diversity_compatible(first) if len(first) >= 2 else None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-tokenizer", action="store_true", required=True)
    parser.add_argument("--tokenizer-url", default=BPE_URL, help="Optional asset mirror; the same pinned SHA256 is mandatory")
    args = parser.parse_args()
    print(prepare_tokenizer(args.tokenizer_url))
