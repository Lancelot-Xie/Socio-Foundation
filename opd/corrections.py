"""One-step teacher relabeling of actual student-visited visible prefixes.

The teacher generates only the next action at each captured prefix. We never replay
another trajectory as environment feedback or ask the teacher to advance a stale
simulator. Original demonstrations are replayed by the experiment orchestrator.
"""

from collections import Counter
import json

from accelerate.utils import set_seed

from .data import load_data, tokenizer_signature, write_rows
from .evaluation import evaluate_response, scalar_score
from .models import TeacherBank, load_student, load_tokenizer
from .rollout import collect_episode, generate_response
from .routing import Router
from .workflows import inference_device, new_output


def build_corrections(cfg, output, candidates=1):
    if candidates < 1:
        raise ValueError("candidates must be positive")
    path = new_output(output)
    set_seed(cfg["seed"])
    tokenizer, device = load_tokenizer(cfg), inference_device()
    student = load_student(cfg).requires_grad_(False).eval().to(device)
    bank = TeacherBank(cfg, tokenizer, device)
    router = Router(cfg)
    signature = tokenizer_signature(tokenizer, cfg)
    rows = load_data(cfg["data"]["train_file"], cfg["data"]["dimension"])
    results, counts = [], Counter()
    for row in rows:
        transitions, _ = collect_episode(student, row, tokenizer, cfg, device)
        teachers, strength = router.resolve(row)
        if not strength:
            continue
        for turn in transitions:
            current = {**row, "messages": turn["messages"]}
            counts["student_prefixes"] += 1
            options = []
            for teacher in teachers:
                model = bank.activate(teacher)
                for trial in range(candidates):
                    ids = generate_response(model, turn["prompt_ids"], tokenizer, cfg, bank.device)
                    text = tokenizer.decode(ids, skip_special_tokens=True)
                    evaluation = (evaluate_response(cfg, current, text) if cfg["quality"]["filter_demos"] else
                                  {"valid": False, "reason": "Unfiltered teacher correction"})
                    counts["teacher_candidates"] += 1
                    score = scalar_score(cfg, evaluation) if evaluation["valid"] else 0.0
                    if not text.strip() or (cfg["quality"]["filter_demos"] and (
                        not evaluation["valid"] or not evaluation["constraint_pass"] or score < cfg["quality"]["min_score"])):
                        counts["rejected"] += 1
                        continue
                    options.append((score, teacher, trial, ids, text, evaluation))
            seen = set()
            for _, teacher, trial, ids, text, evaluation in sorted(options, key=lambda x: x[0], reverse=True):
                if tuple(ids) in seen:
                    continue
                seen.add(tuple(ids))
                results.append({**current, "id": f"{row['id']}:correction:{turn['turn']}:{teacher}:{trial}",
                                "source_id": row.get("source_id", row["id"]), "response": text,
                                "response_token_ids": ids, "prompt_token_ids": turn["prompt_ids"],
                                "tokenizer_signature": signature, "demo_teacher": teacher,
                                "demo_quality": evaluation, "student_response_token_ids": turn["response_ids"],
                                "prefix_origin": "student", "student_checkpoint": cfg["model"].get("student_init")})
                if len(seen) >= cfg["quality"]["keep_per_prompt"]:
                    break
    if not results:
        raise ValueError(f"No teacher corrections accepted: {dict(counts)}")
    write_rows(path, results)
    report = {**dict(counts), "accepted_turns": len(results),
              "scope": "next-action relabeling; single-turn tasks revisit the original input"}
    path.with_name(path.name + ".report.json").write_text(json.dumps(report, indent=2))
    return report
