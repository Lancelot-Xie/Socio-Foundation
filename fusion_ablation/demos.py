"""Fixed teacher trajectories sampled from the original teacher routing weights."""

import json
import random
from pathlib import Path


def build(cfg, output, candidates=1):
    from accelerate.utils import set_seed
    from opd.data import load_data, tokenizer_signature, write_rows
    from opd.models import TeacherBank, load_tokenizer
    from opd.rollout import collect_episode
    from opd.routing import Router
    from opd.workflows import inference_device, new_output

    if candidates < 1:
        raise ValueError("candidates must be positive")
    new_output(output)
    set_seed(cfg["seed"])
    rng = random.Random(cfg["seed"])
    tokenizer, device = load_tokenizer(cfg), inference_device()
    signature = tokenizer_signature(tokenizer, cfg)
    bank, router = TeacherBank(cfg, tokenizer, device), Router(cfg)
    rows = load_data(cfg["data"]["train_file"], cfg["data"]["dimension"])
    generated = []
    counts = {}
    for row in rows:
        weights, strength = router.resolve(row)
        if not weights or strength <= 0:
            raise ValueError(f"Inactive teacher route for {row['id']}; refusing silent prompt filtering")
        for trial in range(candidates):
            teacher = rng.choices(list(weights), weights=list(weights.values()), k=1)[0]
            transitions, _ = collect_episode(bank.activate(teacher), row, tokenizer, cfg, bank.device)
            for transition in transitions:
                response = tokenizer.decode(transition["response_ids"], skip_special_tokens=True)
                if not response:
                    # Preserve EOS-only trajectories without silently filtering them.
                    response = tokenizer.decode(transition["response_ids"], skip_special_tokens=False)
                # No demo_teacher: trainer.teacher_targets must keep the original
                # multi-teacher KL mixture instead of switching to the generator alone.
                clean = {k: v for k, v in row.items() if k not in (
                    "demo_teacher", "response", "response_token_ids", "prompt_token_ids", "tokenizer_signature")}
                generated.append({**clean, "id": f"{row['id']}:offline:{trial}:{transition['turn']}",
                    "source_id": row.get("source_id", row["id"]), "trajectory_teacher": teacher,
                    "messages": transition["messages"], "response": response,
                    "response_token_ids": transition["response_ids"], "prompt_token_ids": transition["prompt_ids"],
                    "tokenizer_signature": signature})
                counts[teacher] = counts.get(teacher, 0) + 1
    if not generated:
        raise ValueError("No teacher trajectories generated")
    write_rows(output, generated)
    report = {"prompts": len(rows), "trajectories": len(generated), "teacher_counts": counts,
              "filter_demos": False, "candidates_per_prompt": candidates,
              "response_tokens": sum(len(r["response_token_ids"]) for r in generated),
              "kl_target": "original routed teacher mixture", "trajectory_source": "teacher"}
    Path(str(output) + ".report.json").write_text(json.dumps(report, indent=2))
    return report
