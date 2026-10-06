"""Online quality-gated policy improvement on identical student-visited states."""

import copy
import json
from collections import defaultdict
from pathlib import Path

import torch

from .data import encode_prompt, make_batch
from .losses import forward_loss, sampled_reverse_loss, sequence_mean
from .models import response_logits
from .quality_basis import DIMENSIONS, evaluate_candidates, finite, select_winner
from .rollout import generate_response, generate_responses, load_callable


class Shortlist:
    """Explicit task routes define compatibility; calibration only changes rank."""
    def __init__(self, cfg, router):
        self.cfg, self.router = cfg, router
        path = cfg["qgpi"]["registry_file"]
        self.registry = json.loads(Path(path).read_text()) if path else None
        if self.registry is not None and (self.registry.get("schema") != "qgpi-quality-v1" or
                              self.registry.get("base_model") != cfg["model"]["base_model"] or
                              self.registry.get("teachers") != cfg["teachers"]):
            raise ValueError("QGPI registry must match the common base and task teachers")

    def resolve(self, row):
        weights, strength = self.router.resolve(row)
        gains = self.registry.get("tasks", {}).get(row["task_id"], {}) if self.registry else {}
        ranked = sorted(weights, key=lambda t: (-gains.get(t, {}).get("lower_gain", 0.0), -weights[t], t))
        return ranked[:self.cfg["qgpi"]["shortlist"]], strength


def branch_candidate(model, messages, row, tokenizer, cfg, device, env, snapshot, horizon):
    """Snapshots must include environment-owned RNG; never branch a live external service."""
    if env:
        env.restore(copy.deepcopy(snapshot))
    visible = copy.deepcopy(messages)
    trajectory, first, next_snapshot, next_state = [], None, None, None
    for offset in range(horizon):
        prompt = encode_prompt(tokenizer, visible, cfg)
        ids = generate_response(model, prompt, tokenizer, cfg, device)
        text = tokenizer.decode(ids, skip_special_tokens=True)
        state = env.step(text) if env else {"done": True, "messages": visible}
        record = {"messages": copy.deepcopy(visible), "response": text,
                  "next_messages": copy.deepcopy(state.get("messages", [])), "done": bool(state["done"])}
        if "reward" in state:
            if not finite(state["reward"]):
                raise ValueError("Environment reward must be a finite number")
            record["reward"] = state["reward"]
        trajectory.append(record)
        if offset == 0:
            first = {"prompt_ids": prompt, "response_ids": ids, "row": row, "messages": copy.deepcopy(visible)}
            next_snapshot = copy.deepcopy(env.snapshot()) if env else None
            next_state = copy.deepcopy(state)
        if state["done"]:
            break
        visible = copy.deepcopy(state["messages"])
    return {**first, "response": trajectory[0]["response"], "trajectory": trajectory,
            "_next_snapshot": next_snapshot, "_next_state": next_state}


def collect_decisions(student, bank, shortlist, row, tokenizer, cfg, device):
    factory = cfg["rollout"]["environment_factory"]
    env = load_callable(factory)(row) if factory else None
    messages = copy.deepcopy(env.reset() if env else row["messages"])
    if env and not all(callable(getattr(env, name, None)) for name in ("snapshot", "restore")):
        raise ValueError("QGPI candidate branching requires environment.snapshot() and restore(snapshot)")
    decisions = []
    for turn in range(cfg["rollout"]["max_turns"]):
        snapshot = copy.deepcopy(env.snapshot()) if env else None
        horizon = min(cfg["qgpi"]["branch_horizon"], cfg["rollout"]["max_turns"] - turn) if env else 1
        candidates = [branch_candidate(student, messages, row, tokenizer, cfg, device, env, snapshot, horizon)]
        candidates[0]["teacher"] = None
        teachers, strength = shortlist.resolve(row)
        for teacher in teachers:
            for _ in range(cfg["qgpi"]["candidates_per_teacher"]):
                candidate = branch_candidate(bank.activate(teacher), messages, row, tokenizer, cfg, bank.device,
                                             env, snapshot, horizon)
                candidate["teacher"] = teacher
                candidates.append(candidate)
        spec, evaluations = evaluate_candidates(cfg, row, messages, candidates)
        selection = select_winner(cfg, spec, evaluations)
        selection["weight"] *= strength
        decisions.append({**selection, "row_id": row["id"], "task_id": row["task_id"], "turn": turn,
                          "spec": spec, "evaluations": evaluations, "candidates": candidates})
        # Persist the student's first action, NEVER the teacher winner's state or
        # a speculative future continuation. Restoring avoids a stochastic re-step.
        state = candidates[0]["_next_state"]
        if env:
            env.restore(copy.deepcopy(candidates[0]["_next_snapshot"]))
        if state["done"]:
            break
        messages = copy.deepcopy(state["messages"])
    returns = 0.0
    for decision in reversed(decisions):
        state = decision["candidates"][0]["_next_state"]
        if cfg["qgpi"]["rl_coef"] and "reward" not in state:
            raise ValueError("Environment RL needs actual reward on every executed student transition")
        returns = state.get("reward", 0.0) + cfg["qgpi"]["discount"] * returns
        decision["student_return"] = returns
    return decisions


def collect_decision_batch(student, bank, shortlist, rows, tokenizer, cfg, device):
    """Batch identical-adapter work without changing routing, gates or states."""
    if cfg["rollout"]["environment_factory"] or cfg["rollout"]["generation_batch_size"] == 1:
        return [d for row in rows for d in collect_decisions(student, bank, shortlist, row, tokenizer, cfg, device)]
    prompts = [encode_prompt(tokenizer, row["messages"], cfg) for row in rows]
    candidates = [[] for _ in rows]

    def candidate(i, ids, teacher):
        row, messages = rows[i], copy.deepcopy(rows[i]["messages"])
        response = tokenizer.decode(ids, skip_special_tokens=True)
        return {"prompt_ids": prompts[i], "response_ids": ids, "row": row, "messages": messages,
                "response": response, "teacher": teacher,
                "trajectory": [{"messages": copy.deepcopy(messages), "response": response,
                                "next_messages": copy.deepcopy(messages), "done": True}],
                "_next_snapshot": None, "_next_state": {"done": True, "messages": messages}}

    for i, ids in enumerate(generate_responses(student, prompts, tokenizer, cfg, device)):
        candidates[i].append(candidate(i, ids, None))
    requests, strengths = defaultdict(list), []
    for i, row in enumerate(rows):
        teachers, strength = shortlist.resolve(row)
        strengths.append(strength)
        for teacher in teachers:
            for _ in range(cfg["qgpi"]["candidates_per_teacher"]):
                # Reserve slots to preserve per-row shortlist/candidate ordering.
                requests[teacher].append((i, len(candidates[i])))
                candidates[i].append(None)
    for teacher, locations in requests.items():
        responses = generate_responses(bank.activate(teacher), [prompts[i] for i, _ in locations],
                                       tokenizer, cfg, bank.device)
        for (i, slot), ids in zip(locations, responses, strict=True):
            candidates[i][slot] = candidate(i, ids, teacher)
    decisions = []
    for row, options, strength in zip(rows, candidates, strengths, strict=True):
        spec, evaluations = evaluate_candidates(cfg, row, row["messages"], options)
        selection = select_winner(cfg, spec, evaluations)
        selection["weight"] *= strength
        decisions.append({**selection, "row_id": row["id"], "task_id": row["task_id"], "turn": 0,
                          "spec": spec, "evaluations": evaluations, "candidates": options, "student_return": 0.0})
    return decisions


def take_batch(batch, indices):
    return {k: v[indices] if torch.is_tensor(v) else v for k, v in batch.items()}


def quality_loss(model, bank, decisions, tokenizer, cfg, accelerator):
    """One consistent distributed forward even if this rank accepts zero teachers.

    Imitation and winner KL live on teacher candidate tokens. Environment RL and
    the sampled base anchor live only on actual student-trajectory tokens.
    Rejected states contribute zero distillation, not synthetic negative labels.
    """
    q, device = cfg["qgpi"], accelerator.device
    n = len(decisions)
    accepted = [(i, d) for i, d in enumerate(decisions) if d["winner"] is not None and d["weight"] > 0]
    # Default QGPI has no student-trajectory RL/anchor loss. Those student
    # sequences have EXACTLY zero gradient, so omit their expensive forward.
    # Still call the distributed model once on a zero-weight dummy when this
    # rank rejects everything; no rank skips backward or gradient collectives.
    student_count = n if q["rl_coef"] or cfg["train"]["anchor_coef"] else 0
    sequences = ([d["candidates"][0] for d in decisions] if student_count else []) + [
        d["candidates"][d["winner"]] for _, d in accepted]
    strengths = [0.0] * student_count + [d["weight"] for _, d in accepted]
    if not sequences:
        sequences, strengths = [decisions[0]["candidates"][0]], [0.0]
    batch = make_batch([s["prompt_ids"] for s in sequences], [s["response_ids"] for s in sequences],
                       tokenizer.pad_token_id, device)
    student_indices = torch.arange(student_count, device=device)
    student_batch = take_batch(batch, student_indices)
    weights = torch.tensor(strengths, device=device)
    targets, groups = [], defaultdict(list)
    if q["kl_coef"]:
        for index, (_, decision) in enumerate(accepted, student_count):
            groups[decision["candidates"][decision["winner"]]["teacher"]].append(index)
        for teacher, indices in groups.items():
            indices = torch.tensor(indices, device=device)
            result = bank.score(teacher, take_batch(batch, indices), "forward_kl", cfg["train"]["teacher_top_k"])
            result = {k: v.to(device) if torch.is_tensor(v) else v for k, v in result.items()}
            targets.append({**result, "indices": indices, "alpha": torch.ones(len(indices), device=device)})
    old_logp, anchor = None, None
    if cfg["train"]["anchor_coef"]:
        with torch.no_grad(), accelerator.autocast():
            old_logits = response_logits(model, student_batch, cfg["rollout"]["temperature"])
            old_logp = old_logits.log_softmax(-1).gather(-1, student_batch["responses"].unsqueeze(-1)).squeeze(-1)
        anchor = bank.anchor_score(student_batch).to(device)
    model.train()
    with accelerator.autocast():
        logits = response_logits(model, batch, cfg["rollout"]["temperature"])
        distribution = logits.log_softmax(-1)
        logp = distribution.gather(-1, batch["responses"].unsqueeze(-1)).squeeze(-1)
        mask = batch["response_mask"]
        untempered = distribution if cfg["rollout"]["temperature"] == 1.0 else (
            logits * cfg["rollout"]["temperature"]).log_softmax(-1)
        supervised = untempered.gather(-1, batch["responses"].unsqueeze(-1)).squeeze(-1)
        # Divide by ALL visited decisions so small absolute advantages / rare
        # acceptance are not normalized away. Student rows have zero weights.
        imitation = -(sequence_mean(supervised, mask)*weights).sum()/n
        kl, _ = forward_loss(distribution, targets, mask, weights)
        kl = kl * len(sequences)/n
        # Token-length-normalized Monte Carlo policy-gradient surrogate, not an
        # exact sequence-summed REINFORCE objective. Only executed student tokens.
        rl = logp.sum() * 0.0
        if q["rl_coef"]:
            returns = torch.tensor([d["student_return"] - q["reward_baseline"] for d in decisions], device=device)
            rl = -(sequence_mean(logp[:n], mask[:n])*returns).mean()
        anchor_loss = logp.sum()*0.0
        if anchor is not None:
            anchor_loss, _ = sampled_reverse_loss(logp[:n], old_logp, anchor, mask[:n],
                                                   torch.ones(n, device=device), cfg["train"]["advantage_clip"])
        loss = q["imitation_coef"]*imitation + q["kl_coef"]*kl + q["rl_coef"]*rl + cfg["train"]["anchor_coef"]*anchor_loss
    metrics = {"imitation_loss": imitation.detach(), "winner_kl": kl.detach(),
               "environment_rl_loss": rl.detach(), "anchor_loss": anchor_loss.detach()}
    return loss, metrics, mask.sum(), n


METRIC_KEYS = ["visited", "accepted", "advantage_sum", "invalid_candidates", "candidate_count",
               "imitation_loss", "winner_kl", "environment_rl_loss", "anchor_loss"] + [
                   f"{d}_{suffix}" for d in DIMENSIONS for suffix in ("count", "student_sum", "winner_count", "winner_sum")]


def metric_values(decisions, components, device):
    result = dict.fromkeys(METRIC_KEYS, 0.0)
    result.update(components)
    result["visited"] = len(decisions)
    for decision in decisions:
        result["accepted"] += decision["winner"] is not None
        result["advantage_sum"] += decision["advantage"]
        result["candidate_count"] += len(decision["evaluations"])
        result["invalid_candidates"] += sum(not e["valid"] for e in decision["evaluations"])
        for d in DIMENSIONS:
            student = decision["evaluations"][0]
            if student["valid"] and d in student["scores"]:
                result[f"{d}_count"] += 1
                result[f"{d}_student_sum"] += student["scores"][d]
            if decision["winner"] is not None and d in decision["spec"]["dimensions"]:
                result[f"{d}_winner_count"] += 1
                result[f"{d}_winner_sum"] += decision["evaluations"][decision["winner"]]["scores"][d]
    return torch.stack([torch.as_tensor(result[k], device=device, dtype=torch.float32) for k in METRIC_KEYS])


def audit_record(decision):
    return {**{k: v for k, v in decision.items() if k != "candidates"},
            "candidates": [{k: c[k] for k in ("teacher", "response", "trajectory")} for c in decision["candidates"]]}
