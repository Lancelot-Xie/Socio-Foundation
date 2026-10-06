"""Pure transformations exercised with synthetic input only."""

from __future__ import annotations


import argparse


import hashlib


import json


from pathlib import Path


REVISION = "behaviorchain-author-persona100-mcq-v1"


EMPTY_CONTEXT = "[SOURCE_CONTEXT_EMPTY: initial chain behavior has no released predecessor context]"


def text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty text")
    return value.strip()


def valid_mcq(node: dict) -> bool:
    options = node.get("options_out_of_order")
    return (isinstance(options, dict) and set(options) == set("abcd")
            and node.get("right_option_index") in options
            and all(isinstance(v, str) and v.strip() for v in options.values()))


def normalize_persona(raw: dict, selected: dict) -> list[dict]:
    nodes = raw["examples"]
    if len(nodes) < 2 or valid_mcq(nodes[0]) or not all(valid_mcq(n) for n in nodes[1:]):
        raise ValueError("expected one context-only initial node followed by all valid MCQ targets")
    if len(nodes) != selected["raw_node_count"] or len(nodes) - 1 != selected["valid_mcq_node_count"]:
        raise ValueError("source node counts differ from split manifest")
    cutoff = raw["max_index"]
    summaries = raw["summary"]
    if isinstance(cutoff, bool) or not isinstance(cutoff, int) or cutoff < 0:
        raise ValueError("invalid pre-chain summary cutoff")
    if not raw.get("profile"):
        raise ValueError("persona profile is missing")
    history = [{"chapter_num": s.get("chapter_num"),
                "chapter_content": text(s["chapter_content"], "chapter_content")}
               for s in summaries[:cutoff]]
    # This node is known context, not a question: do not invent a score for it.
    initial_behavior = nodes[0].get("key_behavior", "").strip()
    if initial_behavior:
        history.append({"kind": "initial_chain_context",
                        "context": nodes[0].get("summary_refined", "").strip() or EMPTY_CONTEXT,
                        "behavior": initial_behavior})
    chain_length = len(nodes) - 1
    bucket = "short_11_to_12" if chain_length <= 12 else "medium_13_to_16" if chain_length <= 16 else "long_17_plus"
    prior = []
    records = []
    for index, node in enumerate(nodes[1:]):
        context = text(node["summary_refined"], "current context")
        behavior = text(node["key_behavior"], "reference behavior")
        right = node["right_option_index"]
        if node.get("meaningful") != 1 or node["options_out_of_order"][right].strip() != behavior:
            raise ValueError("expected key-behavior question with matching correct option")
        records.append({
            "source_id": f"{selected['persona_id']}::node_{index + 1}",
            "group_source_id": selected["persona_id"], "split": "test",
            "input": {"persona": raw["profile"], "persona_type": "author_profile_object",
                      "history": history, "history_mode": "gold_behavior_history",
                      "task_mode": "multiple_choice", "chain_index": index,
                      "chain_length": chain_length, "prior_nodes": list(prior),
                      "current_context": context,
                      "candidates": [node["options_out_of_order"][k] for k in "abcd"]},
            "gold": "abcd".index(right),
            "strata": {"chain_length_bucket": bucket, "key_behavior_status": "key",
                       "task_mode": "multiple_choice"},
            "metadata": {"source_folder": selected["source_folder"],
                         "source_relpath": selected["source_relpath"],
                         "raw_file_sha256": selected["raw_file_sha256"],
                         "persona_id": selected["persona_id"], "title": raw["title"],
                         "raw_node_index": index + 1, "raw_chain_length": len(nodes),
                         "initial_context_only_nodes": 1,
                         "background_summary_count": min(cutoff, len(summaries)),
                         "declared_summary_cutoff": cutoff,
                         "summary_cutoff_exceeds_available": cutoff > len(summaries),
                         "initial_context_has_behavior": bool(initial_behavior),
                         "seen_in_legacy_odysim_post_train": selected["seen_in_odysim_post_train"],
                         "legacy_post_train_node_count_for_persona": selected["odysim_post_train_node_count"],
                         "normalization_revision": REVISION},
        })
        prior.append({"context": context, "behavior": behavior})
    return records

