"""Pure transformations exercised with synthetic input only."""

from __future__ import annotations


import argparse


import copy


import hashlib


import json


from collections import Counter


from pathlib import Path


from typing import Any


REVISION = "alignx-arbitrary-frozen-mixed-signals-v2"


def project_signals(record: dict) -> dict:
    # These are the fields read by ICA/few-shot; not the PBA profile summary.
    demo = record["Demographic Information"]
    pairs = record["Pair-wise Comparative Feedback"]
    ugc = record["User-Generated Content"]
    if not isinstance(demo, str) or not isinstance(pairs, list) or not isinstance(ugc, list):
        raise ValueError("invalid official arbitrary material")
    if not (demo.strip() or pairs or ugc):
        raise ValueError("arbitrary case has no supplied signals")
    def fields(items, keys):
        result = []
        for item in items:
            if not all(isinstance(item.get(key), str) and item[key].strip() for key in keys):
                raise ValueError("missing official history text")
            result.append({key: item[key] for key in keys})
        return result
    return {"demographic_information": demo,
            "pairwise_feedback": fields(pairs, ("prompt", "chosen", "rejected")),
            "user_generated_content": fields(ugc, ("prompt", "comment"))}


def restore_row(row: dict, source: dict) -> dict:
    if row["input"]["variant"] != "Reddit_arbitrary":
        return copy.deepcopy(row)
    if (row["source_id"] != source["eval_id"] or row["group_source_id"] != source["eval_core_id"]
            or row["metadata"]["core_sha256"] != source["core_sha256"]
            or source["condition"] != "Reddit_arbitrary"):
        raise ValueError("frozen arbitrary identity mismatch")
    record = source["record"]
    if any(row["input"][key] != record[key] for key in ("prompt", "chosen", "rejected")):
        raise ValueError("target preference core changed")
    signals = project_signals(record)
    previous = row["input"]["persona_components"]
    if previous != record["profile"] and previous != signals:
        raise ValueError("unexpected existing arbitrary conditioning; refusing to overwrite")
    result = copy.deepcopy(row)
    result["input"]["persona_components"] = signals
    result["input"]["arbitrary_conditioning_mode"] = REVISION
    return result

