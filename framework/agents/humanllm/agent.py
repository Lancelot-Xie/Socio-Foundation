# Copyright 2025 Anonymous Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
HumanLLM agent — 20-way Item Selection MC eval.

Single-turn benchmark from the HumanLLM (Microsoft KDD '26) paper, Table 1
"Item Selection" task: given a user persona + purchase history + 20 candidate
items, predict which item this user would buy next.

Reward = reciprocal rank (1/rank) of the gold candidate within the model's
top-K ranked predictions; 0 if the gold is not in the top-K or no prediction.

Input row (set by prepare_dataset.process_humanllm_item_select_*):
  extra_info = {
    "prompt_text":   str,        # original full chat prompt (persona + history + 20 candidates)
    "answer_text":   str,        # gold candidate's full item name
    "answer_index":  int,        # 0-based index in [0, 19]
    "answer_letter": str,        # 'A'..'T'
    "candidates":    list[str],  # 20 candidate strings, ordered Candidate 1 .. 20
    "task":          "item_selection",
    "split":         str,
    "index":         int,
  }
"""

from __future__ import annotations

import re
from typing import Any, Optional

from agents.utils import Agent, process_post_chat, remove_think

TOP_K = 5

LETTERS = "ABCDEFGHIJKLMNOPQRST"  # 20-way

SYSTEM_PROMPT = (
    "You are simulating a user choosing items they would purchase next.\n"
    "Read the user description, purchase history, and candidate items carefully.\n"
    "Briefly reason from those signals before answering.\n\n"
    f"Output format (STRICT): your final line must be exactly\n"
    f"  <answer>X1,X2,X3,X4,X5</answer>\n"
    f"where X1..X{TOP_K} are {TOP_K} DISTINCT candidate letters from A-T, "
    f"comma-separated with no spaces, ordered from most to least likely.\n"
    f"Example: <answer>C,A,M,B,K</answer>\n"
)

# reward-fix: original pattern rejected lowercase (no re.I) and any spaces around commas,
# voiding otherwise-correct rankings on trivial format slips. Allow IGNORECASE + \s*,\s*.
ANSWER_PATTERN = re.compile(
    r"<answer>\s*([A-Ta-t](?:\s*,\s*[A-Ta-t]){" + str(TOP_K - 1) + r"})\s*</answer>",
    re.IGNORECASE,
)


def _raw_prompt_from_row(row: dict[str, Any]) -> str:
    prompt_text = str(row.get("prompt_text") or "").strip()
    if prompt_text:
        return prompt_text
    conversations = row.get("conversations") or []
    if conversations:
        msgs = conversations[0].get("messages") or []
        if msgs:
            return str(msgs[0].get("content") or "").strip()
    return ""


def _reformat_history(text: str) -> str:
    """Split inline 'History N: ...' entries onto their own lines."""
    matches = list(re.finditer(r"History\s*\d+\s*:", text))
    if not matches:
        return text
    head = text[: matches[0].start()].rstrip()
    items = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end() : end].strip().rstrip(",;")
        items.append(f"History {i + 1}: {body}")
    out = "\n".join(items)
    return (head + "\n" + out) if head else out


def _build_prompt_from_row(row: dict[str, Any]) -> str:
    """Return the user prompt with history one-per-line and candidates relabeled A-T."""
    raw = _raw_prompt_from_row(row)
    candidates: list[str] = list(row.get("candidates") or [])
    if not raw:
        return raw
    if not candidates:
        return _reformat_history(raw)

    head = raw
    m = re.search(r"Candidate\s*1\s*:", raw)
    if m:
        head = raw[: m.start()].rstrip()

    head = _reformat_history(head)

    lines = [f"{LETTERS[i]}: {str(c).strip()}" for i, c in enumerate(candidates) if i < len(LETTERS)]
    return f"{head}\n\n" + "\n".join(lines)


def _predict_ranking(model_output: str, candidates: list[str]) -> Optional[list[int]]:
    """Return the model's top-K ranked 0-based candidate indices, or None.

    Strict matching: requires exactly the pattern
        <answer>X1,X2,...,XK</answer>
    with K == TOP_K letters in A..T (no spaces, no brackets, no duplicates,
    every letter inside `candidates` range). Anything else returns None.
    """
    if not model_output or not candidates:
        return None

    m = ANSWER_PATTERN.search(model_output)
    if not m:
        return None
    # reward-fix: tolerate spaces/lowercase and accept partial lists — uppercase, strip, and
    # dedup preserving order so a correct ranking isn't voided by minor format slips (e.g. a
    # repeated letter). Out-of-range letters are skipped rather than voiding the whole ranking.
    raw_letters = [p.strip().upper() for p in m.group(1).split(",")]
    letters: list[str] = []
    for ch in raw_letters:
        if ch and ch in LETTERS and ch not in letters:
            letters.append(ch)
    if not letters:
        return None
    indices: list[int] = []
    for ch in letters:
        idx = LETTERS.index(ch)
        if idx >= len(candidates):
            continue
        indices.append(idx)
    return indices or None


async def agent_loop(data: dict, context):
    """HumanLLM Item Selection: 20-way MC."""
    row = data["extra_info"]
    prompt = _build_prompt_from_row(row)
    if not prompt:
        raise ValueError("HumanLLM item-selection row is missing prompt text.")

    candidates: list[str] = list(row.get("candidates") or [])
    if not candidates:
        raise ValueError("HumanLLM item-selection row is missing 'candidates'.")

    gold_index = int(row.get("answer_index", -1))
    if gold_index < 0 or gold_index >= len(candidates):
        raise ValueError(f"HumanLLM item-selection row has invalid answer_index={gold_index}.")

    user_content = (
        f"{prompt}\n\n"
        f"Task: rank the candidates this user is most likely to buy next.\n"
        f"Briefly reason from the user's profile, purchase history, and item similarities. "
        f"Then put the final ranking on a new line in this strict format:\n"
        f"  <answer>X1,X2,X3,X4,X5</answer>\n"
        f"Rules:\n"
        f"  - Exactly {TOP_K} letters, all distinct, from A-T.\n"
        f"  - Comma-separated, no spaces, no brackets, no extra characters.\n"
        f"  - Order: most likely first, least likely last.\n"
        f"  - Use the <answer> tag exactly once, only for the final ranking.\n"
        f"Example: <answer>C,A,M,B,K</answer>"
    )
    chat = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    agent = Agent(
        context.llm_client,
        chat,
        context.tokenizer,
        context.config,
        prompt_turn=2,
        enable_think=False,
    )
    response = await agent.step()
    content = remove_think(response or "")

    ranking = _predict_ranking(content, candidates)
    has_prediction = ranking is not None
    rank = -1
    reward = 0.0
    if has_prediction:
        topk = ranking[:TOP_K]
        if gold_index in topk:
            rank = topk.index(gold_index) + 1  # 1-based
            reward = 1.0 / rank
    top1_correct = 1.0 if (has_prediction and ranking[0] == gold_index) else 0.0
    hit_at_k = 1.0 if rank > 0 else 0.0

    output = await agent.get_agent_output(
        reward,
        extra_info={
            "humanllm/reward": reward,
            f"humanllm/hit@{TOP_K}": hit_at_k,
            "humanllm/top1_acc": top1_correct,
            # "humanllm/rank": rank,
            # "humanllm/has_prediction": 1.0 if has_prediction else 0.0,
            "humanllm/response_length": len(response.split()) if response else 0,
            "all/score": reward,
        },
    )
    await process_post_chat(data, context, agent.chat, output)
    return output
