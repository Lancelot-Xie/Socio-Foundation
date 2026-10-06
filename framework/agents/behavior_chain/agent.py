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
BehaviorChain multiple-choice agent for Harmony evaluation.

Each node asks the model to choose the next behavior (A/B/C/D) given a
persona profile, pre-chain history, prior context-behavior nodes, and the
current context.
"""

from __future__ import annotations

import json
import re

# reward-fix: import remove_think (like other agents) to strip reasoning before parsing.
from agents.utils import Agent, process_post_chat, remove_think

PROMPT_TEMPLATE = """You are simulating a persona's continuous behavior in a narrative benchmark.

# Persona Profile
{profile_text}

# Historical Narrative Before the Behavior Chain
{history_text}

# Prior Context-Behavior Chain
{prior_chain_text}

# Current Context
{current_context}

Which behavior would {persona_name} most likely take next?

{options_text}

Reason briefly from the persona, narrative history, and prior behavior chain. Then give the final choice on a new line as <answer>X</answer>, where X is exactly A, B, C, or D. Use the <answer> tag only once, for the final choice."""


def build_prompt(row: dict) -> str:
    profile_text = str(row.get("profile_text") or row.get("profile") or "(none)").strip() or "(none)"

    history = row.get("history_truncated")
    if isinstance(history, list) and history:
        parts = []
        for item in history:
            if isinstance(item, dict):
                chapter = str(item.get("chapter_num", "")).strip()
                content = str(item.get("chapter_content", "")).strip()
                parts.append(f"{chapter}\n{content}" if chapter and content else content)
            elif item:
                parts.append(str(item).strip())
        history_text = "\n\n".join(p for p in parts if p) or "(none)"
    else:
        history_text = str(row.get("history_text", "")).strip() or "(none)"

    prior_chain = row.get("prior_chain")
    if isinstance(prior_chain, list) and prior_chain:
        lines = []
        for i, step in enumerate(prior_chain):
            if not isinstance(step, dict):
                continue
            lines.append(f"Step {i}:")
            context = str(step.get("context", "")).strip()
            behavior = str(step.get("behavior", "")).strip()
            if context:
                lines.append(f"Context: {context}")
            if behavior:
                lines.append(f"Behavior: {behavior}")
        prior_chain_text = "\n".join(lines) or "(none)"
    else:
        prior_chain_text = str(row.get("prior_chain_text", "")).strip() or "(none)"

    options = row.get("options_dict")
    if isinstance(options, dict):
        options_text = "\n".join(f"{l}. {options.get(l, '')}" for l in "ABCD")  # noqa: E741
    else:
        options_text = "\n".join(f"{l}. {c}" for l, c in zip("ABCD", row.get("options") or [], strict=False))  # noqa: E741

    return PROMPT_TEMPLATE.format(
        profile_text=profile_text,
        history_text=history_text,
        prior_chain_text=prior_chain_text,
        current_context=row.get("current_context", ""),
        persona_name=row.get("persona_name", "the persona"),
        options_text=options_text,
    )


async def agent_loop(data: dict, context):
    row = json.loads(data["extra_info"]["raw"])
    chat = [
        {"role": "system", "content": ""},
        {"role": "user", "content": build_prompt(row)},
    ]

    agent = Agent(context.llm_client, chat, context.tokenizer, context.config, prompt_turn=2)
    response = await agent.step()

    # reward-fix: strip think/reasoning before extracting so an <answer> tag inside a
    # reasoning/draft block cannot be matched; take the LAST tag to honor self-corrections.
    content = remove_think(response or "")
    matches = re.findall(r"<answer>\s*([ABCD])\s*</answer>", content, flags=re.I)
    predicted_letter = matches[-1].upper() if matches else None
    if predicted_letter is None:
        # reward-fix: bare-letter fallback when no <answer> tag is present.
        bare = re.findall(r"\b([ABCD])\b", content, flags=re.I)
        predicted_letter = bare[-1].upper() if bare else None
    correct_letter = str(row.get("right_option_letter", "")).strip().upper()
    if not correct_letter:
        raise ValueError("BehaviorChain row missing right_option_letter")
    if correct_letter not in "ABCD":
        raise ValueError(f"BehaviorChain row has invalid right_option_letter={correct_letter!r}")
    reward = 1.0 if predicted_letter == correct_letter else 0.0

    output = await agent.get_agent_output(
        reward,
        extra_info={"behavior_chain/reward": reward, "all/score": reward},
    )
    await process_post_chat(data, context, agent.chat, output)
    return output
