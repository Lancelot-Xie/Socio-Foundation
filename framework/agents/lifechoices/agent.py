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
LifeChoices agent for Harmony evaluation.

Evaluates LLMs' role-playing capabilities in making persona-driven life choices
for literary characters. Single-phase: prompt LLM with character profile + MCQ,
extract choice, compare to ground truth.

Based on: "Character is Destiny: Can Large Language Models Simulate
Persona-Driven Decisions in Role-Playing?" (Xu et al., 2024)
"""

import copy
import uuid

from agents.lifechoices.prompt import PROMPT_TEMPLATE, _clean_option, create_prompt, extract_choice
from agents.utils import Agent, process_post_chat, remove_think


async def agent_loop(data: dict, context):
    """
    LifeChoices: Evaluate a single character's persona-driven decision.

    Args:
        data: {
            "character_data": dict with keys: character_name, input_text,
                              book, Multiple Choice Question (with Scenario,
                              Question, Options, Correct Answer)
        }
        context: {
            "client": AsyncOpenAI instance,
            "model": str model name
        }

    Returns:
        {"reward": float, "chat": list, "predicted": str, "correct": str}
    """
    character_data = data["extra_info"]

    prompt = create_prompt(character_data)
    chat = [{"role": "system", "content": ""}, {"role": "user", "content": prompt}]

    agent = Agent(context.llm_client, chat, context.tokenizer, context.config, prompt_turn=2, enable_think=False)
    response = await agent.step()

    content = remove_think(response)
    predicted = extract_choice(content)

    correct_answer = character_data.get("Multiple Choice Question", {}).get("Correct Answer", "")
    if correct_answer:
        correct_answer = correct_answer.strip().upper()
        if len(correct_answer) > 1:
            correct_answer = correct_answer[0]

    is_correct = predicted == correct_answer if predicted else False
    reward = 1.0 if is_correct else 0.0

    output = await agent.get_agent_output(
        reward,
        extra_info={
            "lifechoices/reward": reward,
            "lifechoices/response_length": len(response.split()) if response else 0,
            "all/score": reward,
            "all/score_v1": reward,
        },
    )

    # ===========================================================================
    # Hint + second attempt (mirrors sotopia copy-agent pattern)
    # Only generate a hint when the model answered incorrectly.
    # ===========================================================================
    extra = {}
    hint = None
    if getattr(context.config.algorithm, "agent_version", None) == "copy" and not is_correct:
        from agents.lifechoices.hint import generate_hint

        hint = await generate_hint(character_data, content)
        if hint:
            extra["hint"] = hint

    if getattr(context.config.algorithm, "agent_version", None) == "copy" and context.is_train and hint:
        from agents.lifechoices.hint_agent import agent_loop as hint_agent_loop

        data["extra_info"]["hint"] = hint
        data["extra_info"]["old_reward"] = reward

        hint_agent_output = await hint_agent_loop(data, context)
        copy_agent_output = copy.deepcopy(hint_agent_output)
        copy_agent_output.prompt_ids = copy.deepcopy(output.prompt_ids)
        copy_agent_output.extra_fields["gen_uid"] = str(uuid.uuid4())
        hint_agent_output.extra_fields["agent_role"] = "hint_agent"
        output = [output, copy_agent_output, hint_agent_output]
    # ===========================================================================

    await process_post_chat(data, context, agent.chat, output, extra=extra if extra else None)
    return output

    return {
        "reward": reward,
        "chat": chat,
        "predicted": predicted,
        "correct": correct_answer,
        "character_name": character_data.get("character_name"),
        "book": character_data.get("book"),
    }
