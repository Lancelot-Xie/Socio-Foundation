"""Pure LifeChoices prompt construction and answer parsing utilities.

This module intentionally has no training-framework dependencies. Both the
training agent and standalone inference/evaluation programs import it so that
they always use the same prompt and parser.
"""

from __future__ import annotations

import re


PROMPT_TEMPLATE = """# Relevant Background / Memory
{input_text}

# Character
Name: {character_name}
Source work: {book}

# Current Scenario
{scenario}

# Question
{question}

# Candidate Continuations
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}

# How to Decide
Predict which option best matches what {character_name} actually says, does, or
decides in the source narrative. This is narrative prediction, not advice: do
not prefer an option merely because it is more moral, reasonable, cautious, or
strategically optimal.

Use evidence in this order:
1. The final turns and immediate situation in the current scene.
2. The character's explicit current goals, emotions, obligations, and constraints.
3. Established relationships, personality, and relevant past events.
4. Continuity of dialogue, tone, actions, and wording.

For a next-line question, prioritize the immediate transcript. For a high-level
life-decision question, use the background to identify the character's actual
outcome in the source story. If background and scene emphasis differ, answer
the specific question being asked.

Text in square brackets is a candidate internal motivation. Text in
parentheses is a candidate action or stage direction. Evaluate the complete
option, including its motivation, action, and dialogue. Do not infer the answer
from option position.

# Outputs:
Reason with one concise evidence sentence of at most 30 words. Do not restate
all options or mention an option letter in that sentence. Then put the final
choice on a new line using this format:

<answer>X</answer>

X must be exactly one letter: A, B, C, or D. Use the <answer> tag exactly once,
only for the final choice."""


def _clean_option(option: str) -> str:
    """Remove an option label already stored in the parquet row."""
    return re.sub(r"^\s*[A-D][.):]\s*", "", str(option)).strip()


def create_prompt(character_data: dict) -> str:
    """Create the LifeChoices canonical-narrative prediction prompt."""
    mcq = character_data["Multiple Choice Question"]
    options = mcq.get("Options", [])

    def get_option(index: int) -> str:
        if index >= len(options):
            return ""
        return _clean_option(options[index])

    return PROMPT_TEMPLATE.format(
        character_name=character_data.get("character_name", ""),
        book=character_data.get("book", ""),
        input_text=character_data.get("input_text", ""),
        scenario=mcq.get("Scenario", ""),
        question=mcq.get("Question", ""),
        option_a=get_option(0),
        option_b=get_option(1),
        option_c=get_option(2),
        option_d=get_option(3),
    )


def extract_choice(response: str) -> str | None:
    """Extract the choice letter (A, B, C, or D) from the LLM response."""
    if not response:
        return None

    response = response.strip().upper()
    if response in ["A", "B", "C", "D"]:
        return response

    patterns = [
        r"<answer>\s*([ABCD])\s*</answer>",
        r"^([ABCD])\s*[.):,]",
        r"choice[:\s]+([ABCD])",
        r"answer[:\s]+([ABCD])",
        r"\(([ABCD])\)",
        r"^([ABCD])$",
        r"([ABCD])\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # Deliberately do not accept the first A-D letter anywhere in the response;
    # that fallback creates false positives from analysis such as "Option A".
    return None
