# Extracted from Supplemental agents/utils.py; Apache-2.0.
import re

def remove_think(text: str, remove_unclosed: bool = False) -> str:
    """Remove thinking blocks from model response. Supports <think>...</think> and <seed:think>...</seed:think>."""
    if not text:
        return text
    text = re.sub(r"<seed:think>.*?</seed:think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if remove_unclosed:
        text = re.sub(r"<seed:think>.*$", "", text, flags=re.DOTALL)
        text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    return text.strip()
