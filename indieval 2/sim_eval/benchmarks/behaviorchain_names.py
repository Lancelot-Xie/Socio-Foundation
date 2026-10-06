"""Deterministic, chain-stable pseudonyms for explicitly declared character names.

Upstream 08_evaluation.moved_prompt renames name components using sequential
str.replace calls. Here token boundaries and a single replacement pass avoid
substring/cascading corruption. Names come only from the fixed profile; history
is used solely to avoid choosing an alias that already occurs in the story.
No gold, current options, future nodes, model identity or per-case seed is used.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from typing import Any

from ..contracts import BenchmarkCase
from ..json_utils import canonical_json


NAME_POLICY_REVISION = "behaviorchain-chain-stable-boundary-pseudonyms-v1"
_WORDS = re.compile(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*", re.UNICODE)
# Relationship categories, honorifics, annotations and high-risk function words
# are not character identifiers. In particular do not rewrite 'Will you ...'.
_PROTECTED = frozenset("""
the a an and or of at in to is are i you he she it we they will may can must
mr mrs miss ms dr sir lady lord king queen professor captain father mother
brother sister uncle aunt son daughter mom dad mama nana pop matka
parents grandparents grandmother grandfather stepmother stepfather family
friend friends students teachers colleagues interrogators children roommates
first last name unknown psychic husband wife cousin abuela abuelita
""".split())
_ALIASES = tuple("""
Alex Sam Taylor Jordan Casey Morgan Riley Jamie Drew Jesse Skyler Quinn Blake
Kerry Adrian Cameron Devon Finley Harley Marley Peyton Reese Sidney Terry Tracy
Charlie Dallas Emery Hayden Jody Kendall Leslie Mackenzie Pat Regan Shannon
Stevie Toby Valerie Wesley Aven Belen Corin Daren Elian Faren Galen Halen Ilan
Jorin Kalen Loren Maren Nerin Oren Perrin Quinlan Riven Soren Taren Ulen Varen
Wren Xeran Yaren Zerin
""".split())


def name_aliases_for_case(case: BenchmarkCase) -> dict[str, str]:
    profile = case.input_data.get("persona")
    if not isinstance(profile, Mapping):
        # Legacy description-only cases do not declare an auditable name list.
        return {}
    relationships = profile.get("Relationships") or {}
    labels = [str(profile.get("Name") or "")]
    if isinstance(relationships, Mapping):
        labels.extend(str(name) for name in relationships)
    components: set[str] = set()
    for label in labels:
        if label.startswith("The "):
            # Descriptive epithets are retained; don't turn 'The'/'Man' into names.
            continue
        label = re.split(r"['’]s\s", label, maxsplit=1)[0]
        for token in _WORDS.findall(label):
            if len(token) > 1 and token[0].isupper() and token.casefold() not in _PROTECTED:
                components.add(token)
    fixed_material = canonical_json({"persona": profile, "history": case.input_data.get("history")})
    occupied = {word.casefold() for word in _WORDS.findall(fixed_material)}
    occupied.update(word.casefold() for word in components)
    key = hashlib.sha256(canonical_json({"policy": NAME_POLICY_REVISION, "group": case.group_id,
                                        "fixed_material": fixed_material}).encode()).hexdigest()
    aliases: dict[str, str] = {}
    for source in sorted(components):
        candidates = [name for name in _ALIASES if name.casefold() not in occupied]
        if candidates:
            alias = min(candidates, key=lambda name: hashlib.sha256(
                f"{key}\0{source}\0{name}".encode()).hexdigest())
        else:
            # No finite pool limit: retain unique, pronounceable synthetic names.
            index = 0
            while True:
                digest = hashlib.sha256(f"{key}\0{source}\0{index}".encode()).digest()
                syllables = ("ba", "de", "fi", "go", "ha", "ju", "ka", "le",
                             "mi", "no", "pa", "ri", "se", "ta", "ve", "zo")
                alias = "".join(syllables[b % 16] for b in digest[:4]).capitalize()
                if alias.casefold() not in occupied:
                    break
                index += 1
        aliases[source] = alias
        occupied.add(alias.casefold())
    return aliases


def pseudonymize_material(value: Any, aliases: Mapping[str, str]) -> Any:
    if not aliases:
        return value
    pattern = re.compile(r"(?<!\w)(?:" + "|".join(
        re.escape(name) for name in sorted(aliases, key=lambda name: (-len(name), name))
    ) + r")(?!\w)")

    def text(raw: str) -> str:
        return pattern.sub(lambda match: aliases[match.group()], raw)

    def rewrite(item: Any, *, name_keys: bool = False) -> Any:
        if isinstance(item, str):
            return text(item)
        if isinstance(item, Mapping):
            return {(text(key) if name_keys and isinstance(key, str) else key):
                    (child if key == "id" else rewrite(child, name_keys=key == "Relationships"))
                    for key, child in item.items()}
        if isinstance(item, Sequence) and not isinstance(item, (bytes, bytearray)):
            return [rewrite(child) for child in item]
        return item

    return rewrite(value)
