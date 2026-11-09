"""Deterministic claim splitting and citation extraction for verification."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Claim:
    text: str
    cited_ids: list[str]


def extract_claims(answer: str) -> list[Claim]:
    claims = []
    nonfactual = re.compile(
        r"^(hello|hi\b|please check the official|check the official|for (?:indian|uk|us) passport holders,? i assume"
        r"|i (?:couldn't|could not) (?:find|determine))",  # refusals are not factual claims (Phase 4 §4)
        re.IGNORECASE,
    )
    for sentence in re.split(r"(?<=[.!?])\s+", answer.strip()):
        sentence = sentence.strip()
        if not sentence or nonfactual.search(sentence):
            continue
        citations = list(dict.fromkeys(re.findall(r"\[(E\d+)\]", sentence)))
        text = re.sub(r"\s*\[E\d+\]", "", sentence).strip()
        if text:
            claims.append(Claim(text, citations))
    return claims