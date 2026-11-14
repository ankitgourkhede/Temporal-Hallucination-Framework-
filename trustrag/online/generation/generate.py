"""Trust-ranked context construction and evidence-bounded generation."""

from __future__ import annotations

import json
from typing import Any

from trustrag.online.generation.llm import LLM, LLMResult


SYSTEM_PROMPT = (
    "Answer only from the provided evidence. Every factual sentence must end with citations "
    "such as [E1]. Never present a conflicting view as fact. If evidence does not answer the "
    "question, say so plainly. Use plain language and at most 120 words."
)
BASELINE_PROMPT = (
    "Answer the user's travel advisory question using only the supplied raw retrieved passages. "
    "Cite factual sentences with their evidence IDs and use at most 120 words."
)


def _clip(text: str, limit: int = 280) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " ..."


def build_generation_payload(
    query: str,
    policy: dict[str, Any],
    trust_result: dict[str, Any],
    *,
    baseline_evidence: list[dict[str, Any]] | None = None,
    max_evidence: int = 6,
    min_trust: float = 0.20,
    required_slots: set[str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if trust_result.get("evidence"):
        ranked = sorted(trust_result["evidence"], key=lambda item: item.get("Trust", 0.0), reverse=True)
        chosen = [item for item in ranked if item.get("Trust", 0.0) >= min_trust][:max_evidence]
        # The slot the user asked about must reach the answer even when a conflict penalty has
        # pushed its winning evidence below min_trust; otherwise the reply silently omits it.
        for slot in trust_result.get("fusion", []):
            if slot["slot"] not in (required_slots or set()):
                continue
            if any(item.get("slot") == slot["slot"] and item.get("value_norm") == slot["value"] for item in chosen):
                continue
            best = next(
                (item for item in ranked if item.get("slot") == slot["slot"] and item.get("value_norm") == slot["value"]),
                None,
            )
            if best is not None:
                chosen = [best, *chosen][:max_evidence]
        trust_key = "Trust"
    else:
        chosen = sorted(baseline_evidence or [], key=lambda item: item.get("relevance", 0.0), reverse=True)[:max_evidence]
        trust_key = "relevance"

    evidence = []
    citation_by_evidence_id = {}
    for index, item in enumerate(chosen, start=1):
        citation_id = f"E{index}"
        enriched = {**item, "citation_id": citation_id}
        evidence.append(enriched)
        citation_by_evidence_id[item.get("evidence_id", item.get("chunk_id", citation_id))] = citation_id

    facts = []
    if trust_result.get("fusion"):
        for slot in trust_result["fusion"]:
            value = slot["value"]
            matching = next(
                (item for item in evidence if item.get("slot") == slot["slot"] and item.get("value_norm") == value),
                None,
            )
            if matching:
                facts.append({"slot": slot["slot"], "value": value, "evidence_id": matching["citation_id"]})
    else:
        seen = set()
        for item in evidence:
            key = (item.get("slot"), item.get("value_norm"))
            if item.get("slot") and key not in seen:
                seen.add(key)
                facts.append({"slot": item["slot"], "value": item["value_norm"], "evidence_id": item["citation_id"]})

    # Keep the prompt small: on CPU every prompt token costs time. The verifier works from the
    # full evidence list returned below, not from this payload, so trimming it is safe.
    payload = {
        "query": query,
        "nationality": policy.get("entities", {}).get("nationality"),
        "destination": policy.get("entities", {}).get("destination"),
        "facts": facts,
        "conflicts": [
            {"slot": slot["slot"], "chosen": slot["value"], "other": (slot.get("conflicting_view") or {}).get("value")}
            for slot in trust_result.get("fusion", [])
            if slot.get("conflicting_view")
        ],
        "evidence": [
            {
                "evidence_id": item["citation_id"],
                "source_type": item.get("source_type", "unknown"),
                "slot": item.get("slot"),
                "value": item.get("value_norm"),
                "text": _clip(str(item.get("chunk_text", item.get("text", "")))),
            }
            for item in evidence
        ],
    }
    return payload, evidence


def generate_answer(
    llm: LLM,
    payload: dict[str, Any],
    *,
    baseline: bool = False,
) -> LLMResult:
    return llm.generate(
        BASELINE_PROMPT if baseline else SYSTEM_PROMPT,
        json.dumps(payload, ensure_ascii=True),
        temperature=0.0,
        max_tokens=256,  # answers are limited to ~120 words
    )