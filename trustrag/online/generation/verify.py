"""Claim-level support checks with deterministic slot/value contradiction guards."""

from __future__ import annotations

import re
from typing import Any

from trustrag.online.generation.claims import Claim, extract_claims
from trustrag.online.retrieval.evidence import extract_claims as extract_slot_values
from trustrag.online.retrieval.evidence import normalize_days, normalize_enum
from trustrag.online.trust.scoring import _same_value


def _claim_slot_values(claim: str, slots: set[str], item: str | None) -> dict[str, Any]:
    """Read slot values back out of an answer sentence with the evidence extractor."""
    if not slots:
        return {}
    extracted = extract_slot_values(
        chunk={"chunk_id": "answer", "text": claim},
        document={"document_id": "answer"},
        slots=sorted(slots),
        destination="",
        item=item,
    )
    return {entry["slot"]: entry["value_norm"] for entry in extracted}


def _slot_verdict(claim: str, cited: list[dict[str, Any]], fusion: list[dict[str, Any]]) -> str | None:
    """SUPPORTED / CONTRADICTED when the sentence states a slot value, else None.

    A sentence is supported when a value it states equals the value its cited evidence
    asserts for that slot, and contradicted when every stated value disagrees with both
    the cited evidence and the fused answer for that slot.
    """
    slots = {str(item.get("slot")) for item in cited if item.get("slot")} | {str(entry["slot"]) for entry in fusion}
    item_name = next((str(item.get("value_raw")) for item in cited if item.get("slot") == "item_allowed"), None)
    stated = _claim_slot_values(claim, slots, item_name)
    if not stated:
        return None
    for slot, value in stated.items():
        if any(item.get("slot") == slot and _same_value(item.get("value_norm"), value) for item in cited):
            return "SUPPORTED"
    fused = {str(entry["slot"]): entry.get("value") for entry in fusion}
    if any(slot in fused and not _same_value(fused[slot], value) for slot, value in stated.items()):
        return "CONTRADICTED"
    return None


def _slot_check(claim: str, fusion: list[dict[str, Any]]) -> bool:
    for slot in fusion:
        if slot["slot"] == "max_stay_days":
            days = normalize_days(claim)
            if days is not None and days != slot["value"]:
                return False
        if slot["slot"] == "visa_type":
            value = normalize_enum(claim)
            if value is not None and value != slot["value"]:
                return False
    return True


def _supported_by_text(claim: str, evidence: list[dict[str, Any]]) -> bool:
    claim_days = normalize_days(claim)
    claim_enum = normalize_enum(claim)
    claim_item_value = next(
        (value for value in ("prohibited", "restricted", "allowed") if re.search(rf"\b{value}\b", claim, re.I)),
        None,
    )
    if claim_item_value is None and re.search(r"\bapproval\b", claim, re.I):
        claim_item_value = "restricted"
    for item in evidence:
        if claim_days is not None and item.get("value_norm") == claim_days:
            return True
        if claim_enum is not None and item.get("value_norm") == claim_enum:
            return True
        if (
            claim_item_value is not None
            and item.get("slot") == "item_allowed"
            and item.get("value_norm") == claim_item_value
        ):
            return True
        claim_tokens = set(re.findall(r"[a-z0-9]+", claim.lower()))
        evidence_tokens = set(re.findall(r"[a-z0-9]+", str(item.get("chunk_text", "")).lower()))
        meaningful = {token for token in claim_tokens if len(token) > 3}
        if meaningful and len(meaningful & evidence_tokens) / len(meaningful) >= 0.6:
            return True
    return False


def verify_answer(
    answer: str,
    evidence: list[dict[str, Any]],
    fusion: list[dict[str, Any]],
) -> dict[str, Any]:
    claims = extract_claims(answer)
    if not claims:
        refusal = "couldn't find" in answer.lower() or "could not find" in answer.lower()
        return {"faithfulness": 1.0 if refusal else 0.0, "claims": [], "flags": [], "verifier": "rule_slot_fallback"}

    by_citation = {item.get("citation_id"): item for item in evidence if item.get("citation_id")}
    results = []
    flags = []
    for claim in claims:
        cited = [by_citation[citation] for citation in claim.cited_ids if citation in by_citation]
        if not claim.cited_ids:
            cited = evidence
            flags.append("uncited_claim")
        elif not cited:
            results.append({"text": claim.text, "cited_ids": claim.cited_ids, "label": "UNSUPPORTED"})
            flags.append("fabricated_citation")
            continue
        verdict = _slot_verdict(claim.text, cited, fusion)
        if verdict is not None:
            label = verdict
        elif not _slot_check(claim.text, fusion):
            label = "CONTRADICTED"
        elif _supported_by_text(claim.text, cited):
            label = "SUPPORTED"
        else:
            label = "UNSUPPORTED"
        results.append({"text": claim.text, "cited_ids": claim.cited_ids, "label": label})
    if any(item["label"] == "CONTRADICTED" for item in results):
        flags.append("contradicted_claim")
    supported = sum(item["label"] == "SUPPORTED" for item in results)
    return {
        "faithfulness": supported / len(results) if results else 0.0,
        # A fabricated citation or a contradiction blocks the answer regardless of the
        # supported-claim ratio, so true sentences cannot launder a hallucinated one.
        "blocking": any(flag in flags for flag in ("fabricated_citation", "contradicted_claim")),
        "claims": results,
        "flags": list(dict.fromkeys(flags)),
        "verifier": "rule_slot_fallback",
    }