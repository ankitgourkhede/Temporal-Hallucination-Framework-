"""Claim-level support checks with deterministic slot/value contradiction guards."""

from __future__ import annotations

import re
from typing import Any

from trustrag.online.generation.claims import Claim, extract_claims
from trustrag.online.retrieval.evidence import normalize_days, normalize_enum


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
            continue
        if not _slot_check(claim.text, fusion):
            label = "CONTRADICTED"
        elif _supported_by_text(claim.text, cited):
            label = "SUPPORTED"
        else:
            label = "UNSUPPORTED"
        results.append({"text": claim.text, "cited_ids": claim.cited_ids, "label": label})
    supported = sum(item["label"] == "SUPPORTED" for item in results)
    return {
        "faithfulness": supported / len(results) if results else 0.0,
        "claims": results,
        "flags": list(dict.fromkeys(flags)),
        "verifier": "rule_slot_fallback",
    }