"""Claim-level faithfulness verification (Phase 4 spec §4).

Why this module exists: RAGTruth-style claim-level checking tells us *which* sentence of an answer is
unsupported, which is what lets the UI highlight it and what RQ3's hallucination rate is measured on.

Verifiers, in order of preference:
  1. ``nli``  - cross-encoder NLI (``cross-encoder/nli-deberta-v3-small``). For each claim c and each premise
     sentence p from the evidence c cites, take max entailment and max contradiction over p:
        max_contradict >= tau_contradict -> CONTRADICTED
        max_entail     >= tau_entail     -> SUPPORTED
        otherwise                        -> UNSUPPORTED
  2. ``rule`` - deterministic token/value overlap. Used when the NLI model is not available locally (e.g. in
     offline unit tests). The verifier actually used is always reported in the result and the trace.

On top of either verifier, a deterministic slot check marks a claim CONTRADICTED when it states a slot value
(days, visa type) that differs from the fused value v*. NLI is weakest on exact numbers, so this guard catches
"45 days" vs "90 days" errors.

Faithfulness = #SUPPORTED / #claims. With 0 claims: 1.0 for an explicit "couldn't find" refusal (flagged
``no_answer_found``), else 0.0.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any

import numpy as np

from trustrag.online.generation.claims import extract_claims
from trustrag.online.retrieval.evidence import normalize_days, normalize_enum

LOGGER = logging.getLogger(__name__)
DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
# Label order documented on the model card for the cross-encoder/nli-deberta-v3-* family.
NLI_LABELS = ("contradiction", "entailment", "neutral")
REFUSAL_PATTERNS = ("couldn't find", "could not find", "couldn't determine", "could not determine")


def _slot_check(claim: str, fusion: list[dict[str, Any]]) -> bool:
    """Return False when the claim states a slot value that disagrees with the fused value v*."""
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
    """Rule fallback: value match or >= 60% overlap of meaningful claim tokens with the evidence text."""
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


def _premise_sentences(evidence: list[dict[str, Any]]) -> list[str]:
    sentences: list[str] = []
    for item in evidence:
        text = str(item.get("chunk_text") or item.get("text") or "")
        sentences.extend(part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if len(part.strip()) > 3)
    return list(dict.fromkeys(sentences))


@lru_cache(maxsize=2)
def load_nli_model(model_name: str = DEFAULT_NLI_MODEL, allow_download: bool = False) -> Any | None:
    """Load the NLI cross-encoder once per process; return None when it is not available locally."""
    try:
        from sentence_transformers import CrossEncoder
    except Exception:  # pragma: no cover - depends on the environment
        return None
    try:
        try:
            return CrossEncoder(model_name, device="cpu", local_files_only=not allow_download)
        except TypeError:  # older sentence-transformers without local_files_only
            if not allow_download:
                return None
            return CrossEncoder(model_name, device="cpu")
    except Exception as error:  # model not cached / no network
        LOGGER.info("NLI model unavailable (%s); using rule verifier", type(error).__name__)
        return None


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def nli_scores(model: Any, premises: list[str], claim: str) -> tuple[float, float]:
    """Return (max_entailment, max_contradiction) of the claim over all premise sentences."""
    if not premises:
        return 0.0, 0.0
    logits = np.asarray(model.predict([(premise, claim) for premise in premises]), dtype=float)
    if logits.ndim == 1:
        logits = logits.reshape(1, -1)
    probs = _softmax(logits)
    entail = float(probs[:, NLI_LABELS.index("entailment")].max())
    contradict = float(probs[:, NLI_LABELS.index("contradiction")].max())
    return entail, contradict


def verify_answer(
    answer: str,
    evidence: list[dict[str, Any]],
    fusion: list[dict[str, Any]],
    *,
    mode: str = "nli",
    entailment_threshold: float = 0.50,
    contradiction_threshold: float = 0.50,
    nli_model: Any | None = None,
    model_name: str = DEFAULT_NLI_MODEL,
    allow_download: bool = False,
) -> dict[str, Any]:
    claims = extract_claims(answer)
    model = None
    if mode == "nli":
        model = nli_model if nli_model is not None else load_nli_model(model_name, allow_download)
    verifier = "nli" if model is not None else "rule_slot_fallback"

    if not claims:
        refusal = any(pattern in answer.lower() for pattern in REFUSAL_PATTERNS)
        return {
            "faithfulness": 1.0 if refusal else 0.0,
            "claims": [],
            "flags": ["no_answer_found"] if refusal else [],
            "verifier": verifier,
        }

    by_citation = {item.get("citation_id"): item for item in evidence if item.get("citation_id")}
    results = []
    flags = []
    for claim in claims:
        cited = [by_citation[citation] for citation in claim.cited_ids if citation in by_citation]
        if not claim.cited_ids:
            cited = evidence
            flags.append("uncited_claim")
        elif not cited:
            results.append({"text": claim.text, "cited_ids": claim.cited_ids, "label": "UNSUPPORTED",
                            "reason": "cited evidence id not in context"})
            continue

        entail = contradict = None
        if not _slot_check(claim.text, fusion):
            label, reason = "CONTRADICTED", "slot value differs from fused value"
        elif model is not None:
            entail, contradict = nli_scores(model, _premise_sentences(cited), claim.text)
            if contradict >= contradiction_threshold:
                label, reason = "CONTRADICTED", "nli contradiction"
            elif entail >= entailment_threshold:
                label, reason = "SUPPORTED", "nli entailment"
            else:
                label, reason = "UNSUPPORTED", "nli neutral"
        elif _supported_by_text(claim.text, cited):
            label, reason = "SUPPORTED", "rule value/token match"
        else:
            label, reason = "UNSUPPORTED", "rule no match"
        results.append({
            "text": claim.text,
            "cited_ids": claim.cited_ids,
            "label": label,
            "reason": reason,
            "entailment": entail,
            "contradiction": contradict,
        })
    supported = sum(item["label"] == "SUPPORTED" for item in results)
    return {
        "faithfulness": supported / len(results) if results else 0.0,
        "claims": results,
        "flags": list(dict.fromkeys(flags)),
        "verifier": verifier,
    }
