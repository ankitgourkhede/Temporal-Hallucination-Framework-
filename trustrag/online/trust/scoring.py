"""Per-evidence trust feature calculation and weighted scoring."""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path
from typing import Any

import yaml


HEDGING_TERMS = ("may", "might", "usually", "subject to", "at the discretion", "depends", "typically", "generally")
SLOT_TERMS = {
    "visa_required": ("visa", "entry"),
    "visa_type": ("visa", "evisa", "e-visa", "visa-free"),
    "max_stay_days": ("stay", "days", "duration"),
    "passport_validity_months": ("passport", "months", "valid"),
    "vaccine_required": ("vaccine", "vaccination", "vaccination proof"),
    "vaccine_name": ("vaccine", "vaccination", "yellow fever"),
    "certificate_required": ("certificate", "proof", "vaccination"),
    "item_allowed": ("allowed", "prohibited", "restricted", "carry", "bring"),
    "duty_free_limit": ("duty-free", "limit", "allowance"),
    "declaration_required": ("declare", "declaration", "customs"),
}


def _as_date(value: Any) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _authority_levels(path: str | Path | None) -> dict[str, float]:
    if path is None:
        return {
            "government": 1.0, "embassy": 0.95, "intl_body": 0.9, "airline": 0.8,
            "news": 0.6, "travel_blog": 0.35, "forum_social": 0.2, "unknown": 0.3,
        }
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return {key: float(value) for key, value in config.get("tiers", config).items() if isinstance(value, (int, float))}


def _authority(evidence: dict[str, Any], levels: dict[str, float]) -> float:
    source_type = str(evidence.get("source_type", "unknown")).lower()
    aliases = {"blog": "travel_blog", "forum": "forum_social", "iata": "intl_body", "who": "intl_body"}
    return float(levels.get(aliases.get(source_type, source_type), levels.get("unknown", 0.3)))


def _freshness(evidence: dict[str, Any], as_of: date, half_life_days: int) -> tuple[float, str, str]:
    is_primary = bool(evidence.get("is_primary"))
    ref_field = "last_verified_ts" if is_primary else "content_date"
    ref_date = _as_date(evidence.get(ref_field))
    ref_rule = ref_field
    no_date = ref_date is None
    if no_date:
        ref_field = "crawl_timestamp"
        ref_rule = "crawl_timestamp_fallback_capped"
        ref_date = _as_date(evidence.get(ref_field))
    if ref_date is None:
        return 0.5, "unavailable", "no_date_capped"
    age_days = max(0, (as_of - ref_date).days)
    freshness = math.exp(-math.log(2) * age_days / half_life_days)
    if no_date:
        freshness = min(freshness, 0.5)
    return freshness, ref_date.isoformat(), ref_rule


def _content_relative_anchors(evidence: list[dict[str, Any]], as_of: date) -> dict[tuple[str, str], date | None]:
    """Newest content date per (slot, destination), taken from primary sources when any have one.

    Only primary sources set the anchor, so a recent forum post cannot make an official page
    look superseded; secondary sources still compete against the anchor like everyone else.
    """
    anchors: dict[tuple[str, str], date | None] = {}
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in evidence:
        groups.setdefault((str(item.get("slot", "")), str(item.get("destination", ""))), []).append(item)
    for key, items in groups.items():
        def dated(candidates: list[dict[str, Any]]) -> list[date]:
            values = [_as_date(item.get("content_date")) for item in candidates]
            return [value for value in values if value is not None and value <= as_of]

        primary_dates = dated([item for item in items if item.get("is_primary")])
        all_dates = primary_dates or dated(items)
        anchors[key] = max(all_dates) if all_dates else None
    return anchors


def _freshness_content_relative(
    evidence: dict[str, Any], anchor: date | None, as_of: date, half_life_days: int
) -> tuple[float, str, str]:
    """Freshness as supersession: how much older this content is than the newest official content.

    An unrevised page keeps a recent crawl date, so last_verified_ts cannot reveal that it is
    outdated; its content_date can. Sources without a content date fall back to the legacy
    rule, capped at 0.5 because their age is unknown.
    """
    ref_date = _as_date(evidence.get("content_date"))
    if ref_date is None or ref_date > as_of:
        legacy, legacy_ref, _ = _freshness(evidence, as_of, half_life_days)
        return min(legacy, 0.5), legacy_ref, "no_content_date_capped"
    reference = anchor or ref_date
    age_days = max(0, (reference - ref_date).days)
    return math.exp(-math.log(2) * age_days / half_life_days), ref_date.isoformat(), "content_date_vs_newest_primary"


def _evidence_quality(evidence: dict[str, Any], destination: str, slot: str) -> dict[str, float]:
    q_date = 1.0 if evidence.get("content_date") else 0.5 if evidence.get("crawl_timestamp") else 0.0
    text = str(evidence.get("chunk_text", ""))
    normalized = text.lower()
    has_value = evidence.get("value_norm") is not None
    hedged = bool(evidence.get("hedged")) or any(term in normalized for term in HEDGING_TERMS)
    q_spec = (0.5 if hedged else 1.0) if has_value else 0.0
    q_conf = min(1.0, max(0.0, float(evidence.get("extraction_confidence", 0.0))))
    has_destination = bool(destination and destination.lower() in normalized)
    terms = SLOT_TERMS.get(slot, (slot.replace("_", " "),))
    has_subject = any(term in normalized for term in terms)
    if len(text) >= 200 and has_destination and has_subject:
        q_complete = 1.0
    elif len(text) >= 200 and (has_destination or has_subject):
        q_complete = 0.5
    else:
        q_complete = 0.0
    if evidence.get("nationality_scope") == "UNKNOWN" and slot in {"visa_required", "visa_type", "max_stay_days"}:
        q_complete = min(q_complete, 0.5)
    return {"q_date": q_date, "q_spec": q_spec, "q_conf": q_conf, "q_complete": q_complete}


def _same_value(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return False
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if isinstance(left, int) and isinstance(right, int):
            return left == right
        return abs(float(left) - float(right)) <= max(0.01, abs(float(left)) * 0.01)
    if (
        isinstance(left, (tuple, list))
        and isinstance(right, (tuple, list))
        and len(left) == len(right) == 2
        and str(left[1]).upper() == str(right[1]).upper()
    ):
        left_amount, right_amount = float(left[0]), float(right[0])
        return abs(left_amount - right_amount) <= max(0.01, abs(left_amount) * 0.01)
    return left == right


def _value_family(value: Any) -> tuple[str, str | None] | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "bool", None
    if isinstance(value, (int, float)):
        return "number", None
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return "money", str(value[1]).upper()
    return "text", None


def _deduplicate_domains(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_domain: dict[str, dict[str, Any]] = {}
    for item in evidence:
        domain = str(item.get("source_domain", item.get("document_id", "unknown")))
        if domain not in by_domain or float(item.get("relevance", 0.0)) > float(by_domain[domain].get("relevance", 0.0)):
            by_domain[domain] = item
    return list(by_domain.values())


def _slot_severity(evidence: list[dict[str, Any]]) -> tuple[str, float]:
    evidence = _deduplicate_domains(evidence)
    comparable_groups: dict[tuple[str, str | None], list[dict[str, Any]]] = {}
    for item in evidence:
        family = _value_family(item.get("value_norm"))
        if family is not None:
            comparable_groups.setdefault(family, []).append(item)
    levels = {"None": 0.0, "Low": 0.2, "Medium": 0.5, "High": 0.8}
    best = ("None", 0.0)
    for group in comparable_groups.values():
        distinct = []
        for item in group:
            value = item.get("value_norm")
            if not any(_same_value(value, existing) for existing in distinct):
                distinct.append(value)
        if len(distinct) <= 1:
            continue
        primary = [item for item in group if item.get("is_primary")]
        primary_values = []
        for item in primary:
            value = item.get("value_norm")
            if not any(_same_value(value, existing) for existing in primary_values):
                primary_values.append(value)
        current = ("High", 0.8) if len(primary_values) > 1 else ("Medium", 0.5) if primary else ("Low", 0.2)
        if levels[current[0]] > levels[best[0]]:
            best = current
    return best


def score_evidence(
    evidence: list[dict[str, Any]],
    weights: dict[str, float],
    *,
    as_of: str | date,
    freshness_half_life_days: int,
    authority_path: str | Path | None = None,
    reputation: dict[str, tuple[int, int]] | None = None,
    alpha: float = 2.0,
    beta: float = 2.0,
    conflict_mode: str = "both",
    freshness_mode: str = "last_verified",
) -> list[dict[str, Any]]:
    as_of_date = date.fromisoformat(as_of) if isinstance(as_of, str) else as_of
    levels = _authority_levels(authority_path)
    reputation = reputation or {}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in evidence:
        grouped.setdefault((str(item.get("slot", "")), str(item.get("destination", ""))), []).append(item)

    if freshness_mode not in {"last_verified", "content_relative"}:
        raise ValueError(f"Unsupported freshness mode: {freshness_mode}")
    anchors = _content_relative_anchors(evidence, as_of_date) if freshness_mode == "content_relative" else {}

    def freshness_of(item: dict[str, Any]) -> tuple[float, str, str]:
        if freshness_mode == "content_relative":
            key = (str(item.get("slot", "")), str(item.get("destination", "")))
            return _freshness_content_relative(item, anchors.get(key), as_of_date, freshness_half_life_days)
        return _freshness(item, as_of_date, freshness_half_life_days)

    def vote_weight(peer: dict[str, Any]) -> float:
        # In content_relative mode a superseded source's agreement counts for less, so outdated
        # pages echoing each other cannot outvote a newer official statement.
        weight = _authority(peer, levels)
        if freshness_mode == "content_relative":
            weight *= freshness_of(peer)[0]
        return weight

    results = []
    for item in evidence:
        current = dict(item)
        peers = grouped[(str(item.get("slot", "")), str(item.get("destination", "")))]
        other_domains = [peer for peer in _deduplicate_domains(peers) if peer.get("source_domain") != item.get("source_domain")]
        comparable_peers = [
            peer for peer in other_domains
            if _value_family(peer.get("value_norm")) == _value_family(item.get("value_norm"))
        ]
        denominator = sum(vote_weight(peer) for peer in comparable_peers)
        numerator = sum(
            vote_weight(peer)
            for peer in comparable_peers
            if _same_value(peer.get("value_norm"), item.get("value_norm"))
        )
        agreement = numerator / denominator if denominator > 1e-9 else 0.5
        current["A"] = _authority(item, levels)
        current["F"], current["freshness_ref_date"], current["freshness_ref_rule"] = freshness_of(item)
        current["freshness_mode"] = freshness_mode
        current["G"] = agreement
        current["G_voters"] = [
            {
                "source_domain": peer.get("source_domain"),
                "authority": _authority(peer, levels),
                "value_norm": peer.get("value_norm"),
                "agrees": _same_value(peer.get("value_norm"), item.get("value_norm")),
            }
            for peer in other_domains
            if peer.get("value_norm") is not None
        ]
        current["qe_components"] = _evidence_quality(
            item, str(item.get("destination", "")), str(item.get("slot", ""))
        )
        current["QE"] = sum(current["qe_components"].values()) / 4.0
        n, correct = reputation.get(str(item.get("source_domain", "")), (0, 0))
        current["SR_n"] = n
        current["SR_correct"] = correct
        current["SR"] = (correct + alpha) / (n + alpha + beta)
        severity_name, severity = _slot_severity(peers)
        current["severity_level"] = severity_name
        current["severity"] = severity
        current["CP"] = severity * (1.0 - agreement)
        positive = (
            float(weights["w_A"]) * current["A"]
            + float(weights["w_F"]) * current["F"]
            + float(weights["w_G"]) * current["G"]
            + float(weights["w_E"]) * current["QE"]
            + float(weights["w_SR"]) * current["SR"]
        )
        current["T_raw"] = positive - float(weights["w_C"]) * current["CP"]
        # Source quality before the conflict multiplier; used by quality_x_share confidence,
        # where disagreement is already captured by the winning value's support share.
        current["T_quality"] = min(1.0, max(0.0, current["T_raw"]))
        clipped = min(1.0, max(0.0, current["T_raw"]))
        if conflict_mode == "subtract":
            trust = clipped
        elif conflict_mode == "multiply":
            trust = min(1.0, max(0.0, positive)) * (1.0 - current["CP"])
        elif conflict_mode == "both":
            trust = clipped * (1.0 - current["CP"])
        else:
            raise ValueError(f"Unsupported conflict mode: {conflict_mode}")
        current["Trust"] = min(1.0, max(0.0, trust))
        current["weights"] = dict(weights)
        current["conflict_mode"] = conflict_mode
        results.append(current)
    return results