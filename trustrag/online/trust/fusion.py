"""Authority-independent value fusion and answer confidence calculation."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from trustrag.online.trust.scoring import _same_value


def fuse_slots(
    evidence: list[dict[str, Any]],
    *,
    flags: list[str] | None = None,
    no_primary_evidence_slots: set[str] | None = None,
    primary_conflict_slots: set[str] | None = None,
    confidence_slope: float = 10.0,
) -> list[dict[str, Any]]:
    flags = set(flags or [])
    no_primary_evidence_slots = no_primary_evidence_slots or set()
    primary_conflict_slots = primary_conflict_slots or set()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in evidence:
        grouped[str(item["slot"])].append(item)

    output = []
    for slot, items in grouped.items():
        by_value: list[tuple[Any, list[dict[str, Any]]]] = []
        for item in items:
            value = item.get("value_norm")
            group = next((group for existing, group in by_value if _same_value(value, existing)), None)
            if group is None:
                by_value.append((value, [item]))
            else:
                group.append(item)

        support = {}
        representatives = {}
        for value, value_items in by_value:
            per_domain: dict[str, dict[str, Any]] = {}
            for item in value_items:
                domain = str(item.get("source_domain", item.get("document_id", "unknown")))
                if domain not in per_domain or float(item["Trust"]) > float(per_domain[domain]["Trust"]):
                    per_domain[domain] = item
            support[str(value)] = sum(float(item["Trust"]) for item in per_domain.values())
            representatives[str(value)] = (value, list(per_domain.values()))

        winner_key = max(support, key=support.get)
        winner_value, winner_items = representatives[winner_key]
        total_support = sum(support.values())
        share = support[winner_key] / total_support if total_support else 0.0
        best_trust = max((float(item["Trust"]) for item in winner_items), default=0.0)
        answer_trust = share * best_trust
        confidence_before_caps = 1.0 / (1.0 + math.exp(-confidence_slope * (answer_trust - 0.5)))
        confidence = confidence_before_caps
        applied_caps = []
        if slot in no_primary_evidence_slots or (
            "no_primary_evidence" in flags and not no_primary_evidence_slots
        ):
            confidence = min(confidence, 0.60)
            applied_caps.append("no_primary_evidence")
        if slot in primary_conflict_slots or ("primary_conflict" in flags and not primary_conflict_slots):
            confidence = min(confidence, 0.60)
            applied_caps.append("primary_conflict")

        runner_up = sorted(support.items(), key=lambda item: item[1], reverse=True)[1:2]
        conflicting_view = None
        if runner_up:
            value, amount = runner_up[0]
            conflicting_view = {"value": representatives[value][0], "support": amount}
        output.append(
            {
                "slot": slot,
                "value": winner_value,
                "support_by_value": support,
                "share": share,
                "max_trust": best_trust,
                "answer_trust": answer_trust,
                "confidence_before_caps": confidence_before_caps,
                "confidence": confidence,
                "conflicting_view": conflicting_view,
                "confidence_caps": applied_caps,
            }
        )
    return output


def summarize_sources(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sources: dict[str, dict[str, Any]] = {}
    for item in evidence:
        document_id = str(item["document_id"])
        current = sources.get(document_id)
        if current is None or float(item["Trust"]) > current["trust"]:
            sources[document_id] = {
                "document_id": document_id,
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "source_domain": item.get("source_domain"),
                "source_type": item.get("source_type"),
                "nationality_scope": item.get("nationality_scope", "UNKNOWN"),
                "trust": float(item["Trust"]),
            }
    return list(sources.values())