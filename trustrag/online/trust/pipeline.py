"""Assemble scored evidence, per-source trust, slot fusion, and trace output."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from trustrag.offline.knowledge_store import KnowledgeStore
from trustrag.online.trust.fusion import fuse_slots, summarize_sources
from trustrag.online.trust.scoring import score_evidence


class TrustPipeline:
    def __init__(self, config_path: str | Path, store: KnowledgeStore | None = None):
        self.config_path = Path(config_path)
        with self.config_path.open("r", encoding="utf-8") as handle:
            self.config = yaml.safe_load(handle) or {}
        self.store = store

    def score(
        self,
        *,
        retrieval_result: dict[str, Any],
        policy: dict[str, Any],
        as_of: str | None = None,
    ) -> dict[str, Any]:
        raw_evidence = retrieval_result.get("evidence", [])
        domains = {str(item.get("source_domain", "")) for item in raw_evidence}
        reputation: dict[str, tuple[int, int]] = {}
        if self.store is not None:
            reputation = {domain: self.store.reputation(domain) for domain in domains}
        else:
            try:
                with KnowledgeStore.from_config(self.config_path) as store:
                    reputation = {domain: store.reputation(domain) for domain in domains}
            except FileNotFoundError:
                reputation = {}

        trust_config = self.config.get("trust", {})
        reputation_config = self.config.get("reputation", {})
        half_life = int(policy.get("freshness_half_life_days", 60))
        scored = score_evidence(
            raw_evidence,
            policy["weights"],
            as_of=as_of or str(self.config.get("project", {}).get("as_of_date", "2026-01-01")),
            freshness_half_life_days=half_life,
            authority_path=self.config_path.parent / "policies" / "authority.yaml",
            reputation=reputation,
            alpha=float(reputation_config.get("alpha", 2.0)),
            beta=float(reputation_config.get("beta", 2.0)),
            conflict_mode=str(trust_config.get("conflict_mode", "both")),
            freshness_mode=str(trust_config.get("freshness_mode", "last_verified")),
        )
        flags = list(retrieval_result.get("flags", []))
        primary_conflict_slots = {
            str(item["slot"]) for item in scored if item.get("severity_level") == "High"
        }
        if primary_conflict_slots:
            flags.append("primary_conflict")
        flags = list(dict.fromkeys(flags))
        fusion = fuse_slots(
            scored,
            flags=flags,
            no_primary_evidence_slots=set(
                retrieval_result.get("trace", {}).get("no_primary_evidence_slots", [])
            ),
            primary_conflict_slots=primary_conflict_slots,
        )
        trace = {
            "evidence": [
                {
                    "evidence_id": item.get("evidence_id"),
                    "slot": item.get("slot"),
                    "value_norm": item.get("value_norm"),
                    "source_domain": item.get("source_domain"),
                    "relevance": item.get("relevance"),
                    "A": item.get("A"),
                    "F": item.get("F"),
                    "freshness_ref_date": item.get("freshness_ref_date"),
                    "freshness_ref_rule": item.get("freshness_ref_rule"),
                    "G": item.get("G"),
                    "G_voters": item.get("G_voters"),
                    "qe_components": item.get("qe_components"),
                    "QE": item.get("QE"),
                    "SR": item.get("SR"),
                    "SR_n": item.get("SR_n"),
                    "SR_correct": item.get("SR_correct"),
                    "severity": item.get("severity"),
                    "CP": item.get("CP"),
                    "T_raw": item.get("T_raw"),
                    "Trust": item.get("Trust"),
                    "weights": item.get("weights"),
                    "weight_mode": policy.get("weight_mode"),
                    "conflict_mode": item.get("conflict_mode"),
                    "freshness_mode": item.get("freshness_mode"),
                }
                for item in scored
            ],
            "slots": fusion,
            "weights_source": policy.get("weights_source"),
            "flags": flags,
            **retrieval_result.get("trace", {}),
        }
        return {
            "evidence": scored,
            "sources": summarize_sources(scored),
            "fusion": fusion,
            "flags": flags,
            "trace": trace,
        }