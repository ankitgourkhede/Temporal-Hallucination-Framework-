"""Category-based volatility and evidence policy decisions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from trustrag.online.policy.volatility import compute_tvs
from trustrag.online.policy.weights import load_weights


class PolicyEngine:
    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        with self.config_path.open("r", encoding="utf-8") as handle:
            self.config = yaml.safe_load(handle) or {}

        policy_config = self.config.get("policy", {})
        categories_path = self.config_path.parent / policy_config.get(
            "categories_file", "policies/categories.yaml"
        )
        with categories_path.open("r", encoding="utf-8") as handle:
            self.categories = yaml.safe_load(handle) or {}
        self.thresholds = policy_config.get("volatility_thresholds", {})
        self.policy_config = policy_config
        self.retrieval_config = self.config.get("retrieval", {})
        self.project_config = self.config.get("project", {})
        strategy_path = self.config_path.parent / policy_config.get(
            "strategy_rules_file", "policies/strategy_rules.yaml"
        )
        self.strategy_rules = yaml.safe_load(strategy_path.read_text(encoding="utf-8")) or []

    @classmethod
    def from_config(cls, config_path: str | Path) -> "PolicyEngine":
        return cls(config_path)

    def decide(
        self,
        category: str,
        volatility_score: float | None = None,
        *,
        intent_confidence: float = 1.0,
        entities: dict[str, Any] | None = None,
        flags: list[str] | None = None,
        category_alternates: list[str] | None = None,
        as_of: str | None = None,
        live_requested: bool = False,
        mode: str = "demo",
    ) -> dict[str, Any]:
        if category not in self.categories:
            raise ValueError(f"Unsupported policy category: {category}")

        flags = list(flags or [])
        entities = entities or {}
        selected_category = category
        if category_alternates:
            candidates = [category, *category_alternates]
            selected_category = max(
                candidates,
                key=lambda value: float(self.categories[value].get("expert_prior", 0.5)),
            )
        category_policy = self.categories[selected_category]
        if volatility_score is None:
            events = []
            try:
                from trustrag.offline.knowledge_store import KnowledgeStore

                with KnowledgeStore.from_config(self.config_path) as store:
                    events = store.change_history(selected_category)
            except FileNotFoundError:
                pass
            tvs_result = compute_tvs(
                events,
                float(category_policy["expert_prior"]),
                as_of or str(self.project_config.get("as_of_date", "2026-01-01")),
            )
            score = float(tvs_result["tvs"])
        else:
            score = float(volatility_score)
            tvs_result = {"tvs": score, "tvs_source": "provided"}
        if not 0.0 <= score <= 1.0:
            raise ValueError("TVS must be between 0.0 and 1.0")

        low_below = float(self.thresholds.get("low_below", 0.35))
        high_from = float(self.thresholds.get("high_from", 0.65))
        if low_below >= high_from:
            raise ValueError("low_below must be less than high_from")
        volatility = "Low" if score < low_below else "Medium" if score < high_from else "High"

        live_config = self.config.get("live", {})
        live_enabled = bool(live_config.get("enabled", False)) and mode != "benchmark"
        live_active = live_enabled and (
            live_requested
            or (live_config.get("auto_on_high_volatility", False) and volatility == "High")
        )

        if intent_confidence < 0.60:
            flags.append("low_intent_confidence")
        strategy = "PRIMARY_ONLY"
        top_k = {"primary": 8, "secondary": 0}
        for rule in self.strategy_rules:
            condition = rule.get("when") or {}
            matched = (
                ("flag" in condition and condition["flag"] in flags)
                or (condition.get("live_requested") is True and live_active)
                or (condition.get("volatility") == volatility)
            )
            if matched:
                strategy = str(rule["strategy"])
                top_k = {key: int(value) for key, value in rule["top_k"].items()}
                break
        if strategy == "LIVE_API" and not live_active:
            strategy = "PRIMARY_AND_VALIDATION"
            top_k = {"primary": 8, "secondary": 5}

        mode = str(self.policy_config.get("weight_mode", "policy"))
        weights, weights_source = load_weights(self.config_path, selected_category, mode)
        target_slots = list(category_policy.get("target_slots", []))
        if entities.get("item") is None:
            target_slots = [slot for slot in target_slots if slot != "item_allowed"]
        collections = ["primary"] + (["secondary"] if top_k["secondary"] else [])
        trace = {
            "tvs_source": tvs_result["tvs_source"],
            "weights_source": weights_source,
            "flags": list(dict.fromkeys(flags)),
        }

        return {
            "category": selected_category,
            "tvs": score,
            "tvs_source": tvs_result["tvs_source"],
            "volatility_score": score,
            "volatility": volatility,
            "volatility_level": volatility,
            "freshness_half_life_days": int(category_policy["freshness_half_life_days"]),
            "strategy": strategy,
            "evidence_strategy": strategy,
            "live_requested": live_active,
            "retrieval_policy": {
                "collections": collections,
                "top_k": top_k,
                "tau_rel": float(self.policy_config.get("tau_rel", 0.45)),
            },
            "target_slots": target_slots,
            "weights": weights,
            "weight_mode": mode,
            "weights_source": weights_source,
            "flags": trace["flags"],
            "trace": trace,
        }