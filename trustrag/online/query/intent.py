"""Query intent/category fallback, entity extraction, and policy routing."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from trustrag.online.query.classifier import CalibratedQueryModels


class QueryUnderstanding:
    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        with self.config_path.open("r", encoding="utf-8") as handle:
            self.config = yaml.safe_load(handle) or {}

        policy_config = self.config.get("policy", {})
        query_path = self.config_path.parent / policy_config.get("query_file", "policies/query.yaml")
        with query_path.open("r", encoding="utf-8") as handle:
            self.rules = yaml.safe_load(handle) or {}

        categories_path = self.config_path.parent / policy_config.get(
            "categories_file", "policies/categories.yaml"
        )
        with categories_path.open("r", encoding="utf-8") as handle:
            self.category_policies = yaml.safe_load(handle) or {}
        self.classifier_models = CalibratedQueryModels(self.config_path)
        self.classifier_config = self.config.get("query_classifier", {})

    @classmethod
    def from_config(cls, config_path: str | Path) -> "QueryUnderstanding":
        return cls(config_path)

    @staticmethod
    def _matches(query: str, keywords: list[str]) -> list[str]:
        matches = []
        for keyword in keywords:
            pattern = rf"(?<!\w){re.escape(keyword)}(?!\w)"
            if re.search(pattern, query, flags=re.IGNORECASE):
                matches.append(keyword)
        return matches

    @classmethod
    def _classify(cls, query: str, rules: dict[str, list[str]]) -> tuple[str | None, float]:
        ranked = []
        for label, keywords in rules.items():
            matches = cls._matches(query, keywords)
            if matches:
                confidence = min(0.95, 0.60 + 0.05 * (len(matches) - 1))
                ranked.append((len(matches), confidence, label))
        if not ranked:
            return None, 0.0
        _, confidence, label = max(ranked, key=lambda item: (item[0], item[1]))
        return label, confidence

    @staticmethod
    def _extract_entity(query: str, entities: dict[str, list[str]]) -> str | None:
        matches = []
        for canonical, aliases in entities.items():
            for alias in aliases:
                pattern = rf"(?<!\w){re.escape(alias)}(?!\w)"
                match = re.search(pattern, query, flags=re.IGNORECASE)
                if match:
                    matches.append((match.start(), -len(alias), canonical))
        return min(matches)[2] if matches else None

    def _destination(self, query: str) -> str | None:
        gazetteer = self.rules.get("entities", {}).get("destinations", {})
        candidates = []
        for country, aliases in gazetteer.items():
            for alias in aliases:
                escaped = re.escape(alias)
                pattern = rf"\b(?:to|for|into|in|enter|visit)\s+({escaped})\b"
                match = re.search(pattern, query, flags=re.IGNORECASE)
                if match:
                    candidates.append((match.start(1), country))
        if candidates:
            return min(candidates)[1]

        nationality = self._extract_entity(
            query, self.rules.get("entities", {}).get("nationalities", {})
        )
        destinations = self._extract_entity(query, gazetteer)
        return None if destinations == nationality else destinations

    def _entities(self, query: str) -> tuple[dict[str, str | None], list[str]]:
        rules = self.rules.get("entities", {})
        nationality = self._extract_entity(query, rules.get("nationalities", {}))
        flags = []
        if nationality is None:
            nationality = str(self.config.get("default_nationality", "India"))
            flags.append("nationality_defaulted")
        return (
            {
                "nationality": nationality,
                "destination": self._destination(query),
                "item": self._extract_entity(query, rules.get("items", {})),
            },
            flags,
        )

    @staticmethod
    def _clarification_question(category: str, entities: dict[str, str | None]) -> str:
        if category != "unknown" and entities.get("destination") is None:
            return "Which country are you travelling to?"
        return "Are you asking about entry/visa rules, health requirements, or customs?"

    def _model_or_fallback(
        self, query: str, task: str, rules: dict[str, list[str]]
    ) -> tuple[str | None, float, list[tuple[str, float]], str]:
        ranked = self.classifier_models.predict(query, task)
        fallback_label, fallback_confidence = self._classify(query, rules)
        if not ranked:
            return fallback_label, fallback_confidence, [], "keyword_fallback"
        tie_margin = float(self.classifier_config.get("tie_margin", 0.05))
        if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < tie_margin:
            return fallback_label, fallback_confidence, ranked, "keyword_tie_break"
        return ranked[0][0], ranked[0][1], ranked, "calibrated_logistic_regression"

    def analyze(self, query: str, as_of: str | None = None) -> dict[str, Any]:
        query = query.strip()
        intent_rules = self.rules.get("intents", {})
        category_rules = self.rules.get("categories", {})
        intent, intent_confidence, intent_ranked, intent_model = self._model_or_fallback(
            query, "intent", intent_rules
        )
        category, category_confidence, category_ranked, category_model = self._model_or_fallback(
            query, "category", category_rules
        )
        category = category if category in self.category_policies else "unknown"
        category_alternates = []
        if len(category_ranked) > 1 and category_confidence < 0.55:
            if category_confidence - category_ranked[1][1] <= 0.15:
                alternate = category_ranked[1][0]
                if alternate in self.category_policies:
                    category_alternates.append(alternate)
        entities, flags = self._entities(query)
        if intent == "customs_regulations" and category != "customs_regulation":
            category, category_confidence = "customs_regulation", max(category_confidence, intent_confidence)
        elif intent == "health_safety" and category != "vaccination_requirement":
            category, category_confidence = "vaccination_requirement", max(category_confidence, intent_confidence)

        intent = intent or "out_of_scope"
        high_threshold = float(self.classifier_config.get("intent_threshold_high", 0.60))
        clarify_threshold = float(self.classifier_config.get("intent_threshold_clarify", 0.40))
        if intent == "out_of_scope" and intent_confidence >= high_threshold:
            action = "OUT_OF_SCOPE"
        elif intent_confidence < clarify_threshold or category == "unknown" or entities["destination"] is None:
            action = "CLARIFY"
        else:
            action = "RETRIEVE"
        if action == "CLARIFY":
            flags.append("destination_missing" if entities["destination"] is None else "low_intent_confidence")

        result = {
            "query": query,
            "intent": intent,
            "intent_confidence": intent_confidence,
            "category": category,
            "category_confidence": category_confidence if category != "unknown" else 0.0,
            "category_alternates": category_alternates,
            "entities": entities,
            "action": action,
            "flags": list(dict.fromkeys(flags)),
            "trace": {"intent_classifier": intent_model, "category_classifier": category_model},
            "as_of": as_of or self.config.get("project", {}).get("as_of_date"),
        }
        if action == "CLARIFY":
            result["clarification_question"] = self._clarification_question(category, entities)
        return result

    def summary(
        self,
        query: str,
        as_of: str | None = None,
        *,
        live_check: bool = False,
        mode: str = "demo",
    ) -> dict[str, Any]:
        result = self.analyze(query, as_of=as_of)
        if result["action"] != "RETRIEVE" or result["category"] == "unknown":
            result.update({"tvs": None, "volatility_level": None, "evidence_strategy": None, "weights": {}})
            return result

        from trustrag.online.policy.engine import PolicyEngine

        decision = PolicyEngine(self.config_path).decide(
            result["category"],
            intent_confidence=result["intent_confidence"],
            entities=result["entities"],
            flags=result["flags"],
            category_alternates=result["category_alternates"],
            as_of=result["as_of"],
            live_requested=live_check,
            mode=mode,
        )
        result.update(
            {
                "tvs": decision["tvs"],
                "tvs_source": decision["tvs_source"],
                "volatility": decision["volatility"],
                "volatility_level": decision["volatility_level"],
                "strategy": decision["strategy"],
                "evidence_strategy": decision["evidence_strategy"],
                "live_requested": decision["live_requested"],
                "retrieval_policy": decision["retrieval_policy"],
                "target_slots": decision["target_slots"],
                "freshness_half_life_days": decision["freshness_half_life_days"],
                "weights": decision["weights"],
                "weight_mode": decision["weight_mode"],
                "weights_source": decision["weights_source"],
                "flags": decision["flags"],
                "trace": {**result["trace"], **decision["trace"]},
            }
        )
        return result