from trustrag.online.policy.engine import PolicyEngine
from trustrag.online.policy.volatility import compute_tvs
from trustrag.online.query.intent import QueryUnderstanding


def test_query_understanding_categories_and_entities() -> None:
    analyzer = QueryUnderstanding.from_config("trustrag/config.yaml")
    result = analyzer.analyze("Do Indians need a visa for Vietnam?")

    assert result["intent"] == "travel_entry"
    assert result["category"] == "visa_policy"
    assert result["entities"]["destination"] == "Vietnam"
    assert result["entities"]["nationality"] == "India"
    assert result["intent_confidence"] >= 0.5
    assert result["category_confidence"] >= 0.5


def test_policy_engine_selects_volatility_and_weights() -> None:
    engine = PolicyEngine.from_config("trustrag/config.yaml")
    decision = engine.decide("visa_policy", 0.87)

    assert decision["volatility"] in {"Low", "Medium", "High"}
    assert decision["strategy"]
    assert decision["weights"]
    assert "w_A" in decision["weights"]


def test_live_strategy_is_demo_only() -> None:
    engine = PolicyEngine.from_config("trustrag/config.yaml")
    demo = engine.decide("customs_regulation", 0.2, live_requested=True)
    benchmark = engine.decide("customs_regulation", 0.2, live_requested=True, mode="benchmark")

    assert demo["strategy"] == "LIVE_API"
    assert benchmark["strategy"] == "PRIMARY_ONLY"


def test_query_analysis_cli_summary_is_available() -> None:
    analyzer = QueryUnderstanding.from_config("trustrag/config.yaml")
    summary = analyzer.summary("Do Indians need a visa for Vietnam?")

    assert summary["intent"]
    assert summary["category"]
    assert summary["evidence_strategy"]


def test_real_travel_query_gets_policy_summary() -> None:
    analyzer = QueryUnderstanding.from_config("trustrag/config.yaml")
    summary = analyzer.summary("How long can Indian travellers stay in Vietnam?")

    assert summary["intent"] == "travel_entry"
    assert summary["category"] == "visa_policy"
    assert summary["entities"]["destination"] == "Vietnam"
    assert summary["entities"]["nationality"] == "India"
    assert summary["volatility"] == "High"
    assert summary["evidence_strategy"] == "PRIMARY_AND_VALIDATION"
    assert abs(sum(value for key, value in summary["weights"].items() if key != "w_C") - 1.0) < 1e-9


def test_health_customs_and_action_routing() -> None:
    analyzer = QueryUnderstanding.from_config("trustrag/config.yaml")
    health = analyzer.summary("Is yellow fever vaccination mandatory for Kenya?")
    customs = analyzer.summary("Can I carry a drone into Singapore?")
    out_of_scope = analyzer.summary("Best beaches in Bali")
    clarify = analyzer.summary("Do I need a visa?")

    assert health["intent"] == "health_safety"
    assert health["category"] == "vaccination_requirement"
    assert health["entities"]["destination"] == "Kenya"
    assert customs["intent"] == "customs_regulations"
    assert customs["category"] == "customs_regulation"
    assert customs["entities"]["item"] == "drone"
    assert out_of_scope["action"] == "OUT_OF_SCOPE"
    assert clarify["action"] == "CLARIFY"
    assert clarify["clarification_question"]


def test_tvs_matches_spec_example_and_expert_fallback() -> None:
    visa_events = [
        {"detected_at": "2025-11-09", "diff_magnitude": 0.35},
        {"detected_at": "2026-03-01", "diff_magnitude": 0.35},
        {"detected_at": "2026-08-01", "diff_magnitude": 0.35},
    ]
    measured = compute_tvs(visa_events, expert_prior=0.9, as_of="2026-11-09")
    fallback = compute_tvs(visa_events[:2], expert_prior=0.8, as_of="2026-11-09")

    assert measured["tvs_source"] == "change_history"
    assert measured["tvs"] == 0.66
    assert fallback["tvs_source"] == "expert"
    assert fallback["tvs"] == 0.8


def test_near_tied_categories_ask_the_user_to_clarify(monkeypatch) -> None:
    from trustrag.online.query import classifier as classifier_module

    def tied(self, query, task):
        if task == "intent":
            return [("customs_regulations", 0.50), ("health_safety", 0.45)]
        return [("customs_regulation", 0.48), ("vaccination_requirement", 0.42), ("visa_policy", 0.10)]

    monkeypatch.setattr(classifier_module.CalibratedQueryModels, "predict", tied)
    analyzer = QueryUnderstanding.from_config("trustrag/config.yaml")
    result = analyzer.analyze("How much is the duty free GST relief for travellers in Singapore?")
    assert result["action"] == "CLARIFY"
    assert "ambiguous_category" in result["flags"]
    assert "customs rules or health and vaccination requirements for Singapore" in result["clarification_question"]
