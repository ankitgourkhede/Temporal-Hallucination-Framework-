"""Regression tests for the spec-alignment review fixes (2026-11-09)."""

from __future__ import annotations

import numpy as np

from experiments.stats import paired_bootstrap
from trustrag.offline.embed_index import _FallbackEncoder
from trustrag.online.generation.verify import verify_answer
from trustrag.online.live.google import GoogleSearchFallback


class FakeNLI:
    """Returns logits in the model's label order (contradiction, entailment, neutral)."""

    def __init__(self, label: str):
        self.label = label

    def predict(self, pairs):
        row = {"contradiction": [5.0, 0.0, 0.0], "entailment": [0.0, 5.0, 0.0], "neutral": [0.0, 0.0, 5.0]}[self.label]
        return np.asarray([row for _ in pairs])


EVIDENCE = [{"citation_id": "E1", "slot": "max_stay_days", "value_norm": 90,
             "chunk_text": "Indian nationals can stay in Vietnam for up to 90 days with an eVisa."}]
FUSION = [{"slot": "max_stay_days", "value": 90}]


def test_nli_labels_follow_phase4_thresholds() -> None:
    answer = "Indian travellers can stay up to 90 days [E1]."
    assert verify_answer(answer, EVIDENCE, FUSION, nli_model=FakeNLI("entailment"))["claims"][0]["label"] == "SUPPORTED"
    assert verify_answer(answer, EVIDENCE, FUSION, nli_model=FakeNLI("neutral"))["claims"][0]["label"] == "UNSUPPORTED"
    assert verify_answer(answer, EVIDENCE, FUSION, nli_model=FakeNLI("contradiction"))["claims"][0]["label"] == "CONTRADICTED"
    assert verify_answer(answer, EVIDENCE, FUSION, nli_model=FakeNLI("entailment"))["verifier"] == "nli"


def test_slot_check_overrides_nli_entailment() -> None:
    result = verify_answer("You can stay 45 days [E1].", EVIDENCE, FUSION, nli_model=FakeNLI("entailment"))
    assert result["claims"][0]["label"] == "CONTRADICTED"


def test_refusal_with_no_claims_is_flagged_no_answer_found() -> None:
    result = verify_answer("I couldn't find this in the evidence.", [], [], mode="rule")
    assert result["faithfulness"] == 1.0
    assert "no_answer_found" in result["flags"]


def test_rule_fallback_is_reported_when_nli_unavailable() -> None:
    result = verify_answer("Indian travellers can stay up to 90 days [E1].", EVIDENCE, FUSION, mode="rule")
    assert result["verifier"] == "rule_slot_fallback"


def test_fallback_encoder_is_lexical_not_positional() -> None:
    encoder = _FallbackEncoder(384)
    query, related, unrelated = encoder.encode([
        "vietnam visa stay days", "how many days can I stay in vietnam on a visa", "carry drones into singapore customs",
    ], batch_size=32)
    assert float(query @ related) > float(query @ unrelated)


def test_paired_bootstrap_reports_p_value() -> None:
    better = paired_bootstrap([0.0] * 30, [1.0] * 30)
    same = paired_bootstrap([0.5] * 30, [0.5] * 30)
    assert better["p_value"] < 0.05
    assert same["p_value"] == 1.0


def test_google_embassy_lookalike_domain_is_not_embassy() -> None:
    assert GoogleSearchFallback._source_type("vietnam-embassy-visa.com") == "unknown"
    assert GoogleSearchFallback._source_type("embassy.gov.au") == "embassy"
    assert GoogleSearchFallback._source_type("www.gov.uk") == "government"
