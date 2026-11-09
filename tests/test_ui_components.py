"""The chatbot answer card renders every element the build prompt's UI section asks for."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ui"))

from components import answer_html, badges_html, conflicts_html, sources_html, trust_breakdown_rows  # noqa: E402

RESPONSE = {
    "system": "S2",
    "answer": "Indians need an eVisa [E1]. Visa on arrival is available at all airports [E3].",
    "verification_status": "PARTIAL",
    "faithfulness": 0.8,
    "confidence": 0.71,
    "freshness": {"most_recent_source": "2026-11-06", "age_days": 3},
    "category": "visa_policy",
    "volatility": "High",
    "agreement": "High",
    "evidence_strategy": "PRIMARY_AND_VALIDATION",
    "unsupported_claims": [{"text": "Visa on arrival is available at all airports", "label": "CONTRADICTED"}],
    "sources": [
        {"evidence_id": "E1, E2", "title": "Immigration portal", "url": "https://x.gov.vn", "source_type": "government",
         "trust": 0.83, "cited": True, "nationality_scope": "ALL"},
        {"evidence_id": None, "title": "Old blog <script>", "url": "", "source_type": "travel_blog",
         "trust": 0.02, "cited": False, "in_context": False, "nationality_scope": "UNKNOWN"},
    ],
    "conflicts": [{"slot": "max_stay_days", "chosen": 90, "value": {"value": 30, "support": 0.024},
                   "support": {"90": 2.139, "30": 0.024}}],
    "trace": {"trust": {"evidence": [
        {"source_domain": "x.gov.vn", "slot": "max_stay_days", "value_norm": 90, "A": 1, "F": 0.966, "G": 0.833,
         "qe_components": {"a": 1, "b": 1, "c": 0.95, "d": 1}, "SR": 0.85, "CP": 0.083, "Trust": 0.832},
    ]}},
}


def test_partial_answer_highlights_unsupported_claim() -> None:
    assert "<mark" in answer_html(RESPONSE) and "<sup>[E1]</sup>" in answer_html(RESPONSE)


def test_badges_show_status_confidence_freshness_category() -> None:
    html = badges_html(RESPONSE)
    for text in ("Partially verified", "faithfulness 0.80", "Confidence 71%", "Updated 3 days ago", "high volatility"):
        assert text in html


def test_sources_ranked_with_bars_and_weak_rows_greyed_and_escaped() -> None:
    html = sources_html(RESPONSE["sources"])
    assert "[E1, E2]" in html and "tr-bar" in html and 'class="tr-weak"' in html
    assert "<script>" not in html and "not used: low trust" in html


def test_conflict_panel_is_plain_language() -> None:
    html = conflicts_html(RESPONSE["conflicts"])
    assert "max stay" in html and "<b>90 days</b>" in html and "30 days" in html


def test_trust_breakdown_rows_have_all_six_features() -> None:
    row = trust_breakdown_rows(RESPONSE)[0]
    assert {"A authority", "F freshness", "G agreement", "QE quality", "SR reputation", "CP conflict", "Trust"} <= set(row)
    assert row["QE quality"] == 0.988
