from datetime import date

import pytest

from trustrag.offline.embed_index import OfflineFAISSIndex, encode_query_vector
from trustrag.offline.knowledge_store import KnowledgeStore
from trustrag.online.retrieval.evidence import normalize_days, normalize_enum, normalize_money, scope_matches
from trustrag.online.retrieval.search import OnlineRetriever
from trustrag.online.trust.fusion import fuse_slots
from trustrag.online.trust.pipeline import TrustPipeline
from trustrag.online.trust.scoring import score_evidence
from trustrag.online.query.intent import QueryUnderstanding


def test_golden_trust_example_and_slot_fusion() -> None:
    full_text = (
        "Vietnam immigration policy states that the maximum stay duration is 90 days. "
        "Travellers should review the official Vietnam visa requirements and passport rules "
        "before travel. This notice explains the stay period and entry conditions in detail."
    )
    evidence = [
        {
            "evidence_id": "E1", "document_id": "d1", "source_domain": "immigration.gov.vn",
            "source_type": "government", "is_primary": True, "slot": "max_stay_days",
            "destination": "Vietnam", "value_norm": 90, "content_date": "2026-08-15",
            "last_verified_ts": "2026-11-06", "crawl_timestamp": "2026-11-06T02:30:00Z",
            "extraction_confidence": 0.95, "hedged": False, "chunk_text": full_text,
            "relevance": 0.9,
        },
        {
            "evidence_id": "E2", "document_id": "d2", "source_domain": "embassy.gov.in",
            "source_type": "embassy", "is_primary": True, "slot": "max_stay_days",
            "destination": "Vietnam", "value_norm": 90, "content_date": "2026-08-15",
            "last_verified_ts": "2026-10-20", "crawl_timestamp": "2026-10-20T02:30:00Z",
            "extraction_confidence": 0.9, "hedged": False, "chunk_text": full_text,
            "relevance": 0.8,
        },
        {
            "evidence_id": "E3", "document_id": "d3", "source_domain": "airline.example",
            "source_type": "airline", "is_primary": False, "slot": "max_stay_days",
            "destination": "Vietnam", "value_norm": 90, "content_date": "2026-06-01",
            "last_verified_ts": "2026-11-06", "crawl_timestamp": "2026-11-06T02:30:00Z",
            "extraction_confidence": 0.9, "hedged": False, "chunk_text": full_text,
            "relevance": 0.7,
        },
        {
            "evidence_id": "E4", "document_id": "d4", "source_domain": "travel.example",
            "source_type": "travel_blog", "is_primary": False, "slot": "max_stay_days",
            "destination": "Vietnam", "value_norm": 30, "content_date": "2023-03-15",
            "last_verified_ts": "2026-11-06", "crawl_timestamp": "2026-11-06T02:30:00Z",
            "extraction_confidence": 0.85, "hedged": True,
            "chunk_text": full_text.replace("90 days", "30 days") + " This may vary.", "relevance": 0.8,
        },
    ]
    weights = {"w_A": 0.30, "w_F": 0.30, "w_G": 0.15, "w_E": 0.10, "w_SR": 0.15, "w_C": 0.40}
    reputation = {
        "immigration.gov.vn": (16, 15),
        "embassy.gov.in": (6, 6),
        "airline.example": (6, 5),
        "travel.example": (6, 2),
    }

    scored = score_evidence(
        evidence,
        weights,
        as_of=date(2026, 11, 9),
        freshness_half_life_days=60,
        authority_path="trustrag/policies/authority.yaml",
        reputation=reputation,
    )
    fusion = fuse_slots(scored)

    assert [round(item["Trust"], 3) for item in scored] == [0.832, 0.766, 0.541, 0.024]
    assert fusion[0]["value"] == 90
    assert round(fusion[0]["confidence"], 3) == 0.962
    assert fusion[0]["conflicting_view"]["value"] == 30


def test_day_normalization_handles_days_and_months() -> None:
    assert normalize_days("stay for up to 30 days") == 30
    assert normalize_days("stay for one month") == 30
    assert normalize_days("stay for three months") == 90
    assert normalize_enum("electronic visa") == "evisa"
    assert normalize_enum("visa exemption") == "visa_free"
    assert normalize_money("USD 1,000") == (1000.0, "USD")
    assert normalize_money("₹5000") == (5000.0, "INR")


def test_nationality_scope_filters_dependent_slots_only() -> None:
    assert not scope_matches("visa_type", "GB", "India")
    assert scope_matches("max_stay_days", "IN", "India")
    assert scope_matches("item_allowed", "GB", "India")
    assert scope_matches("visa_required", "UNKNOWN", "India")
    assert scope_matches("visa_required", "ALL", "India")


def test_real_query_trust_pipeline_prefers_official_claim() -> None:
    query = "How long can Indian travellers stay in Vietnam?"
    config_path = "trustrag/config.yaml"
    indexer = OfflineFAISSIndex.from_config(config_path)
    indexer.build_from_directory("data/corpus")
    policy = QueryUnderstanding.from_config(config_path).summary(query, as_of="2026-11-09")
    query_vector = encode_query_vector(query, config_path)

    with KnowledgeStore.from_config(config_path) as store:
        retrieval = OnlineRetriever(store).retrieve(policy=policy, query_vec=query_vector)
        result = TrustPipeline(config_path, store).score(
            retrieval_result=retrieval,
            policy=policy,
            as_of="2026-11-09",
        )

    stay_slot = next(item for item in result["fusion"] if item["slot"] == "max_stay_days")
    assert stay_slot["value"] == 30
    assert stay_slot["conflicting_view"]["value"] == 45
    assert next(item for item in result["sources"] if item["source_type"] == "government")["trust"] > next(
        item for item in result["sources"] if item["source_type"] == "travel_blog"
    )["trust"]
    indexer.metadata_store.close()

def test_content_relative_freshness_exposes_unrevised_primary_page() -> None:
    """A recrawled but never-revised official page must lose freshness to a newer notice."""
    base = {"slot": "visa_required", "destination": "Sri Lanka", "is_primary": True,
            "last_verified_ts": "2026-11-01", "crawl_timestamp": "2026-11-01",
            "extraction_confidence": 1.0, "chunk_text": "", "nationality_scope": "IN"}
    unrevised = {**base, "evidence_id": "old", "value_norm": True, "source_type": "government",
                 "source_domain": "eta.gov.lk.example", "content_date": "2023-04-01"}
    current = {**base, "evidence_id": "new", "value_norm": False, "source_type": "embassy",
               "source_domain": "highcommission.lk.example", "content_date": "2026-02-10"}
    weights = {"w_A": 0.30, "w_F": 0.30, "w_G": 0.15, "w_E": 0.10, "w_SR": 0.15, "w_C": 0.40}

    legacy = {i["evidence_id"]: i for i in score_evidence([unrevised, current], weights, as_of="2026-11-09",
                                                          freshness_half_life_days=60)}
    assert legacy["old"]["F"] == pytest.approx(legacy["new"]["F"])

    relative = {i["evidence_id"]: i for i in score_evidence([unrevised, current], weights, as_of="2026-11-09",
                                                            freshness_half_life_days=60,
                                                            freshness_mode="content_relative")}
    assert relative["new"]["F"] == pytest.approx(1.0)
    assert relative["old"]["F"] < 0.01
    assert relative["new"]["Trust"] > relative["old"]["Trust"]
