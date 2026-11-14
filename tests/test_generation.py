from trustrag.online.generation.llm import MockLLM
from trustrag.online.live.models import LiveDocument
from trustrag.pipeline import TrustRAGPipeline
from trustrag.online.generation.verify import verify_answer


def test_mock_llm_is_deterministic() -> None:
    llm = MockLLM()
    payload = {
        "query": "How long can Indian travellers stay in Vietnam?",
        "entities": {"nationality": "India", "destination": "Vietnam"},
        "facts": [{"slot": "max_stay_days", "value": 30, "evidence_id": "E1"}],
    }

    first = llm.generate("system", payload)
    second = llm.generate("system", payload)

    assert first.text == second.text
    assert "30 days" in first.text
    assert "[E1]" in first.text


def test_pipeline_returns_verified_cited_mock_answer() -> None:
    pipeline = TrustRAGPipeline("trustrag/config.yaml", llm=MockLLM())
    response = pipeline.answer(
        "How long can Indian travellers stay in Vietnam?",
        system="S2",
        llm="mock",
        as_of="2026-11-09",
    )

    assert response["verification_status"] == "VERIFIED"
    assert response["system_name"] == "TrustRAG (policy weights)"
    assert response["faithfulness"] == 1.0
    assert "30 days" in response["answer"]
    assert len(response["sources"]) == 1
    assert response["confidence"] < 0.10
    assert response["trace"]["generations"] == 1


def test_hallucinating_mock_is_bounded_to_one_retry() -> None:
    pipeline = TrustRAGPipeline("trustrag/config.yaml", llm=MockLLM(hallucinate=True))
    response = pipeline.answer(
        "How long can Indian travellers stay in Vietnam?",
        system="S2",
        llm="mock",
        as_of="2026-11-09",
    )

    assert response["verification_status"] == "VERIFICATION_REQUIRED"
    assert response["verification_required"] is True
    assert response["trace"]["generations"] <= 2
    assert response["trace"]["verifications"] <= 2


def test_out_of_scope_query_skips_retrieval() -> None:
    pipeline = TrustRAGPipeline("trustrag/config.yaml", llm=MockLLM())
    response = pipeline.answer("Best beaches in Bali", system="S2", llm="mock")

    assert response["verification_status"] == "OUT_OF_SCOPE"
    assert response["trace"]["retrieval_calls"] == 0


def test_baseline_skips_trust_and_verification() -> None:
    pipeline = TrustRAGPipeline("trustrag/config.yaml", llm=MockLLM())
    response = pipeline.answer(
        "How long can Indian travellers stay in Vietnam?", system="S0", llm="mock"
    )

    assert response["verification_status"] == "UNVERIFIED"
    assert response["faithfulness"] is None
    assert response["trace"]["trust_calls"] == 0
    assert response["trace"]["verifications"] == 0


def test_s3_missing_learned_weights_returns_unavailable_response() -> None:
    pipeline = TrustRAGPipeline("trustrag/config.yaml", llm=MockLLM())
    response = pipeline.answer(
        "Do I need a visa for Vietnam?", system="S3", llm="mock", as_of="2026-11-09"
    )

    assert response["verification_status"] == "SYSTEM_UNAVAILABLE"
    assert "learned weights" in response["answer"].lower()
    assert response["trace"]["retrieval_calls"] == 0


def test_api_ask_schema_health_and_admin_auth() -> None:
    from fastapi.testclient import TestClient

    from trustrag.api import create_app

    client = TestClient(create_app("trustrag/config.yaml"))
    health = client.get("/health")
    answer = client.post(
        "/ask",
        json={"query": "How long can Indian travellers stay in Vietnam?", "llm": "mock", "as_of": "2026-11-09"},
    )
    reindex = client.post("/admin/reindex")

    assert health.status_code == 200
    assert answer.status_code == 200
    assert answer.json()["verification_status"] == "VERIFIED"
    assert reindex.status_code == 401


def test_qualified_medicine_restriction_claim_verifies_against_live_text() -> None:
    evidence = [{
        "citation_id": "E1",
        "slot": "item_allowed",
        "value_norm": "restricted",
        "chunk_text": "Some medicines are restricted and require approval before import.",
    }]
    result = verify_answer(
        "Some medicines are restricted or require approval; check the rule for your specific medicine [E1].",
        evidence,
        [{"slot": "item_allowed", "value": "restricted"}],
    )
    assert result["claims"][0]["label"] == "SUPPORTED"


def test_mock_llm_uses_qualified_medicine_approval_wording() -> None:
    result = MockLLM().generate(
        "system",
        {
            "destination": "Singapore",
            "nationality": "India",
            "facts": [{"slot": "item_allowed", "value": "restricted", "evidence_id": "E1"}],
            "evidence": [{
                "evidence_id": "E1",
                "text": "Check whether approval is needed to bring medication into Singapore.",
            }],
        },
    )
    assert "approval" in result.text.lower()
    assert "specific medicine" in result.text.lower()
    assert "medicines are restricted" not in result.text.lower()


def test_pipeline_uses_mocked_live_medicine_evidence(monkeypatch) -> None:
    live_document = LiveDocument(
        url="https://www.gov.uk/foreign-travel-advice/singapore#medication",
        title="Singapore medication advice",
        source_domain="www.gov.uk",
        source_type="foreign_gov_advisory",
        nationality_scope="ALL",
        country="Singapore",
        category="customs_regulation",
        content_date="2026-03-19",
        fetched_at="2026-11-09T12:00:00+00:00",
        text=(
            "Check whether approval is needed to bring medication into Singapore. "
            "The legal status and regulation of some medicines differs by product. "
            "Contact the Health Sciences Authority to confirm requirements for your specific medicine."
        ),
        section="health:medication",
        raw_hash="a" * 64,
    )

    class FakeLiveFetcher:
        def __init__(self, config_path):
            pass

        def fetch(self, policy, *, mode="demo", live_requested=True):
            assert policy["entities"]["destination"] == "Singapore"
            return [live_document], []

    monkeypatch.setattr("trustrag.pipeline.LiveFetcher", FakeLiveFetcher)
    response = TrustRAGPipeline("trustrag/config.yaml", llm=MockLLM()).answer(
        "Can I carry medicines into Singapore?",
        system="S2",
        llm="mock",
        as_of="2026-11-09",
        live_check=True,
    )

    assert response["verification_status"] == "VERIFIED"
    assert "approval" in response["answer"].lower()
    assert response["trace"]["live_documents"] == 1
    assert response["sources"][0]["live"] is True


def test_uk_visa_query_links_official_checker_without_guessing(monkeypatch) -> None:
    live_document = LiveDocument(
        url="https://www.gov.uk/check-uk-visa",
        title="GOV.UK Check if you need a UK visa",
        source_domain="www.gov.uk",
        source_type="government",
        nationality_scope="ALL",
        country="United Kingdom",
        category="visa_policy",
        content_date=None,
        fetched_at="2026-11-09T12:00:00+00:00",
        text="You may need a visa to visit the UK depending on your nationality. Check the official visa checker.",
        section="visa_policy",
        raw_hash="b" * 64,
    )

    class FakeLiveFetcher:
        def __init__(self, config_path):
            pass

        def fetch(self, policy, *, mode="demo", live_requested=True):
            return [live_document], []

    monkeypatch.setattr("trustrag.pipeline.LiveFetcher", FakeLiveFetcher)
    response = TrustRAGPipeline("trustrag/config.yaml", llm=MockLLM()).answer(
        "Do I need Visa for UK Travel?",
        system="S2",
        llm="mock",
        as_of="2026-11-09",
        live_check=True,
    )

    assert response["verification_status"] == "VERIFICATION_REQUIRED"
    assert "assuming an indian passport" in response["answer"].lower()
    assert "https://www.gov.uk/check-uk-visa" == response["sources"][0]["url"]
    assert response["sources"][0]["live"] is True
    assert "nationality_defaulted" in response["flags"]

def test_each_trust_system_scores_with_its_own_weight_vector() -> None:
    pipeline = TrustRAGPipeline("trustrag/config.yaml", llm=MockLLM())
    used = {}
    for system, expected in (("S1", "weights_static.yaml:default"), ("S2", "weights_policy.yaml:visa_policy")):
        response = pipeline.answer(
            "How long can Indian travellers stay in Vietnam?", system=system, llm="mock", as_of="2026-11-09"
        )
        evidence = response["trace"]["trust"]["evidence"]
        assert response["trace"]["weights_source"] == expected
        used[system] = evidence[0]["weights"]
    assert used["S1"] != used["S2"]
