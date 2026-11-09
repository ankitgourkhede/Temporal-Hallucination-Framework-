from pathlib import Path

import pytest

from trustrag.offline.change_detect import compute_diff_magnitude, hash_content
from trustrag.offline.embed_index import OfflineFAISSIndex
from trustrag.offline.ingest import _parse_content_date, ingest_corpus
from trustrag.offline.metadata import DocumentMetadata, init_metadata_store, upsert_document
from trustrag.offline.knowledge_store import KnowledgeStore
from trustrag.offline.reputation import SourceReputationStore
from trustrag.offline.scheduler import OfflineScheduler


def test_seed_corpus_ingestion() -> None:
    docs = ingest_corpus("data/corpus")
    assert len(docs) >= 3
    assert all(doc.content_hash for doc in docs)
    assert all(doc.source_type for doc in docs)
    assert any(doc.country == "Vietnam" and doc.category == "vaccination_requirement" for doc in docs)
    assert any(doc.country == "India" and doc.category == "customs_regulation" for doc in docs)
    assert next(doc for doc in docs if doc.category == "visa_policy" and doc.source_type == "government").nationality_scope == "IN"


def test_change_detection_is_normalized() -> None:
    old = "This is a government visa notice."
    new = "This is a government visa notice with a new amount."
    diff = compute_diff_magnitude(old, new)
    assert 0.0 < diff < 1.0
    assert hash_content(old) != hash_content(new)


def test_content_date_parser_rejects_future_dates() -> None:
    assert _parse_content_date("Last updated: 15 July 2026", {}, "2026-11-06") == "2026-07-15"
    assert _parse_content_date("Updated on: 2027-01-15", {}, "2026-11-06") is None
    assert _parse_content_date("No date here", {}, "2026-11-06") is None


def test_metadata_store_has_phase_one_contract_tables(tmp_path: Path) -> None:
    store = init_metadata_store(tmp_path / "trustrag.sqlite")
    tables = {
        row[0]
        for row in store.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    }

    assert {"documents", "document_versions", "chunks", "change_events", "reputation"} <= tables


def test_document_versions_and_change_events_are_idempotent(tmp_path: Path) -> None:
    store = init_metadata_store(tmp_path / "trustrag.sqlite")
    baseline = DocumentMetadata(
        document_id="stable-doc-id",
        doc_key="vn_visa_gov_01",
        title="Vietnam visa rule",
        url="https://evisa.example.gov.vn/info",
        source_domain="evisa.example.gov.vn",
        source_type="government",
        authority_level=1.0,
        is_primary=True,
        last_verified_ts="2026-11-06",
        published_ts="2026-08-15",
        crawl_timestamp="2026-11-06T02:30:00Z",
        diff_magnitude=1.0,
        content_hash=hash_content("Stay for 30 days."),
        embedding_version="test|norm=l2|v1",
        category="visa_policy",
        country="Vietnam",
        content="Stay for 30 days.",
        content_date="2026-08-15",
        snapshot_date="2026-11-06",
    )
    upsert_document(store, baseline)
    upsert_document(store, baseline.model_copy(update={"last_verified_ts": "2026-11-09"}))

    versions = store.execute(
        "SELECT COUNT(*) FROM document_versions WHERE document_id = 'stable-doc-id'"
    ).fetchone()[0]
    assert versions == 1
    assert store.execute(
        "SELECT last_verified_ts FROM documents WHERE document_id = 'stable-doc-id'"
    ).fetchone()[0] == "2026-11-09"

    revised_text = "Stay for up to 90 days with single or multiple entry."
    upsert_document(
        store,
        baseline.model_copy(
            update={"content": revised_text, "content_hash": hash_content(revised_text)}
        ),
    )
    versions = store.execute(
        "SELECT COUNT(*) FROM document_versions WHERE document_id = 'stable-doc-id'"
    ).fetchone()[0]
    events = store.execute(
        "SELECT COUNT(*) FROM change_events WHERE document_id = 'stable-doc-id'"
    ).fetchone()[0]
    assert versions == 2
    assert events == 1


def test_source_reputation_is_train_only_and_beta_smoothed(tmp_path: Path) -> None:
    reputation = SourceReputationStore(tmp_path / "reputation.sqlite")
    with pytest.raises(ValueError, match="train split"):
        reputation.update("example.gov.vn", True, split="test")

    reputation.update("example.gov.vn", True, split="train")
    assert reputation.counts("example.gov.vn") == (1, 1)
    assert reputation.score("example.gov.vn") == 0.6
    reputation.reset_train()
    assert reputation.counts("example.gov.vn") == (0, 0)
    assert reputation.score("example.gov.vn") == 0.5


def test_scheduler_is_disabled_by_default_but_runs_on_demand() -> None:
    calls = []
    scheduler = OfflineScheduler.from_config("trustrag/config.yaml", lambda: calls.append("run"))

    assert scheduler.enabled is False
    assert scheduler.scheduler.running is False
    scheduler.run_once()
    assert calls == ["run"]


def test_offline_index_build_and_search() -> None:
    indexer = OfflineFAISSIndex.from_config("trustrag/config.yaml")
    result = indexer.build_from_directory("data/corpus")
    hits = indexer.search("How long can Indian travellers stay in Vietnam?", top_k=3)

    assert result["documents"] >= 3
    assert result["chunks"] >= 3
    assert hits
    assert hits[0]["source_type"] == "government"
    assert any("Vietnam" in hit["country"] or "Vietnam" in hit["title"] for hit in hits)

    store = KnowledgeStore.from_config("trustrag/config.yaml")
    primary_hits = store.search(indexer.encode_query("Vietnam visa duration"), "primary", 3)
    assert primary_hits
    chunk = store.get_chunk(primary_hits[0]["chunk_id"])
    document = store.get_document(primary_hits[0]["document_id"])
    assert chunk["text"]
    assert document["source_type"] == "government"
    assert store.change_history("visa_policy", "Vietnam") == []
    assert store.reputation(document["source_domain"]) == (0, 0)
