import json

import yaml

from trustrag.offline.metadata import init_metadata_store
from trustrag.online.live.fetcher import LiveCache, LiveFetcher
from trustrag.online.live.google import GoogleSearchFallback
from trustrag.online.live.govuk import GovUkSource
from trustrag.online.retrieval.evidence import extract_claims, scope_matches
from trustrag.online.retrieval.search import OnlineRetriever


def test_govuk_parser_separates_passport_and_scope_independent_sections() -> None:
    payload = {
        "title": "Singapore travel advice",
        "public_updated_at": "2026-03-19T14:08:04Z",
        "details": {
            "country": {"name": "Singapore", "slug": "singapore"},
            "parts": [
                {
                    "slug": "entry-requirements",
                    "title": "Entry requirements",
                    "body": "<h2 id='visa-requirements'>Visa requirements</h2><p>British citizens may visit visa free.</p><h2 id='customs-rules'>Customs rules</h2><p>Medicines require approval before import.</p>",
                },
                {
                    "slug": "health",
                    "title": "Health",
                    "body": "<h2 id='medication'>Medication</h2><p>Check approval to bring medicine.</p>",
                },
            ],
        },
    }

    documents = GovUkSource.parse_page(payload, "customs_regulation")

    assert documents
    assert all(document.country == "Singapore" for document in documents)
    assert any(document.nationality_scope == "ALL" for document in documents)
    assert all(document.source_type == "foreign_gov_advisory" for document in documents)
    assert scope_matches("visa_type", "GB", "IN") is False
    assert scope_matches("item_allowed", "GB", "IN") is True


def test_govuk_entry_advice_is_scoped_to_british_passports() -> None:
    payload = {
        "title": "Vietnam travel advice",
        "public_updated_at": "2026-09-21T13:25:10Z",
        "details": {
            "country": {"name": "Vietnam", "slug": "vietnam"},
            "parts": [{"slug": "entry-requirements", "title": "Entry requirements", "body": "<p>You can visit without a visa for up to 45 days.</p>"}],
        },
    }

    documents = GovUkSource.parse_page(payload, "visa_policy")

    assert len(documents) == 1
    assert documents[0].nationality_scope == "GB"
    assert not scope_matches("max_stay_days", documents[0].nationality_scope, "India")


def test_govuk_country_slug_lookup_supports_nested_index_shapes() -> None:
    payload = {
        "details": {
            "children": [
                {"details": {"country": {"name": "Singapore", "slug": "singapore", "synonyms": []}}}
            ]
        }
    }
    assert GovUkSource._find_country_slug("Singapore", payload) == "singapore"


def test_live_fetcher_merges_scoped_govuk_and_whitelist_evidence(tmp_path) -> None:
    metadata_path = tmp_path / "metadata.sqlite"
    init_metadata_store(metadata_path).close()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "paths": {"metadata_db": str(metadata_path)},
                "live": {
                    "enabled": True,
                    "total_timeout_s": 3,
                    "per_adapter_timeout_s": 2,
                    "govuk": {"enabled": True},
                    "whitelist": {
                        "Singapore": {
                            "customs_regulation": [
                                {"url": "https://www.customs.gov.sg/medicines", "source_type": "government", "nationality_scope": "ALL"}
                            ]
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    govuk_index = {
        "children": [{"details": {"country": {"name": "Singapore", "slug": "singapore", "synonyms": []}}}]
    }
    govuk_page = {
        "title": "Singapore travel advice",
        "web_url": "https://www.gov.uk/foreign-travel-advice/singapore",
        "public_updated_at": "2026-03-19T14:08:04Z",
        "details": {
            "country": {"name": "Singapore", "slug": "singapore"},
            "parts": [
                {"slug": "entry-requirements", "title": "Entry requirements", "body": "<h2>Customs rules</h2><p>Controlled medicines require approval before import.</p>"},
                {"slug": "health", "title": "Health", "body": "<h2>Medication</h2><p>Some medicines are restricted and require approval.</p>"},
            ],
        },
    }
    custom_page = b"<html><body><h1>Medicines</h1><p>Controlled medicines require authorization before import.</p></body></html>"
    requests = []

    def fake_http_get(url: str, timeout: float) -> bytes:
        requests.append(url)
        if url.endswith("foreign-travel-advice"):
            return json.dumps(govuk_index).encode()
        if url.endswith("foreign-travel-advice/singapore"):
            return json.dumps(govuk_page).encode()
        return custom_page

    fetcher = LiveFetcher(
        config_path,
        http_get=fake_http_get,
        robots_allowed=lambda _: True,
        cache=LiveCache(metadata_path),
        min_interval_s=0,
    )
    policy = {
        "category": "customs_regulation",
        "entities": {"destination": "Singapore", "nationality": "India", "item": "medicine"},
    }

    documents, flags = fetcher.fetch(policy, live_requested=True)
    claims = []
    for index, document in enumerate(documents):
        doc = document.as_dict()
        claims.extend(
            extract_claims(
                chunk={"chunk_id": f"live-{index}", "text": doc["text"], "score": 1.0},
                document={
                    "document_id": f"live-{index}",
                    "source_domain": doc["source_domain"],
                    "source_type": doc["source_type"],
                    "is_primary": doc["nationality_scope"] == "ALL",
                    "nationality_scope": doc["nationality_scope"],
                    "title": doc["title"],
                    "url": doc["url"],
                    "crawl_timestamp": doc["fetched_at"],
                    "last_verified_ts": doc["fetched_at"],
                },
                slots=["item_allowed"],
                destination="Singapore",
                item="medicine",
            )
        )

    assert documents
    assert flags == []
    assert claims
    assert all(claim["live"] if "live" in claim else True for claim in claims)
    assert any("customs.gov.sg" in request for request in requests)


def test_benchmark_live_fetcher_makes_no_http_calls(tmp_path) -> None:
    metadata_path = tmp_path / "metadata.sqlite"
    init_metadata_store(metadata_path).close()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"paths": {"metadata_db": str(metadata_path)}, "live": {"enabled": True, "govuk": {"enabled": True}}}),
        encoding="utf-8",
    )
    calls = []
    fetcher = LiveFetcher(config_path, http_get=lambda url, timeout: calls.append(url) or b"{}")
    documents, flags = fetcher.fetch(
        {"category": "visa_policy", "entities": {"destination": "Vietnam", "nationality": "India"}},
        mode="benchmark",
        live_requested=True,
    )
    assert documents == []
    assert flags == []
    assert calls == []


def test_retrieval_drops_british_visa_scope_but_keeps_all_scope_customs() -> None:
    class FakeStore:
        chunks = {
            "uk-visa": {"chunk_id": "uk-visa", "text": "British nationals may stay for 90 days using an e-visa."},
            "all-customs": {"chunk_id": "all-customs", "text": "Some medicines are restricted and require approval before import."},
        }
        documents = {
            "uk-visa": {
                "document_id": "uk-visa", "title": "FCDO entry advice", "url": "https://gov.uk/advice",
                "source_domain": "www.gov.uk", "source_type": "foreign_gov_advisory", "is_primary": False,
                "nationality_scope": "GB", "country": "Vietnam", "status": "active",
            },
            "all-customs": {
                "document_id": "all-customs", "title": "Singapore medicine rules", "url": "https://gov.sg/customs",
                "source_domain": "customs.gov.sg", "source_type": "government", "is_primary": True,
                "nationality_scope": "ALL", "country": "Vietnam", "status": "active",
            },
        }

        def search(self, query_vec, collection, k):
            return [{"chunk_id": key, "document_id": key, "score": 0.9} for key in self.chunks]

        def get_chunk(self, chunk_id):
            return self.chunks[chunk_id]

        def get_document(self, document_id):
            return self.documents[document_id]

    policy = {
        "category": "customs_regulation",
        "entities": {"destination": "Vietnam", "nationality": "India", "item": "medicine"},
        "target_slots": ["visa_type", "max_stay_days", "item_allowed"],
        "retrieval_policy": {"collections": ["primary"], "top_k": {"primary": 5}, "tau_rel": 0.45},
    }
    result = OnlineRetriever(FakeStore()).retrieve(policy=policy, query_vec=None)

    assert [item["slot"] for item in result["evidence"]] == ["item_allowed"]
    assert result["trace"]["scope_mismatch_dropped"] > 0
    assert "scope_mismatch_dropped" in result["flags"]


def test_google_fallback_results_are_marked_unverified_and_unknown_scope() -> None:
    documents = GoogleSearchFallback.parse_results(
        {
            "items": [
                {
                    "title": "Official visa checker",
                    "link": "https://www.gov.uk/check-uk-visa",
                    "snippet": "Check if you need a UK visa based on your nationality.",
                },
                {"title": "Blog", "link": "https://example.com/advice", "snippet": "A travel blog result."},
            ]
        },
        category="visa_policy",
        destination="United Kingdom",
        nationality="India",
    )

    assert len(documents) == 2
    assert documents[0].source_type == "government"
    assert documents[0].nationality_scope == "UNKNOWN"
    assert documents[1].source_type == "unknown"
    assert documents[1].nationality_scope == "UNKNOWN"


def test_google_fallback_without_credentials_is_explicitly_unavailable() -> None:
    documents, flags = GoogleSearchFallback(api_key="", search_engine_id="").search(
        query="UK visa", destination="United Kingdom", nationality="India", category="visa_policy"
    )
    assert documents == []
    assert flags == ["google_fallback_unavailable:missing_credentials"]