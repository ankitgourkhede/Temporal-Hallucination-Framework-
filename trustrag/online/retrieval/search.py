"""Policy-directed retrieval over primary and secondary knowledge collections."""

from __future__ import annotations

from typing import Any

from trustrag.offline.knowledge_store import KnowledgeStore
from trustrag.online.retrieval.evidence import extract_claims, scope_matches


class OnlineRetriever:
    def __init__(self, store: KnowledgeStore):
        self.store = store

    def retrieve(
        self,
        *,
        policy: dict[str, Any],
        query_vec: Any,
        live_documents: list[Any] | None = None,
    ) -> dict[str, Any]:
        retrieval = policy["retrieval_policy"]
        entities = policy.get("entities", {})
        destination = entities.get("destination")
        slots = list(policy.get("target_slots", []))
        threshold = float(retrieval.get("tau_rel", 0.45))
        evidence: list[dict[str, Any]] = []
        primary_slots: set[str] = set()
        scope_mismatch_dropped = 0

        for collection in retrieval.get("collections", ["primary"]):
            k = int(retrieval.get("top_k", {}).get(collection, 0))
            if k <= 0:
                continue
            for hit in self.store.search(query_vec, collection, k):
                chunk = self.store.get_chunk(hit["chunk_id"])
                document = self.store.get_document(hit["document_id"])
                if chunk is None or document is None or document.get("status") != "active":
                    continue
                if float(hit["score"]) < threshold:
                    continue
                if destination and str(document.get("country", "")).casefold() != str(destination).casefold():
                    continue
                chunk_with_score = {**chunk, "score": hit["score"]}
                claims = extract_claims(
                    chunk=chunk_with_score,
                    document=document,
                    slots=slots,
                    destination=destination or "",
                    item=entities.get("item"),
                )
                for claim in claims:
                    if not scope_matches(
                        claim["slot"], document.get("nationality_scope", "UNKNOWN"), entities.get("nationality")
                    ):
                        scope_mismatch_dropped += 1
                        continue
                    if collection == "primary":
                        primary_slots.add(claim["slot"])
                    evidence.append(claim)

        for live_document in live_documents or []:
            live = live_document.as_dict() if hasattr(live_document, "as_dict") else dict(live_document)
            source_type = str(live.get("source_type") or "unknown")
            scope = live.get("nationality_scope", "UNKNOWN")
            search_snippet = live.get("section") == "google_search_snippet"
            # A search-result snippet is not a verified official page: it is truncated, undated and
            # out of context. It keeps its domain's Authority tier, but is never treated as primary, so
            # Freshness falls back to the capped no-date rule and Severity treats it as secondary.
            is_primary = not search_snippet and (
                source_type in {"government", "embassy", "intl_body"}
                or (source_type == "foreign_gov_advisory" and scope == "ALL")
            )
            document = {
                "document_id": f"live-{live['raw_hash'][:16]}",
                "title": live.get("title", "Live official source"),
                "url": live["url"],
                "source_domain": live.get("source_domain", "unknown"),
                "source_type": source_type,
                "authority_level": 0.9 if source_type == "foreign_gov_advisory" else 1.0,
                "is_primary": is_primary,
                "nationality_scope": scope,
                "country": live.get("country", destination or ""),
                "category": live.get("category", policy["category"]),
                "content_date": live.get("content_date"),
                "last_verified_ts": live.get("fetched_at") if is_primary else None,
                "crawl_timestamp": live.get("fetched_at"),
                "status": "active",
            }
            chunk = {
                "chunk_id": f"{document['document_id']}-chunk-0",
                "text": live.get("text", ""),
                "chunk_text": live.get("text", ""),
            }
            claims = extract_claims(
                chunk={**chunk, "score": 1.0},
                document=document,
                slots=slots,
                destination=destination or "",
                item=entities.get("item"),
            )
            for claim in claims:
                if not scope_matches(claim["slot"], scope, entities.get("nationality")):
                    scope_mismatch_dropped += 1
                    continue
                claim["live"] = live.get("section") != "google_search_snippet"
                claim["search_fallback"] = live.get("section") == "google_search_snippet"
                claim["fetched_at"] = live.get("fetched_at")
                if is_primary:
                    primary_slots.add(claim["slot"])
                evidence.append(claim)

        flags = list(policy.get("flags", []))
        secondary_slots = {item["slot"] for item in evidence if not item["is_primary"]}
        no_primary_slots = sorted(secondary_slots - primary_slots)
        if no_primary_slots:
            flags.append("no_primary_evidence")
        if scope_mismatch_dropped:
            flags.append("scope_mismatch_dropped")
        return {
            "evidence": evidence,
            "flags": list(dict.fromkeys(flags)),
            "trace": {
                "tau_rel": threshold,
                "target_slots": slots,
                "retrieved_evidence_count": len(evidence),
                "no_primary_evidence_slots": no_primary_slots,
                "scope_mismatch_dropped": scope_mismatch_dropped,
                "live_evidence_count": sum(bool(item.get("live")) for item in evidence),
            },
        }