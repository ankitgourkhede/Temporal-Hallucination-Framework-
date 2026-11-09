"""Single orchestration entry point for query policy, retrieval, trust, and answers."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from trustrag.offline.embed_index import encode_query_vector
from trustrag.offline.knowledge_store import KnowledgeStore
from trustrag.online.generation.generate import build_generation_payload, generate_answer
from trustrag.online.generation.llm import LLM, build_llm
from trustrag.online.generation.verify import verify_answer
from trustrag.online.live.fetcher import LiveFetcher
from trustrag.online.live.google import GoogleSearchFallback
from trustrag.online.policy.weights import load_weights
from trustrag.online.query.intent import QueryUnderstanding
from trustrag.online.retrieval.evidence import extract_claims
from trustrag.online.retrieval.search import OnlineRetriever
from trustrag.online.trust.pipeline import TrustPipeline


class TrustRAGPipeline:
    SYSTEM_NAMES = {
        "S0": "Vanilla RAG",
        "S1": "Standard Trust RAG (static weights)",
        "S2": "TrustRAG (policy weights)",
        "S3": "TrustRAG (learned weights)",
    }

    def __init__(self, config_path: str | Path, llm: LLM | None = None):
        self.config_path = Path(config_path)
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        self.llm = llm

    def answer(
        self,
        query: str,
        system: str = "S2",
        llm: str = "ollama",
        as_of: str | None = None,
        *,
        mode: str = "demo",
        live_check: bool = False,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        if system not in {"S0", "S1", "S2", "S3"}:
            raise ValueError(f"Unsupported system: {system}")
        active_llm = self.llm or build_llm(llm, self.config)
        trace: dict[str, Any] = {
            "system": system,
            "system_name": self.SYSTEM_NAMES[system],
            "llm": llm,
            "mode": mode,
            "retrieval_calls": 0,
            "trust_calls": 0,
            "generations": 0,
            "verifications": 0,
            "layers_ms": {},
            "live_enabled": False,
        }

        layer_started = time.perf_counter()
        policy = QueryUnderstanding.from_config(self.config_path).summary(
            query, as_of=as_of, live_check=live_check, mode=mode
        )
        trace["layers_ms"]["L1_L2_policy"] = (time.perf_counter() - layer_started) * 1000
        if policy["action"] != "RETRIEVE":
            return self._finish(
                query,
                policy,
                answer=policy.get("clarification_question", "I can help with visa, health-entry, and customs rules."),
                verification_status="CLARIFICATION" if policy["action"] == "CLARIFY" else "OUT_OF_SCOPE",
                confidence=0.0,
                faithfulness=None,
                sources=[],
                unsupported_claims=[],
                flags=policy.get("flags", []),
                trace=trace,
                started=started,
            )

        weight_mode = {"S1": "static", "S2": "policy", "S3": "learned"}.get(system)
        if weight_mode is not None:
            try:
                weights, weights_source = load_weights(self.config_path, policy["category"], weight_mode)
            except FileNotFoundError as error:
                trace["system_available"] = False
                trace["unavailable_reason"] = str(error)
                return self._finish(
                    query,
                    policy,
                    answer=(
                        "Learned Trust RAG is unavailable because learned weights for this category have not been fitted. "
                    ),
                    verification_status="SYSTEM_UNAVAILABLE",
                    confidence=0.0,
                        source = next(
                            (item for item in retrieval.get("evidence", []) if item.get("live") or item.get("search_fallback")),
                            None,
                        )
        with KnowledgeStore.from_config(self.config_path) as store:
            if system == "S0":
                retrieval = self._retrieve_with_fallback(store, query, query_vector, policy, mode, live_check)
                raw_evidence = retrieval["evidence"]
                trace["retrieval_calls"] = 1
                trace["flags"] = retrieval["flags"]
                trace["live_enabled"] = bool(retrieval["trace"].get("live_documents", 0))
                trace["live_documents"] = retrieval["trace"].get("live_documents", 0)
                trace["retrieval_fallback_stages"] = retrieval["trace"].get("fallback_stages", ["offline"])
                trace["layers_ms"]["L3_retrieval"] = (time.perf_counter() - index_started) * 1000
                trust_result: dict[str, Any] = {"evidence": [], "fusion": [], "sources": [], "flags": []}
                generation_payload, context_evidence = build_generation_payload(
                    query, policy, trust_result, baseline_evidence=raw_evidence
                )
                trace["context_evidence"] = [
                    {"slot": item.get("slot"), "value": item.get("value_norm"), "evidence_id": item.get("citation_id"), "relevance": item.get("relevance"), "url": item.get("url"), "document_id": item.get("document_id")}
                    for item in context_evidence
                ]
                trace["resolved_slots"] = self._baseline_resolutions(raw_evidence)
                generation_started = time.perf_counter()
                result = generate_answer(active_llm, generation_payload, baseline=True)
                trace["generations"] = 1
                trace["layers_ms"]["L4_generation"] = (time.perf_counter() - generation_started) * 1000
                answer_text = result.text
                response_sources = self._response_sources(context_evidence, answer_text)
                return self._finish(
                    query, policy, answer=answer_text, verification_status="UNVERIFIED",
                    confidence=min(1.0, max(0.0, max((item.get("relevance", 0.0) for item in context_evidence), default=0.0))),
                    faithfulness=None, sources=response_sources, unsupported_claims=[],
                    flags=policy.get("flags", []), trace=trace, started=started,
                )

            original_confidence = 0.0
            final_generation = None
            final_verification = None
            final_context = []
            final_trust = {"evidence": [], "sources": [], "fusion": [], "flags": [], "trace": {}}
            pass_policy = dict(policy)

            for attempt in range(2):
                retrieval_started = time.perf_counter()
                retrieval = self._retrieve_with_fallback(store, query, query_vector, pass_policy, mode, live_check)
                trace["retrieval_calls"] += 1
                trace["live_enabled"] = bool(retrieval["trace"].get("live_documents", 0))
                trace["live_documents"] = retrieval["trace"].get("live_documents", 0)
                trace["retrieval_fallback_stages"] = retrieval["trace"].get("fallback_stages", ["offline"])
                trace["layers_ms"]["L3_retrieval"] = trace["layers_ms"].get("L3_retrieval", 0.0) + (time.perf_counter() - retrieval_started) * 1000

                trust_started = time.perf_counter()
                trust_result = TrustPipeline(self.config_path, store).score(
                    retrieval_result=retrieval, policy=pass_policy, as_of=as_of
                )
                trace["trust_calls"] += 1
                trace["layers_ms"]["L4_trust"] = trace["layers_ms"].get("L4_trust", 0.0) + (time.perf_counter() - trust_started) * 1000

                if (
                    not trust_result.get("fusion")
                    and trace.get("live_documents")
                    and policy.get("category") == "visa_policy"
                    and policy.get("entities", {}).get("destination") == "United Kingdom"
                ):
                    source = next((item for item in final_context if item.get("live")), None)
                    if source is None:
                        source = next(
                            (item for item in retrieval.get("evidence", []) if item.get("live")), None
                        )
                    if source is None:
                        source = type("LiveSourceView", (), {
                            "title": "GOV.UK Check if you need a UK visa",
                            "url": "https://www.gov.uk/check-uk-visa",
                            "source_type": "government",
                            "fetched_at": None,
                            "content_date": None,
                            "nationality_scope": "ALL",
                        })()
                    citation = "E1"
                    nationality = policy.get("entities", {}).get("nationality", "India")
                    nationality_label = {"India": "Indian", "United Kingdom": "British", "United States": "American"}.get(nationality, nationality)
                    fallback_answer = (
                        f"Assuming an {nationality_label} passport, I couldn’t determine the exact visa outcome "
                        f"for your travel purpose from the retrieved page. Check the official UK visa checker [{citation}]."
                    )
                    source_row = {
                        "evidence_id": citation,
                        "title": source.get("title", "GOV.UK Check if you need a UK visa") if source else "GOV.UK Check if you need a UK visa",
                        "url": source.get("url", "https://www.gov.uk/check-uk-visa") if source else "https://www.gov.uk/check-uk-visa",
                        "source_type": source.get("source_type", "government") if source else "government",
                        "trust": 0.0,
                        "trust_scored": False,
                        "cited": True,
                        "live": bool(source and source.get("live")),
                        "search_fallback": bool(source and source.get("search_fallback")),
                        "fetched_at": source.get("fetched_at") if source else None,
                        "nationality_scope": source.get("nationality_scope", "UNKNOWN") if source else "UNKNOWN",
                    }
                    trace["live_enabled"] = True
                    trace["live_fallback"] = trace.get("retrieval_fallback_stages", [])
                    trace["live_fallback"] = "official_interactive_visa_checker"
                    flags = list(dict.fromkeys(policy.get("flags", []) + retrieval.get("flags", []) + ["verification_required"]))
                    return self._finish(
                        query,
                        policy,
                        answer=fallback_answer,
                        verification_status="VERIFICATION_REQUIRED",
                        confidence=0.0,
                        faithfulness=0.0,
                        freshness={"most_recent_source": source.content_date, "age_days": None},
                        agreement="Low",
                        conflicts=[],
                        sources=[source_row],
                        unsupported_claims=[],
                        flags=flags,
                        trace=trace,
                        started=started,
                        verification_required=True,
                    )

                generation_payload, context_evidence = build_generation_payload(query, pass_policy, trust_result)
                generation_started = time.perf_counter()
                generated = generate_answer(active_llm, generation_payload)
                trace["generations"] += 1
                trace["layers_ms"]["L5_generation"] = trace["layers_ms"].get("L5_generation", 0.0) + (time.perf_counter() - generation_started) * 1000

                verify_started = time.perf_counter()
                verification = verify_answer(generated.text, context_evidence, trust_result["fusion"])
                trace["verifications"] += 1
                trace["layers_ms"]["L5_verification"] = trace["layers_ms"].get("L5_verification", 0.0) + (time.perf_counter() - verify_started) * 1000

                final_generation, final_verification = generated, verification
                final_context, final_trust = context_evidence, trust_result
                main_slot = self._main_slot(query, trust_result["fusion"])
                original_confidence = next(
                    (float(item.get("confidence", 0.0)) for item in trust_result["fusion"] if item["slot"] == main_slot),
                    max((item.get("confidence", 0.0) for item in trust_result["fusion"]), default=0.0),
                )
                faithfulness = float(verification["faithfulness"])
                if faithfulness >= 0.70:
                    break
                if attempt == 0:
                    pass_policy = self._widen_policy(policy)
                    trace["retried"] = True

            faithfulness = float(final_verification["faithfulness"])
            answer_text = final_generation.text
            flags = list(dict.fromkeys(policy.get("flags", []) + final_trust.get("flags", []) + final_verification.get("flags", [])))
            if trace.get("retried"):
                flags.append("retried")
            if faithfulness >= 0.90:
                status = "VERIFIED"
                confidence = original_confidence
            elif faithfulness >= 0.70:
                status = "PARTIAL"
                confidence = original_confidence * 0.85
            else:
                status = "VERIFICATION_REQUIRED"
                confidence = min(original_confidence, 0.40)
                flags.append("verification_required")
                primary_sources = [source for source in final_trust.get("sources", []) if source.get("source_type") in {"government", "embassy", "intl_body"}]
                if primary_sources:
                    refs = " ".join(f"[{context.get('citation_id')}]" for context in final_context if context.get("is_primary"))
                    answer_text = f"The saved evidence could not fully verify this advice. Please confirm with the official sources. {refs}".strip()

            freshness, agreement, conflicts = self._response_signals(final_trust, final_context, answer_text, as_of)
            response_sources = self._response_sources(final_context, answer_text)
            trace["faithfulness_verifier"] = final_verification.get("verifier")
            trace["claim_checks"] = final_verification.get("claims", [])
            trace["context_evidence"] = [
                {"slot": item.get("slot"), "value": item.get("value_norm"), "evidence_id": item.get("citation_id"), "trust": item.get("Trust"), "url": item.get("url"), "document_id": item.get("document_id")}
                for item in final_context
            ]
            trace["resolved_slots"] = final_trust.get("fusion", [])
            trace["policy"] = policy
            trace["trust"] = final_trust.get("trace", {})
            return self._finish(
                query,
                policy,
                answer=answer_text,
                verification_status=status,
                confidence=confidence,
                faithfulness=faithfulness,
                freshness=freshness,
                agreement=agreement,
                conflicts=conflicts,
                sources=response_sources,
                unsupported_claims=[claim for claim in final_verification["claims"] if claim["label"] != "SUPPORTED"],
                flags=flags,
                trace=trace,
                started=started,
                verification_required=status == "VERIFICATION_REQUIRED",
            )

    def _fetch_live(
        self, policy: dict[str, Any], mode: str, live_check: bool
    ) -> tuple[list[Any], list[str]]:
        if mode == "benchmark" or not live_check or not policy.get("live_requested"):
            return [], []
        fetcher = LiveFetcher(self.config_path)
        return fetcher.fetch(policy, mode=mode, live_requested=True)

    def _retrieve_with_fallback(
        self,
        store: KnowledgeStore,
        query: str,
        query_vector: Any,
        policy: dict[str, Any],
        mode: str,
        live_check: bool,
    ) -> dict[str, Any]:
        retriever = OnlineRetriever(store)
        offline_result = retriever.retrieve(policy=policy, query_vec=query_vector)
        required_slot = self._requested_slot(query, policy)
        fallback_stages = ["offline"]
        main_slot_available = self._has_slot(offline_result, required_slot)
        combined = offline_result
        live_documents: list[Any] = []

        if not main_slot_available and mode != "benchmark" and live_check and policy.get("live_requested"):
            live_documents, live_flags = self._fetch_live(policy, mode, live_check)
            fallback_stages.append("live_api")
            if live_documents:
                live_result = self._retrieve_documents_only(
                    retriever, query_vector, policy, live_documents
                )
                combined = self._merge_retrieval_results(offline_result, live_result)
            else:
                combined["flags"] = list(dict.fromkeys(combined.get("flags", []) + live_flags))
            main_slot_available = self._has_slot(combined, required_slot)

        google_documents = []
        if not main_slot_available and fallback_stages[-1:] == ["live_api"] and mode != "benchmark":
            google_config = self.config.get("search_fallback", {}).get("google", {})
            if google_config.get("enabled", False):
                google_documents, google_flags = GoogleSearchFallback().search(
                    query=query,
                    destination=str(policy.get("entities", {}).get("destination", "")),
                    nationality=str(policy.get("entities", {}).get("nationality", "")),
                    category=str(policy.get("category", "")),
                    limit=int(google_config.get("results", 5)),
                    timeout_s=float(google_config.get("timeout_s", 5.0)),
                )
                fallback_stages.append("google_search")
                if google_documents:
                    google_result = self._retrieve_documents_only(
                        retriever, query_vector, policy, google_documents
                    )
                    combined = self._merge_retrieval_results(combined, google_result)
                    combined["flags"] = list(dict.fromkeys(combined["flags"] + ["google_search_fallback"]))
                combined["flags"] = list(dict.fromkeys(combined.get("flags", []) + google_flags))

        combined.setdefault("trace", {})["fallback_stages"] = fallback_stages
        combined["trace"]["live_documents"] = len(live_documents)
        combined["trace"]["google_documents"] = len(google_documents)
        combined["trace"]["requested_slot"] = required_slot
        return combined

    @staticmethod
    def _retrieve_documents_only(
        retriever: OnlineRetriever,
        query_vector: Any,
        policy: dict[str, Any],
        documents: list[Any],
    ) -> dict[str, Any]:
        source_policy = dict(policy)
        retrieval_policy = dict(policy["retrieval_policy"])
        retrieval_policy["collections"] = []
        source_policy["retrieval_policy"] = retrieval_policy
        return retriever.retrieve(policy=source_policy, query_vec=query_vector, live_documents=documents)

    @staticmethod
    def _has_slot(retrieval: dict[str, Any], slot: str | None) -> bool:
        if slot is None:
            return bool(retrieval.get("evidence"))
        return any(item.get("slot") == slot for item in retrieval.get("evidence", []))

    @staticmethod
    def _merge_retrieval_results(*results: dict[str, Any]) -> dict[str, Any]:
        evidence = []
        seen = set()
        flags = []
        trace: dict[str, Any] = {}
        for result in results:
            for item in result.get("evidence", []):
                key = (item.get("url"), item.get("slot"), str(item.get("value_norm")))
                if key not in seen:
                    seen.add(key)
                    evidence.append(item)
            flags.extend(result.get("flags", []))
            trace.update(result.get("trace", {}))
        return {"evidence": evidence, "flags": list(dict.fromkeys(flags)), "trace": trace}

    @staticmethod
    def _requested_slot(query: str, policy: dict[str, Any]) -> str | None:
        lowered = query.casefold()
        available = set(policy.get("target_slots", []))
        category = policy.get("category")
        if category == "visa_policy":
            if any(term in lowered for term in ("how long", "duration", "stay", "days")):
                return "max_stay_days"
            if any(term in lowered for term in ("evisa", "e-visa", "visa type")):
                return "visa_type"
            if "visa" in lowered and available:
                return "visa_required"
        elif category == "customs_regulation":
            if any(term in lowered for term in ("duty-free", "duty free", "allowance limit")):
                return "duty_free_limit"
            if any(term in lowered for term in ("declare", "declaration")):
                return "declaration_required"
            if any(term in lowered for term in ("carry", "bring", "allowed", "prohibited")):
                return "item_allowed"
        elif category == "vaccination_requirement":
            if "certificate" in lowered or "proof" in lowered:
                return "certificate_required"
            if any(term in lowered for term in ("which vaccine", "what vaccine", "vaccine name")):
                return "vaccine_name"
            if "vaccine" in lowered or "vaccin" in lowered:
                return "vaccine_required"
        return None

    def _baseline_retrieve(self, store: KnowledgeStore, query_vector: Any, policy: dict[str, Any]) -> list[dict[str, Any]]:
        results = []
        top_k = int(self.config.get("retrieval", {}).get("primary_top_k", 8))
        for collection in ("primary", "secondary"):
            for hit in store.search(query_vector, collection, top_k):
                chunk, document = store.get_chunk(hit["chunk_id"]), store.get_document(hit["document_id"])
                if chunk is None or document is None:
                    continue
                if document.get("country", "").casefold() != str(policy["entities"].get("destination", "")).casefold():
                    continue
                claims = extract_claims(
                    chunk={**chunk, "score": hit["score"]}, document=document,
                    slots=policy["target_slots"], destination=policy["entities"]["destination"],
                    item=policy["entities"].get("item"),
                )
                results.extend(claims)
        return results

    @staticmethod
    def _baseline_resolutions(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
        winners = {}
        for item in sorted(evidence, key=lambda value: value.get("relevance", 0.0), reverse=True):
            winners.setdefault(item.get("slot"), {"slot": item.get("slot"), "value": item.get("value_norm")})
        return list(winners.values())

    @staticmethod
    def _widen_policy(policy: dict[str, Any]) -> dict[str, Any]:
        widened = dict(policy)
        retrieval = dict(policy["retrieval_policy"])
        top_k = dict(retrieval["top_k"])
        top_k["primary"] = max(1, int(top_k.get("primary", 8)) * 2)
        top_k["secondary"] = max(1, int(top_k.get("secondary", 5)) * 2)
        retrieval["top_k"] = top_k
        retrieval["collections"] = ["primary", "secondary"]
        retrieval["tau_rel"] = max(0.0, float(retrieval.get("tau_rel", 0.45)) - 0.05)
        widened["retrieval_policy"] = retrieval
        return widened

    @staticmethod
    def _response_sources(evidence: list[dict[str, Any]], answer: str) -> list[dict[str, Any]]:
        sources_by_document = {}
        for item in evidence:
            document_id = str(item.get("document_id", item.get("citation_id")))
            source = {
                "evidence_id": item.get("citation_id"),
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "source_type": item.get("source_type", "unknown"),
                "trust": float(item.get("Trust", item.get("relevance", 0.0))),
                "cited": f"[{item.get('citation_id')}]" in answer,
                "live": bool(item.get("live", False)),
                "origin": "Google search result" if item.get("search_fallback") else "Live official source" if item.get("live") else "Offline index",
                "search_fallback": bool(item.get("search_fallback", False)),
                "fetched_at": item.get("fetched_at"),
                "nationality_scope": item.get("nationality_scope", "UNKNOWN"),
            }
            previous = sources_by_document.get(document_id)
            if previous is None:
                sources_by_document[document_id] = source
            else:
                previous["cited"] = previous["cited"] or source["cited"]
                if source["trust"] > previous["trust"]:
                    source["cited"] = previous["cited"]
                    sources_by_document[document_id] = source
        return sorted(sources_by_document.values(), key=lambda source: source["trust"], reverse=True)

    @staticmethod
    def _main_slot(query: str, fusion: list[dict[str, Any]]) -> str | None:
        lowered = query.lower()
        preferred = (
            ("max_stay_days", ("how long", "duration", "stay", "days")),
            ("visa_required", ("need a visa", "need visa", "visa required")),
            ("vaccine_required", ("vaccination", "vaccine", "yellow fever")),
            ("item_allowed", ("carry", "bring", "allowed", "prohibited")),
            ("declaration_required", ("declare", "declaration")),
        )
        available = {item["slot"] for item in fusion}
        for slot, terms in preferred:
            if slot in available and any(term in lowered for term in terms):
                return slot
        return fusion[0]["slot"] if fusion else None

    @staticmethod
    def _response_signals(trust_result: dict[str, Any], evidence: list[dict[str, Any]], answer: str, as_of: str | None) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
        cited_ids = {item.get("citation_id") for item in evidence if f"[{item.get('citation_id')}]" in answer}
        cited = [item for item in evidence if item.get("citation_id") in cited_ids]
        primary_dates = [item.get("freshness_ref_date") for item in cited if item.get("is_primary") and item.get("freshness_ref_date")]
        latest = max(primary_dates) if primary_dates else None
        age = (date.fromisoformat(as_of) - date.fromisoformat(latest)).days if latest and as_of else None
        fusion = trust_result.get("fusion", [])
        main_share = max((item.get("share", 0.0) for item in fusion), default=0.0)
        agreement = "High" if main_share >= 0.85 else "Medium" if main_share >= 0.60 else "Low"
        conflicts = [
            {"slot": item["slot"], "value": item["conflicting_view"], "chosen": item["value"], "support": item["support_by_value"]}
            for item in fusion if item.get("conflicting_view")
        ]
        return {"most_recent_source": latest, "age_days": age}, agreement, conflicts

    def _finish(
        self,
        query: str,
        policy: dict[str, Any],
        *,
        answer: str,
        verification_status: str,
        confidence: float,
        faithfulness: float | None,
        sources: list[dict[str, Any]],
        unsupported_claims: list[dict[str, Any]],
        flags: list[str],
        trace: dict[str, Any],
        started: float,
        freshness: dict[str, Any] | None = None,
        agreement: str = "Low",
        conflicts: list[dict[str, Any]] | None = None,
        verification_required: bool = False,
    ) -> dict[str, Any]:
        latency_ms = (time.perf_counter() - started) * 1000
        trace["total_ms"] = latency_ms
        response = {
            "query": query,
            "system": trace.get("system"),
            "system_name": trace.get("system_name"),
            "answer": answer,
            "confidence": round(float(confidence), 6),
            "faithfulness": faithfulness,
            "verification_status": verification_status,
            "verification_required": verification_required,
            "category": policy.get("category"),
            "volatility": policy.get("volatility_level"),
            "evidence_strategy": policy.get("evidence_strategy"),
            "freshness": freshness or {"most_recent_source": None, "age_days": None},
            "agreement": agreement,
            "conflicts": conflicts or [],
            "sources": sources,
            "unsupported_claims": unsupported_claims,
            "flags": list(dict.fromkeys(flags)),
            "latency_ms": latency_ms,
            "trace": trace,
        }
        log_path = Path(self.config.get("paths", {}).get("response_log", "logs/responses.jsonl"))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(response, ensure_ascii=True) + "\n")
        return response