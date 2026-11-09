"""Command-line interface for the TrustRAG phase 1 offline indexer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trustrag.offline.embed_index import OfflineFAISSIndex, encode_query_vector
from trustrag.offline.knowledge_store import KnowledgeStore
from trustrag.online.live.fetcher import LiveFetcher
from trustrag.online.retrieval.search import OnlineRetriever
from trustrag.online.query.intent import QueryUnderstanding
from trustrag.online.trust.pipeline import TrustPipeline


def _print_step(label: str) -> None:
    print(f"\n[STEP] {label}")


def _print_pass(label: str) -> None:
    print(f"[PASS] {label}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TrustRAG offline indexer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Build the FAISS index from the corpus")
    index_parser.add_argument("--corpus", type=str, default="data/corpus", help="Corpus directory")
    index_parser.add_argument("--config", type=str, default="trustrag/config.yaml", help="Config YAML path")

    search_parser = subparsers.add_parser("search", help="Search the built index")
    search_parser.add_argument("query", type=str)
    search_parser.add_argument("--top-k", type=int, default=5)
    search_parser.add_argument("--config", type=str, default="trustrag/config.yaml", help="Config YAML path")

    analyze_parser = subparsers.add_parser("analyze", help="Analyze query intent and policy")
    analyze_parser.add_argument("query", type=str)
    analyze_parser.add_argument("--config", type=str, default="trustrag/config.yaml", help="Config YAML path")

    policy_parser = subparsers.add_parser("policy", help="Build the Phase 2 policy decision")
    policy_parser.add_argument("query", type=str)
    policy_parser.add_argument("--config", type=str, default="trustrag/config.yaml", help="Config YAML path")
    policy_parser.add_argument("--as-of", type=str, default=None)
    policy_parser.add_argument("--live-check", action="store_true")

    trust_parser = subparsers.add_parser("trust", help="Retrieve and score evidence for a query")
    trust_parser.add_argument("query", type=str)
    trust_parser.add_argument("--config", type=str, default="trustrag/config.yaml", help="Config YAML path")
    trust_parser.add_argument("--as-of", type=str, default=None)
    trust_parser.add_argument("--live-check", action="store_true")

    ask_parser = subparsers.add_parser("ask", help="Answer a travel advisory question")
    ask_parser.add_argument("query", type=str)
    ask_parser.add_argument("--system", choices=("S0", "S1", "S2", "S3"), default="S2")
    ask_parser.add_argument("--llm", choices=("mock", "ollama", "openai", "anthropic"), default="ollama")
    ask_parser.add_argument("--as-of", type=str, default=None)
    ask_parser.add_argument("--trace", action="store_true")
    ask_parser.add_argument("--live-check", action="store_true")
    ask_parser.add_argument("--config", type=str, default="trustrag/config.yaml", help="Config YAML path")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "index":
        _print_step("Phase 1: building offline corpus index")
        indexer = OfflineFAISSIndex.from_config(Path(args.config))
        result = indexer.build_from_directory(Path(args.corpus))
        print(json.dumps(result, indent=2))
        _print_pass("Index built successfully")
        return

    if args.command == "search":
        _print_step(f"Phase 1: searching collections for query: '{args.query}'")
        query_vector = encode_query_vector(args.query, args.config)
        with KnowledgeStore.from_config(args.config) as store:
            hits = {
                collection: store.search(query_vector, collection, args.top_k)
                for collection in ("primary", "secondary")
            }
        print(json.dumps(hits, indent=2))
        total = sum(len(collection_hits) for collection_hits in hits.values())
        _print_pass(f"Search returned {total} result(s) across two collections")
        return

    if args.command == "analyze":
        _print_step(f"Phase 2: analyzing query: '{args.query}'")
        analyzer = QueryUnderstanding.from_config(Path(args.config))
        summary = analyzer.summary(args.query)
        print(json.dumps(summary, indent=2))
        _print_pass("Query analysis complete")
        return

    if args.command == "policy":
        _print_step(f"Phase 2: building policy for query: '{args.query}'")
        summary = QueryUnderstanding.from_config(Path(args.config)).summary(
            args.query, as_of=args.as_of, live_check=args.live_check
        )
        print(json.dumps(summary, indent=2))
        _print_pass(f"Policy action: {summary['action']}")
        return

    if args.command == "trust":
        _print_step(f"Phase 3: retrieving and scoring query: '{args.query}'")
        policy = QueryUnderstanding.from_config(Path(args.config)).summary(
            args.query, as_of=args.as_of, live_check=args.live_check
        )
        if policy["action"] != "RETRIEVE":
            print(json.dumps({"policy": policy, "trust_result": None}, indent=2))
            _print_pass(f"No retrieval performed: {policy['action']}")
            return
        query_vector = encode_query_vector(args.query, args.config)
        live_documents, live_flags = LiveFetcher(args.config).fetch(
            policy, mode="demo", live_requested=args.live_check
        )
        with KnowledgeStore.from_config(args.config) as store:
            retrieval = OnlineRetriever(store).retrieve(
                policy=policy, query_vec=query_vector, live_documents=live_documents
            )
            retrieval["flags"] = list(dict.fromkeys(retrieval["flags"] + live_flags))
            result = TrustPipeline(args.config, store).score(
                retrieval_result=retrieval,
                policy=policy,
                as_of=args.as_of,
            )
        print(json.dumps({"policy": policy, "trust_result": result}, indent=2))
        _print_pass(f"Trust scored for {len(result['evidence'])} evidence item(s)")
        return

    if args.command == "ask":
        from trustrag.pipeline import TrustRAGPipeline

        response = TrustRAGPipeline(args.config).answer(
            args.query,
            system=args.system,
            llm=args.llm,
            as_of=args.as_of,
            live_check=args.live_check,
        )
        print(f"System: {response['system_name']} ({response['system']})")
        print(response["answer"])
        print(f"\nStatus: {response['verification_status']} | Confidence: {response['confidence']:.1%}")
        if response["sources"]:
            print("\nSources:")
            for source in response["sources"]:
                print(f"- {source['source_type']} | trust {source['trust']:.2f} | {source['title']} | {source['url']}")
        if args.trace:
            print("\nTrace:")
            print(json.dumps(response["trace"], indent=2))
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
