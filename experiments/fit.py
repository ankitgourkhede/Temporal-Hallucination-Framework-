"""Fit train-split source reputation first, then collect evidence features for S3."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from experiments.benchmark import validate_benchmark
from experiments.learn_weights import fit_all
from trustrag.offline.reputation import SourceReputationStore
from trustrag.pipeline import TrustRAGPipeline


def _equal(left: Any, right: Any) -> bool:
    try:
        return float(left) == float(right)
    except (TypeError, ValueError):
        return str(left).strip().casefold() == str(right).strip().casefold()


def fit_from_benchmark(benchmark_path: str, config_path: str = "trustrag/config.yaml") -> dict[str, Any]:
    benchmark = validate_benchmark(benchmark_path)
    training = benchmark[benchmark["split"] == "train"]
    if training.empty:
        raise ValueError("Benchmark train split is empty")
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    reputation = SourceReputationStore(config["paths"]["metadata_db"], alpha=2.0, beta=2.0)
    reputation.reset_train()
    pipeline = TrustRAGPipeline(config_path)
    train_rows = training.to_dict("records")
    for query in train_rows:
        response = pipeline.answer(query["query"], system="S2", llm="mock", as_of=query["as_of_date"], mode="benchmark")
        evidence = response["trace"].get("trust", {}).get("evidence", [])
        for item in evidence:
            if item.get("slot") != query["gold_slot"]:
                continue
            correct = _equal(item.get("value_norm"), query["gold_value"])
            reputation.update(str(item.get("source_domain", "unknown")), correct, split="train")

    rows = []
    for query in train_rows:
        response = pipeline.answer(query["query"], system="S2", llm="mock", as_of=query["as_of_date"], mode="benchmark")
        evidence = response["trace"].get("trust", {}).get("evidence", [])
        for item in evidence:
            if item.get("slot") != query["gold_slot"]:
                continue
            correct = _equal(item.get("value_norm"), query["gold_value"])
            rows.append(
                {
                    "query_id": query["query_id"],
                    "category": query["category"],
                    "split": "train",
                    "A": item["A"], "F": item["F"], "QE": item["QE"], "SR": item["SR"],
                    "CP": item["CP"], "G": item["G"], "correct": int(correct),
                }
            )
    if not rows:
        raise ValueError("No evidence on benchmark gold slots was produced for weight fitting")
    feature_path = Path("data/train/trust_features.csv")
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(feature_path, index=False)
    result = fit_all(feature_path)
    return {"training_examples": len(rows), "categories": list(result), "features": str(feature_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit SR and learned trust weights from benchmark train split")
    parser.add_argument("--benchmark", default="data/benchmark/benchmark.csv")
    parser.add_argument("--config", default="trustrag/config.yaml")
    args = parser.parse_args()
    print(yaml.safe_dump(fit_from_benchmark(args.benchmark, args.config), sort_keys=False))


if __name__ == "__main__":
    main()