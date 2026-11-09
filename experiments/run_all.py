"""Run benchmark systems with shared data/config and an offline socket guard."""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from experiments.benchmark import check_nationality_coverage, validate_benchmark
from experiments.metrics import answer_accuracy, expected_calibration_error, hallucination_rate
from experiments.stats import bootstrap_ci, mcnemar
from trustrag.pipeline import TrustRAGPipeline


def smoke_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "query_id": "smoke_vietnam_visa",
                "query": "How long can Indian travellers stay in Vietnam?",
                "nationality": "India",
                "destination": "Vietnam",
                "category": "visa_policy",
                "volatility": "High",
                "gold_slot": "max_stay_days",
                "gold_value": "30",
                "gold_answer": "Up to 30 days under the eligible visa policy.",
                "gold_source_url": "https://example.gov.vn/visa/vietnam/indian-travellers",
                "as_of_date": "2026-11-09",
                "difficulty": "hard",
                "has_conflict": "1",
                "conflict_type": "wrong_secondary",
                "paraphrase_group": "smoke_vietnam_visa",
                "split": "test",
            }
        ]
    )


@contextmanager
def benchmark_network_guard() -> Iterator[None]:
    original_connect = socket.socket.connect

    def guarded_connect(sock: socket.socket, address: Any) -> Any:
        host = address[0] if isinstance(address, tuple) and address else ""
        if host not in {"localhost", "127.0.0.1", "::1"}:
            raise RuntimeError(f"Network access is forbidden in benchmark mode: {host}")
        return original_connect(sock, address)

    socket.socket.connect = guarded_connect
    try:
        yield
    finally:
        socket.socket.connect = original_connect


def _equal(left: Any, right: Any) -> bool:
    try:
        return float(left) == float(right)
    except (TypeError, ValueError):
        return str(left).strip().casefold() == str(right).strip().casefold()


def _run_query(row: dict[str, Any], system: str, config_path: str) -> dict[str, Any]:
    response = TrustRAGPipeline(config_path).answer(
        str(row["query"]), system=system, llm="mock", as_of=str(row["as_of_date"]), mode="benchmark"
    )
    trace = response["trace"]
    resolved = next(
        (item.get("value") for item in trace.get("resolved_slots", []) if item.get("slot") == row["gold_slot"]),
        None,
    )
    retrieved = [item for item in trace.get("retrieved_evidence", []) if item.get("slot") == row["gold_slot"]]
    selected = [item for item in trace.get("context_evidence", []) if item.get("slot") == row["gold_slot"]]
    relevant = {item.get("document_id") for item in retrieved if _equal(item.get("value"), row["gold_value"])}
    selected_ids = {item.get("document_id") for item in selected}
    relevant_selected = len(selected_ids & relevant)
    precision = relevant_selected / len(selected_ids) if selected_ids else 0.0
    recall = relevant_selected / len(relevant) if relevant else 0.0
    conflict_evidence = trace.get("trust", {}).get("evidence", [])
    predicted_conflict = any(
        item.get("slot") == row["gold_slot"] and float(item.get("severity", 0.0)) > 0
        for item in conflict_evidence
    )
    claims = trace.get("claim_checks", [])
    return {
        "query_id": row["query_id"],
        "system": system,
        "gold_value": row["gold_value"],
        "predicted_value": resolved,
        "correct": bool(_equal(resolved, row["gold_value"])) and response["verification_status"] != "VERIFICATION_REQUIRED",
        "verification_status": response["verification_status"],
        "confidence": response["confidence"],
        "faithfulness": response["faithfulness"],
        "hallucination_rate": hallucination_rate(claims),
        "latency_ms": response["latency_ms"],
        "retry": bool(trace.get("retried", False)),
        "evidence_precision": precision,
        "evidence_recall": recall,
        "has_conflict": str(row["has_conflict"]) == "1",
        "predicted_conflict": predicted_conflict,
        "trust_labels": [item.get("value_norm") == row["gold_value"] for item in conflict_evidence if item.get("slot") == row["gold_slot"]],
        "trust_scores": [item.get("Trust") for item in conflict_evidence if item.get("slot") == row["gold_slot"]],
    }


def _summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for system, frame in predictions.groupby("system"):
        records = frame.to_dict("records")
        accuracy = answer_accuracy(records)
        faithfulness = frame["faithfulness"].dropna().astype(float).tolist()
        claims_hallucination = frame["hallucination_rate"].astype(float).tolist()
        confidences = frame["confidence"].astype(float).tolist()
        correct = frame["correct"].astype(bool).tolist()
        confidence_interval = bootstrap_ci([float(value) for value in correct])
        try:
            trust_labels = np.concatenate(frame["trust_labels"].map(lambda value: np.asarray(value, dtype=bool)).values)
            trust_scores = np.concatenate(frame["trust_scores"].map(lambda value: np.asarray(value, dtype=float)).values)
            auc = roc_auc_score(trust_labels, trust_scores) if len(np.unique(trust_labels)) > 1 else float("nan")
        except (ValueError, TypeError):
            auc = float("nan")
        rows.append(
            {
                "system": system,
                **accuracy,
                "accuracy_ci_low": confidence_interval[0],
                "accuracy_ci_high": confidence_interval[1],
                "hallucination_rate": float(np.mean(claims_hallucination)) if claims_hallucination else 0.0,
                "faithfulness": float(np.mean(faithfulness)) if faithfulness else float("nan"),
                "latency_p50_ms": float(frame["latency_ms"].quantile(0.50)),
                "latency_p95_ms": float(frame["latency_ms"].quantile(0.95)),
                "retry_rate": float(frame["retry"].mean()),
                "ece": expected_calibration_error(confidences, correct),
                "trust_auroc": float(auc),
            }
        )
    return pd.DataFrame(rows)


def _plot_results(metrics: pd.DataFrame, output: Path) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for metric, title, filename in (
        ("accuracy", "Answer accuracy", "accuracy.png"),
        ("hallucination_rate", "Hallucination rate", "hallucination_rate.png"),
    ):
        fig, axis = plt.subplots(figsize=(7, 4))
        axis.bar(metrics["system"], metrics[metric], color="#26736b")
        axis.set_ylim(0, 1)
        axis.set_ylabel(metric.replace("_", " ").title())
        axis.set_title(title)
        fig.tight_layout()
        fig.savefig(figures / filename, dpi=150)
        plt.close(fig)


def run_experiments(*, smoke: bool = False, benchmark_path: str | None = None, config_path: str = "trustrag/config.yaml") -> tuple[pd.DataFrame, pd.DataFrame]:
    experiment_config_path = Path("experiments/config.yaml")
    experiment_config = yaml.safe_load(experiment_config_path.read_text(encoding="utf-8")) or {}
    if smoke:
        benchmark = smoke_frame()
    else:
        benchmark = validate_benchmark(
            benchmark_path or experiment_config["benchmark_path"],
            experiment_config.get("classifier_data"),
        )
        check_nationality_coverage(benchmark)
    benchmark = benchmark[benchmark["split"] == "test"]
    if benchmark.empty:
        raise ValueError("Benchmark has no test split rows")
    index_paths = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))["paths"]
    if not Path(index_paths.get("primary_faiss_path", "data/store/primary.faiss")).exists():
        raise FileNotFoundError("Offline index is missing; run `make index` before experiments")

    predictions = []
    with benchmark_network_guard():
        for row in benchmark.to_dict("records"):
            for system in experiment_config.get("systems", ["S0", "S1", "S2", "S3"]):
                try:
                    predictions.append(_run_query(row, system, config_path))
                except FileNotFoundError as error:
                    if system != "S3":
                        raise
                    predictions.append({"query_id": row["query_id"], "system": system, "skipped": str(error)})

    prediction_frame = pd.DataFrame(predictions)
    completed = prediction_frame[prediction_frame.get("correct").notna()] if "correct" in prediction_frame else prediction_frame.iloc[0:0]
    metrics_frame = _summarize(completed) if not completed.empty else pd.DataFrame()
    output = Path(experiment_config.get("results_dir", "results"))
    output.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_csv(output / "predictions.csv", index=False)
    metrics_frame.to_csv(output / "main_results.csv", index=False)
    (output / "run_manifest.json").write_text(
        json.dumps(
            {
                "git_hash": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
                "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
                "model": yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))["embedding"]["model_name"],
                "llm": "mock",
                "seed": experiment_config.get("seed", 42),
                "as_of_date": experiment_config.get("as_of_date", "2026-11-09"),
                "benchmark_mode": True,
                "smoke": smoke,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if not metrics_frame.empty:
        _plot_results(metrics_frame, output)
    return prediction_frame, metrics_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TrustRAG benchmark systems")
    parser.add_argument("--smoke", action="store_true", help="Run the one-query synthetic smoke benchmark")
    parser.add_argument("--benchmark", default=None)
    parser.add_argument("--config", default="trustrag/config.yaml")
    args = parser.parse_args()
    predictions, metrics = run_experiments(smoke=args.smoke, benchmark_path=args.benchmark, config_path=args.config)
    print(metrics.to_string(index=False) if not metrics.empty else "No systems produced metrics")
    print(f"Wrote {len(predictions)} prediction row(s) to results/predictions.csv")


if __name__ == "__main__":
    main()