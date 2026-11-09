"""Train calibrated intent/category classifiers with paraphrase-grouped evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sentence_transformers import SentenceTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold


def _expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, predictions: np.ndarray) -> float:
    confidences = probabilities.max(axis=1)
    correct = predictions == labels
    ece = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        mask = (confidences >= lower) & (confidences < upper if upper < 1.0 else confidences <= 1.0)
        if mask.any():
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidences[mask].mean()))
    return ece


def _new_classifier() -> CalibratedClassifierCV:
    estimator = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=42)
    return CalibratedClassifierCV(estimator=estimator, method="sigmoid", cv=5)


def _train_task(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    task: str,
    output_dir: Path,
) -> dict[str, Any]:
    unique_groups = np.unique(groups)
    if len(unique_groups) < 5:
        raise ValueError(f"At least five distinct paraphrase groups are required to train {task}")
    splitter = GroupKFold(n_splits=5)
    predictions = np.empty(labels.shape, dtype=object)
    probabilities = np.zeros((len(labels), len(np.unique(labels))), dtype=float)
    for train_indices, test_indices in splitter.split(features, labels, groups):
        model = _new_classifier()
        model.fit(features[train_indices], labels[train_indices])
        fold_predictions = model.predict(features[test_indices])
        fold_probabilities = model.predict_proba(features[test_indices])
        predictions[test_indices] = fold_predictions
        for row_idx, label in enumerate(model.classes_):
            class_index = list(np.unique(labels)).index(label)
            probabilities[test_indices, class_index] = fold_probabilities[:, row_idx]

    final_model = _new_classifier()
    final_model.fit(features, labels)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, output_dir / f"{task}.joblib")
    classes = np.unique(labels)
    return {
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=classes).tolist(),
        "labels": classes.tolist(),
        "ece": _expected_calibration_error(probabilities, labels, predictions),
    }


def train_classifiers(csv_path: str | Path, config_path: str | Path) -> dict[str, Any]:
    csv_path, config_path = Path(csv_path), Path(config_path)
    frame = pd.read_csv(csv_path).fillna("")
    required = {"query", "intent", "category", "paraphrase_of"}
    if not required <= set(frame.columns):
        raise ValueError(f"Training CSV must contain columns: {', '.join(sorted(required))}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    embedding = config["embedding"]
    encoder = SentenceTransformer(embedding["model_name"], device="cpu")
    texts = [str(embedding.get("query_prefix", "")) + query for query in frame["query"]]
    features = np.asarray(
        encoder.encode(texts, batch_size=int(embedding.get("batch_size", 32)), normalize_embeddings=True),
        dtype="float32",
    )
    classifier_config = config.get("query_classifier", {})
    output_dir = Path(classifier_config.get("model_dir", "models/classifiers"))
    if not output_dir.is_absolute():
        output_dir = config_path.parent.parent / output_dir
    groups = frame["paraphrase_of"].astype(str).to_numpy()
    if np.any(groups == ""):
        raise ValueError("Every training row must have a paraphrase_of group to prevent leakage")

    results = {}
    results["intent"] = _train_task(features, frame["intent"].astype(str).to_numpy(), groups, "intent", output_dir)
    in_scope = frame["category"].astype(str).str.strip() != ""
    results["category"] = _train_task(
        features[in_scope.to_numpy()],
        frame.loc[in_scope, "category"].astype(str).to_numpy(),
        groups[in_scope.to_numpy()],
        "category",
        output_dir,
    )
    report_path = output_dir / "evaluation.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train calibrated query intent and category models")
    parser.add_argument("--csv", default="data/train/queries_labeled.csv")
    parser.add_argument("--config", default="trustrag/config.yaml")
    args = parser.parse_args()
    if not Path(args.csv).exists():
        raise FileNotFoundError(f"Labeled query dataset not found: {args.csv}")
    print(json.dumps(train_classifiers(args.csv, args.config), indent=2))


if __name__ == "__main__":
    main()