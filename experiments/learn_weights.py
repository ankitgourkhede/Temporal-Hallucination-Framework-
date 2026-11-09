"""Fit per-category trust weights from train-split evidence labels."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

FEATURES = ("A", "F", "QE", "SR", "CP")
POSITIVE_FEATURES = ("A", "F", "QE", "SR")


def coefficients_to_weights(coefficients: dict[str, float]) -> tuple[dict[str, float], list[str]]:
    clipped = {name: max(0.0, float(coefficients.get(name, 0.0))) for name in POSITIVE_FEATURES}
    negative = [name for name in POSITIVE_FEATURES if float(coefficients.get(name, 0.0)) < 0]
    total = sum(clipped.values())
    if total <= 0:
        raise ValueError("All positive-feature coefficients are non-positive; cannot derive weights")
    weights = {
        "w_A": clipped["A"] / total,
        "w_F": clipped["F"] / total,
        "w_G": 0.0,
        "w_E": clipped["QE"] / total,
        "w_SR": clipped["SR"] / total,
        "w_C": max(0.0, -float(coefficients.get("CP", 0.0))) / total,
    }
    return weights, negative


def _vif(frame: pd.DataFrame, features: list[str]) -> dict[str, float]:
    from sklearn.linear_model import LinearRegression

    result = {}
    for feature in features:
        others = [name for name in features if name != feature]
        if not others:
            result[feature] = 1.0
            continue
        model = LinearRegression().fit(frame[others], frame[feature])
        r_squared = model.score(frame[others], frame[feature])
        result[feature] = float(1.0 / max(1e-8, 1.0 - r_squared))
    return result


def fit_category(frame: pd.DataFrame, *, seed: int = 42, bootstrap_samples: int = 1_000) -> dict[str, Any]:
    if not set((*FEATURES, "correct")) <= set(frame.columns):
        raise ValueError(f"Feature rows must include {', '.join((*FEATURES, 'correct'))}")
    X = frame[list(FEATURES)].astype(float).to_numpy()
    y = frame["correct"].astype(int).to_numpy()
    if len(np.unique(y)) < 2 or len(y) < 10:
        raise ValueError("Each category needs at least 10 examples and both correctness classes")
    estimator = LogisticRegression(penalty="l2", C=1.0, class_weight="balanced", max_iter=2_000, random_state=seed)
    fold_count = min(5, int(np.bincount(y).min()))
    if fold_count < 2:
        raise ValueError("Each correctness class needs at least two rows for cross-validation")
    cv = StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=seed)
    auc_values = cross_val_score(estimator, X, y, cv=cv, scoring="roc_auc")
    estimator.fit(X, y)
    coefficients = dict(zip(FEATURES, estimator.coef_[0].tolist()))
    weights, clipped = coefficients_to_weights(coefficients)

    rng = np.random.default_rng(seed)
    bootstrap_weights = []
    for _ in range(bootstrap_samples):
        indices = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[indices])) < 2:
            continue
        model = LogisticRegression(penalty="l2", C=1.0, class_weight="balanced", max_iter=2_000, random_state=seed)
        model.fit(X[indices], y[indices])
        sample_coefficients = dict(zip(FEATURES, model.coef_[0].tolist()))
        try:
            bootstrap_weights.append(coefficients_to_weights(sample_coefficients)[0])
        except ValueError:
            continue
    intervals = {
        key: [float(np.quantile([sample[key] for sample in bootstrap_weights], 0.025)),
              float(np.quantile([sample[key] for sample in bootstrap_weights], 0.975))]
        for key in weights
    } if bootstrap_weights else {key: [weights[key], weights[key]] for key in weights}
    frame_features = frame[list(FEATURES) + (["G"] if "G" in frame.columns else [])].astype(float)
    vif_without_g = _vif(frame_features, list(FEATURES))
    vif_with_g = _vif(frame_features, list(FEATURES) + (["G"] if "G" in frame_features else []))
    return {
        "weights": weights,
        "coefficients": coefficients,
        "clipped_negative_positive_coefficients": clipped,
        "intercept": float(estimator.intercept_[0]),
        "n": int(len(frame)),
        "class_balance": {str(label): int((y == label).sum()) for label in np.unique(y)},
        "cv_auc_mean": float(auc_values.mean()),
        "cv_auc_std": float(auc_values.std()),
        "weight_95_ci": intervals,
        "vif_without_g": vif_without_g,
        "vif_with_g": vif_with_g,
    }


def fit_all(features_csv: str | Path, output_dir: str | Path = "trustrag/policies/weights_learned") -> dict[str, Any]:
    frame = pd.read_csv(features_csv)
    if "split" not in frame or not (frame["split"] == "train").all():
        raise ValueError("Learned weights may only be fitted from rows in the train split")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for category, category_frame in frame.groupby("category"):
        result = fit_category(category_frame)
        results[str(category)] = result
        with (output / f"{category}.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(result, handle, sort_keys=False)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit learned trust weights from train evidence features")
    parser.add_argument("--features", default="data/train/trust_features.csv")
    parser.add_argument("--output", default="trustrag/policies/weights_learned")
    args = parser.parse_args()
    if not Path(args.features).exists():
        raise FileNotFoundError(f"Training evidence features not found: {args.features}")
    print(yaml.safe_dump(fit_all(args.features, args.output), sort_keys=False))


if __name__ == "__main__":
    main()