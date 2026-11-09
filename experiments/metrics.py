"""Metric definitions for answer, evidence, faithfulness, and calibration evaluation."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def expected_calibration_error(confidences: Iterable[float], correct: Iterable[bool], bins: int = 10) -> float:
    confidence_values = np.asarray(list(confidences), dtype=float)
    correct_values = np.asarray(list(correct), dtype=float)
    if confidence_values.size != correct_values.size:
        raise ValueError("confidence and correctness arrays must have equal length")
    if not confidence_values.size:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        upper_inclusive = index == bins - 1
        mask = (confidence_values >= edges[index]) & (
            confidence_values <= edges[index + 1] if upper_inclusive else confidence_values < edges[index + 1]
        )
        if mask.any():
            ece += float(mask.mean()) * abs(float(confidence_values[mask].mean()) - float(correct_values[mask].mean()))
    return ece


def answer_accuracy(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        return {"accuracy": 0.0, "abstention_rate": 0.0, "selective_accuracy": 0.0}
    answered = [item for item in records if item.get("verification_status") != "VERIFICATION_REQUIRED"]
    correct = [bool(item.get("correct", False)) and item.get("verification_status") != "VERIFICATION_REQUIRED" for item in records]
    return {
        "accuracy": sum(correct) / len(records),
        "abstention_rate": 1.0 - len(answered) / len(records),
        "selective_accuracy": sum(bool(item.get("correct", False)) for item in answered) / len(answered) if answered else 0.0,
    }


def evidence_precision_recall(selected: set[Any], relevant: set[Any]) -> dict[str, float]:
    true_positive = len(selected & relevant)
    precision = true_positive / len(selected) if selected else 0.0
    recall = true_positive / len(relevant) if relevant else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def hallucination_rate(claims: Iterable[dict[str, Any]]) -> float:
    claims = list(claims)
    if not claims:
        return 0.0
    unsupported = sum(item.get("label") in {"UNSUPPORTED", "CONTRADICTED"} for item in claims)
    return unsupported / len(claims)


def faithfulness_bands(values: Iterable[float]) -> dict[str, float]:
    values = list(values)
    if not values:
        return {"ge_0_90": 0.0, "0_70_to_0_90": 0.0, "lt_0_70": 0.0}
    return {
        "ge_0_90": sum(value >= 0.90 for value in values) / len(values),
        "0_70_to_0_90": sum(0.70 <= value < 0.90 for value in values) / len(values),
        "lt_0_70": sum(value < 0.70 for value in values) / len(values),
    }