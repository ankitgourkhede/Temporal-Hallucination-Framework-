import pandas as pd
import pytest

from experiments.benchmark import check_nationality_coverage, validate_benchmark
from experiments.metrics import answer_accuracy, expected_calibration_error, evidence_precision_recall
from experiments.learn_weights import coefficients_to_weights
from experiments.stats import mcnemar


def test_ece_fixture_matches_spec() -> None:
    confidences = [0.95, 0.92, 0.88, 0.81, 0.76, 0.66, 0.62, 0.55, 0.35, 0.22]
    correctness = [1, 1, 1, 0, 1, 1, 0, 1, 0, 0]
    assert expected_calibration_error(confidences, correctness) == pytest.approx(0.236, abs=0.001)


def test_mcnemar_fixture_and_exact_branch() -> None:
    result = mcnemar(12, 37)
    exact = mcnemar(3, 5)
    assert result["statistic"] == pytest.approx(11.755, abs=0.001)
    assert result["p_value"] == pytest.approx(0.0006, abs=0.0001)
    assert exact["method"] == "exact"


def test_accuracy_counts_verification_required_as_wrong_and_abstention() -> None:
    result = answer_accuracy([
        {"correct": True, "verification_status": "VERIFIED"},
        {"correct": True, "verification_status": "VERIFICATION_REQUIRED"},
    ])
    assert result == {"accuracy": 0.5, "abstention_rate": 0.5, "selective_accuracy": 1.0, "wrong_answer_rate": 0.0}


def test_clarification_is_an_abstention_not_a_wrong_answer() -> None:
    result = answer_accuracy([
        {"correct": True, "verification_status": "VERIFIED"},
        {"correct": False, "verification_status": "VERIFIED"},
        {"correct": False, "verification_status": "CLARIFICATION"},
    ])
    assert result["accuracy"] == pytest.approx(1 / 3)
    assert result["abstention_rate"] == pytest.approx(1 / 3)
    assert result["wrong_answer_rate"] == pytest.approx(1 / 3)
    assert result["selective_accuracy"] == pytest.approx(0.5)


def test_evidence_precision_recall_fixture() -> None:
    result = evidence_precision_recall({"e1", "e2"}, {"e1", "e3"})
    assert result == {"precision": 0.5, "recall": 0.5, "f1": 0.5}


def test_benchmark_rejects_paraphrase_split_leakage(tmp_path) -> None:
    row = {
        "query_id": "q1", "query": "q", "nationality": "India", "destination": "Vietnam",
        "category": "visa_policy", "volatility": "High", "gold_slot": "max_stay_days",
        "gold_value": "30", "gold_answer": "30 days", "gold_source_url": "https://example.test",
        "as_of_date": "2026-11-09", "difficulty": "easy", "has_conflict": "0",
        "conflict_type": "none", "paraphrase_group": "g1", "split": "train",
    }
    frame = pd.DataFrame([row, {**row, "query_id": "q2", "split": "test"}])
    path = tmp_path / "benchmark.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="cross splits"):
        validate_benchmark(path)


def test_benchmark_checks_non_indian_coverage(tmp_path) -> None:
    frame = pd.DataFrame({"nationality": ["India", "United Kingdom", "United States"]})
    assert check_nationality_coverage(frame) >= 0.15
    with pytest.raises(ValueError, match="requires at least"):
        check_nationality_coverage(pd.DataFrame({"nationality": ["India"]}))


def test_learned_coefficients_convert_to_spec_weights() -> None:
    weights, clipped = coefficients_to_weights({"A": 2.4, "F": 1.8, "QE": 0.6, "SR": 1.2, "CP": -3.0})
    assert weights["w_A"] == pytest.approx(0.4)
    assert weights["w_F"] == pytest.approx(0.3)
    assert weights["w_E"] == pytest.approx(0.1)
    assert weights["w_SR"] == pytest.approx(0.2)
    assert weights["w_G"] == 0.0
    assert weights["w_C"] == pytest.approx(0.5)
    assert clipped == []


def test_learned_coefficients_report_negative_positive_features() -> None:
    weights, clipped = coefficients_to_weights({"A": 1.0, "F": -0.5, "QE": 1.0, "SR": 1.0, "CP": -1.0})
    assert weights["w_F"] == 0.0
    assert clipped == ["F"]