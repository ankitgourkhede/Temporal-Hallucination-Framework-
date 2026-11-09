"""Benchmark schema and leakage checks for reproducible evaluation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {
    "query_id", "query", "nationality", "destination", "category", "volatility",
    "gold_slot", "gold_value", "gold_answer", "gold_source_url", "as_of_date",
    "difficulty", "has_conflict", "conflict_type", "paraphrase_group", "split",
}


def validate_benchmark(path: str | Path, classifier_data: str | Path | None = None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark CSV not found: {path}")
    frame = pd.read_csv(path, dtype=str).fillna("")
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Benchmark is missing columns: {', '.join(sorted(missing))}")
    if frame["query_id"].duplicated().any():
        raise ValueError("query_id values must be unique")
    if not set(frame["difficulty"]) <= {"easy", "medium", "hard"}:
        raise ValueError("difficulty must be easy, medium, or hard")
    if not set(frame["split"]) <= {"train", "dev", "test"}:
        raise ValueError("split must be train, dev, or test")
    if not set(frame["has_conflict"]) <= {"0", "1"}:
        raise ValueError("has_conflict must be 0 or 1")
    groups_per_split = frame.groupby("paraphrase_group")["split"].nunique()
    if (groups_per_split > 1).any():
        leaked = groups_per_split[groups_per_split > 1].index.tolist()
        raise ValueError(f"Paraphrase groups cross splits: {leaked}")
    if classifier_data is not None and Path(classifier_data).exists():
        classifier = pd.read_csv(classifier_data, dtype=str).fillna("")
        if "paraphrase_of" in classifier.columns:
            overlap = set(frame["paraphrase_group"]) & set(classifier["paraphrase_of"])
            if overlap:
                raise ValueError(f"Benchmark overlaps classifier training groups: {sorted(overlap)}")
    return frame


def check_nationality_coverage(frame: pd.DataFrame, minimum: float = 0.15) -> float:
    if frame.empty:
        raise ValueError("Benchmark is empty")
    fraction = float((frame["nationality"].str.casefold() != "india").mean())
    if fraction < minimum:
        raise ValueError(f"Non-Indian nationality coverage is {fraction:.1%}; requires at least {minimum:.0%}")
    return fraction