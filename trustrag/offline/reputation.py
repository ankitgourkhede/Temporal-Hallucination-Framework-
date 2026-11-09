"""Offline source reputation scoring.

This is intentionally conservative: reputation is updated only in evaluation mode or
via manual audit, which keeps production trust stable and avoids unrealistic swings
from sparse sources. The smoothing formula is a direct implementation of the
confidence-regularised estimate used in sparse-source settings.
"""

from __future__ import annotations

import sqlite3
import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path


class SourceReputationStore:
    def __init__(self, db_path: str | Path, prior: float = 0.5, alpha: float = 2.0, beta: float = 2.0):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.prior = prior
        self.alpha = alpha
        self.beta = beta
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(str(self.db_path))
        existing = conn.execute("PRAGMA table_info(reputation)").fetchall()
        columns = {row[1] for row in existing}
        if columns and "source_domain" not in columns:
            conn.execute("ALTER TABLE reputation RENAME TO reputation_legacy")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS reputation ("
            "source_domain TEXT PRIMARY KEY, n INTEGER NOT NULL DEFAULT 0, "
            "correct INTEGER NOT NULL DEFAULT 0, updated_at TEXT, split_used TEXT)"
        )
        conn.commit()
        conn.close()

    def update(self, source: str, correctness: bool, *, split: str = "train") -> None:
        if split != "train":
            raise ValueError("Source reputation can only be updated from the train split")
        conn = sqlite3.connect(str(self.db_path))
        row = conn.execute(
            "SELECT correct, n FROM reputation WHERE source_domain = ?",
            (source,),
        ).fetchone()
        correct = (row[0] if row else 0) + (1 if correctness else 0)
        count = (row[1] if row else 0) + 1
        conn.execute(
            "INSERT INTO reputation (source_domain, n, correct, updated_at, split_used) "
            "VALUES (?, ?, ?, ?, 'train') ON CONFLICT(source_domain) DO UPDATE SET "
            "n=excluded.n, correct=excluded.correct, updated_at=excluded.updated_at, split_used='train'",
            (source, count, correct, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        conn.close()

    def reset_train(self) -> None:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("DELETE FROM reputation WHERE split_used = 'train'")
        conn.commit()
        conn.close()

    def counts(self, source: str) -> tuple[int, int]:
        conn = sqlite3.connect(str(self.db_path))
        row = conn.execute(
            "SELECT n, correct FROM reputation WHERE source_domain = ?", (source,)
        ).fetchone()
        conn.close()
        return (int(row[0]), int(row[1])) if row else (0, 0)

    def score(self, source: str) -> float:
        n, correct = self.counts(source)
        return (correct + self.alpha) / (n + self.alpha + self.beta)

    def scores_for_sources(self, sources: list[str]) -> dict[str, float]:
        return {source: self.score(source) for source in sources}


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap source reputation from training labels")
    parser.add_argument("--split", choices=("train",), required=True)
    parser.add_argument("--labels", default="data/train/reputation_labels.csv")
    parser.add_argument("--db", default="data/store/trustrag.sqlite")
    args = parser.parse_args()
    labels_path = Path(args.labels)
    if not labels_path.exists():
        raise FileNotFoundError(f"Training reputation labels not found: {labels_path}")
    store = SourceReputationStore(args.db)
    with labels_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            store.update(row["source_domain"], row["correct"].lower() in {"1", "true", "yes"}, split=args.split)
