"""Read-only access to the indexed offline knowledge base."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import yaml


class KnowledgeStore:
    """Search collections and read provenance without mutating offline data."""

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        with self.config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        paths = config["paths"]
        self.index_paths = {
            "primary": Path(paths.get("primary_faiss_path", "data/store/primary.faiss")),
            "secondary": Path(paths.get("secondary_faiss_path", "data/store/secondary.faiss")),
        }
        self.metadata_path = Path(paths["metadata_db"]).resolve()
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata store not found: {self.metadata_path}")
        self.connection = sqlite3.connect(self.metadata_path.as_uri() + "?mode=ro", uri=True)
        self.connection.row_factory = sqlite3.Row

    @classmethod
    def from_config(cls, config_path: str | Path) -> "KnowledgeStore":
        return cls(config_path)

    def search(self, query_vec: np.ndarray, collection: str, k: int) -> list[dict[str, Any]]:
        if collection not in self.index_paths:
            raise ValueError("collection must be 'primary' or 'secondary'")
        if k <= 0:
            return []
        index_path = self.index_paths[collection]
        if not index_path.exists():
            return []
        index = faiss.read_index(str(index_path))
        if index.ntotal == 0:
            return []
        vector = np.asarray(query_vec, dtype="float32")
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        scores, row_ids = index.search(vector, min(k, index.ntotal))
        hits = []
        for score, row_id in zip(scores[0], row_ids[0]):
            if row_id < 0:
                continue
            row = self.connection.execute(
                "SELECT c.chunk_id, c.document_id FROM chunks AS c "
                "JOIN documents AS d ON d.document_id = c.document_id "
                "WHERE c.faiss_collection = ? AND c.faiss_row = ? AND d.status = 'active'",
                (collection, int(row_id)),
            ).fetchone()
            if row is not None:
                hits.append(
                    {
                        "chunk_id": row["chunk_id"],
                        "document_id": row["document_id"],
                        "score": float(score),
                    }
                )
        return hits

    def get_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        return dict(row) if row is not None else None

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM documents WHERE document_id = ?", (document_id,)
        ).fetchone()
        return dict(row) if row is not None else None

    def change_history(self, category: str, country: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM change_events WHERE category = ?"
        parameters: tuple[Any, ...] = (category,)
        if country is not None:
            query += " AND country = ?"
            parameters += (country,)
        query += " ORDER BY detected_at, event_id"
        return [dict(row) for row in self.connection.execute(query, parameters).fetchall()]

    def reputation(self, source_domain: str) -> tuple[int, int]:
        row = self.connection.execute(
            "SELECT n, correct FROM reputation WHERE source_domain = ?", (source_domain,)
        ).fetchone()
        return (int(row["n"]), int(row["correct"])) if row is not None else (0, 0)

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "KnowledgeStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()