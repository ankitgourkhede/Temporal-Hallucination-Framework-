"""FAISS index building and cosine-similarity search for the offline store.

The design matches the project specification: the index stores evidence chunks, each
connected to a document metadata record, and it keeps separate logical collections for
primary and secondary sources. This allows policy-governed retrieval to prefer trusted
sources before consulting lower-trust validation sources.
"""

from __future__ import annotations

import json
import hashlib
import logging
import math
import re
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import yaml

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - graceful fallback for constrained environments.
    SentenceTransformer = None  # type: ignore[assignment]

from trustrag.offline.ingest import ingest_corpus
from trustrag.offline.metadata import DocumentMetadata, fetch_chunk_rows, fetch_document_rows, init_metadata_store, upsert_chunk, upsert_document


LOGGER = logging.getLogger(__name__)

class _FallbackEncoder:
    """Deterministic fallback encoder used when sentence-transformers is unavailable."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        LOGGER.warning(
            "sentence-transformers model unavailable: using the lexical fallback encoder. "
            "Retrieval quality is reduced - run `make models` before producing thesis results."
        )

    def encode(self, texts: list[str], **_: object) -> np.ndarray:
        """Hashed bag-of-words: each token maps to a fixed signed dimension, so texts sharing words are similar.

        (The previous version hashed token *positions*, which made similarity essentially random.)
        """
        vectors = []
        for text in texts:
            vector = np.zeros(self.dimension, dtype="float32")
            tokens = [token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 1]
            if not tokens:
                vectors.append(vector)
                continue
            for token in tokens:
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big") % self.dimension
                vector[index] += 1.0 if digest[4] % 2 == 0 else -1.0
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            vectors.append(vector)
        return np.vstack(vectors) if vectors else np.zeros((0, self.dimension), dtype="float32")


def encode_query_vector(query: str, config_path: str | Path) -> np.ndarray:
    """Encode a query without opening the writable offline index or metadata store."""
    config = OfflineFAISSIndex._load_config(Path(config_path))
    embedding = config["embedding"]
    dimension = int(embedding.get("dimension", 384))
    prefix = str(embedding.get("query_prefix", ""))
    model_name = str(embedding.get("model_name", "BAAI/bge-small-en-v1.5"))
    text = prefix + query
    encoder = None
    if SentenceTransformer is not None:
        try:
            encoder = SentenceTransformer(
                model_name, device="cpu", local_files_only=True
            )
        except Exception:
            pass
    encoder = encoder or _FallbackEncoder(dimension)
    vector = np.asarray(encoder.encode([text])[0], dtype="float32")[:dimension]
    if vector.shape[0] < dimension:
        vector = np.pad(vector, (0, dimension - vector.shape[0]))
    if bool(embedding.get("normalize", True)):
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
    return vector


class OfflineFAISSIndex:
    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        self.config = self._load_config(self.config_path)
        paths = self.config["paths"]
        self.faiss_paths = {
            "primary": Path(paths.get("primary_faiss_path", paths.get("faiss_path", "data/store/primary.faiss"))),
            "secondary": Path(paths.get("secondary_faiss_path", "data/store/secondary.faiss")),
        }
        self.faiss_path = self.faiss_paths["primary"]
        self.metadata_db = Path(self.config["paths"]["metadata_db"])
        self.index_dir = Path(self.config["paths"]["index_dir"])
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_cfg = self.config["embedding"]
        self.offline_cfg = self.config.get("offline", {})
        self.primary_collections = {
            source.lower() for source in self.offline_cfg.get("primary_collections", [])
        }
        self.secondary_collections = {
            source.lower() for source in self.offline_cfg.get("secondary_collections", [])
        }
        self.dimension = int(self.embedding_cfg.get("dimension", 384))
        self.model_name = self.embedding_cfg.get("model_name", "BAAI/bge-small-en-v1.5")
        self.normalise = bool(self.embedding_cfg.get("normalize", True))
        self.embedding_version = f"{self.model_name}|norm=l2|v1"
        self.encoder = self._build_encoder()
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.dimension))
        self.metadata_store = init_metadata_store(self.metadata_db)
        self.doc_rows: list[dict[str, Any]] = fetch_document_rows(self.metadata_store)
        self.chunk_rows: list[dict[str, Any]] = fetch_chunk_rows(self.metadata_store)

    @staticmethod
    def _load_config(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    def _build_encoder(self):
        if SentenceTransformer is not None:
            try:
                return SentenceTransformer(self.model_name, device="cpu")
            except Exception:  # pragma: no cover - fallback path
                pass
        return _FallbackEncoder(self.dimension)

    def _chunk_text(self, text: str, chunk_size: int = 180, overlap: int = 30) -> list[str]:
        sentences = [sentence.strip() for sentence in __import__("re").split(r"(?<=[.!?])\s+", text) if sentence.strip()]
        chunks: list[str] = []
        if not sentences:
            return chunks
        word_chunks: list[list[str]] = []
        current: list[str] = []
        for sentence in sentences:
            sentence_words = sentence.split()
            if current and len(current) + len(sentence_words) > chunk_size:
                word_chunks.append(current)
                current = current[max(0, len(current) - overlap) :]
            current.extend(sentence_words)
        if current:
            word_chunks.append(current)
        for words in word_chunks:
            if len(words) >= 25 or len(word_chunks) == 1:
                chunks.append(" ".join(words))
        return chunks

    def build_from_directory(self, corpus_dir: str | Path) -> dict[str, Any]:
        """Ingest documents and rebuild separate primary and secondary indexes."""
        docs = ingest_corpus(corpus_dir)
        chunks_by_collection: dict[str, list[dict[str, Any]]] = {"primary": [], "secondary": []}
        self.metadata_store.execute("DELETE FROM chunks")

        for doc in docs:
            upsert_document(self.metadata_store, doc)
            version = int(self.metadata_store.execute(
                "SELECT current_version FROM documents WHERE document_id = ?", (doc.document_id,)
            ).fetchone()[0])
            collection = "primary" if doc.is_primary else "secondary"
            chunk_list = self._chunk_text(
                doc.content,
                int(self.embedding_cfg.get("chunk_size", 180)),
                int(self.embedding_cfg.get("chunk_overlap", 30)),
            )
            for index, chunk in enumerate(chunk_list):
                row = len(chunks_by_collection[collection])
                chunk_id = f"{doc.document_id}-v{version}-chunk-{index}"
                upsert_chunk(
                    self.metadata_store,
                    chunk_id,
                    doc.document_id,
                    chunk,
                    doc.title,
                    doc.country,
                    doc.category,
                    doc.source_type,
                    doc.authority_level,
                    version=version,
                    chunk_index=index,
                    faiss_collection=collection,
                    faiss_row=row,
                    embedding_version=self.embedding_version,
                )
                chunks_by_collection[collection].append({
                    "document_id": doc.document_id,
                    "title": doc.title,
                    "source_type": doc.source_type,
                    "country": doc.country,
                    "category": doc.category,
                    "chunk_text": chunk,
                    "authority_level": doc.authority_level,
                    "chunk_id": chunk_id,
                    "faiss_row": row,
                })

        for collection, chunks in chunks_by_collection.items():
            index = faiss.IndexIDMap(faiss.IndexFlatIP(self.dimension))
            if chunks:
                texts = [chunk["chunk_text"] for chunk in chunks]
                vectors = np.asarray(
                    self.encoder.encode(texts, batch_size=int(self.embedding_cfg.get("batch_size", 32))),
                    dtype="float32",
                )
                if self.normalise:
                    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                    norms[norms == 0] = 1.0
                    vectors = vectors / norms
                row_ids = np.asarray([chunk["faiss_row"] for chunk in chunks], dtype="int64")
                index.add_with_ids(vectors, row_ids)
            index_path = self.faiss_paths[collection]
            index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(index, str(index_path))
            if collection == "primary":
                self.index = index

        self.doc_rows = fetch_document_rows(self.metadata_store)
        self.chunk_rows = fetch_chunk_rows(self.metadata_store)
        total_chunks = sum(len(chunks) for chunks in chunks_by_collection.values())
        return {
            "documents": len(docs),
            "chunks": total_chunks,
            "primary_chunks": len(chunks_by_collection["primary"]),
            "secondary_chunks": len(chunks_by_collection["secondary"]),
            "index_path": str(self.faiss_path),
        }

    def _encode_single(self, text: str) -> np.ndarray:
        vectors = self.encoder.encode([text]) if hasattr(self.encoder, "encode") else np.array([np.zeros(self.dimension)], dtype="float32")
        arr = np.asarray(vectors[0], dtype="float32")
        if arr.shape[0] != self.dimension:
            pad = np.zeros(self.dimension, dtype="float32")
            pad[: arr.shape[0]] = arr[: self.dimension]
            arr = pad
        norm = np.linalg.norm(arr)
        if self.normalise and norm > 0:
            arr = arr / norm
        return arr

    def encode_query(self, query: str) -> np.ndarray:
        prefix = str(self.embedding_cfg.get("query_prefix", ""))
        return self._encode_single(prefix + query)

    @classmethod
    def from_config(cls, config_path: str | Path) -> "OfflineFAISSIndex":
        return cls(config_path)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Search by similarity, prioritizing configured primary source types."""
        self.doc_rows = fetch_document_rows(self.metadata_store)
        self.chunk_rows = fetch_chunk_rows(self.metadata_store)
        query_vector = np.asarray([self.encode_query(query)], dtype="float32")
        hits = []
        chunks_by_row = {
            collection: {
                int(chunk["faiss_row"]): chunk
                for chunk in self.chunk_rows
                if chunk["faiss_collection"] == collection and chunk["faiss_row"] is not None
            }
            for collection in ("primary", "secondary")
        }
        for collection in ("primary", "secondary"):
            index_path = self.faiss_paths[collection]
            if not index_path.exists():
                continue
            index = faiss.read_index(str(index_path))
            if index.ntotal == 0:
                continue
            similarity, indices = index.search(query_vector, min(index.ntotal, max(top_k, 1)))
            for score, idx in zip(similarity[0], indices[0]):
                chunk = chunks_by_row[collection].get(int(idx))
                if chunk is None:
                    continue
                doc_lookup = next((doc for doc in self.doc_rows if doc.get("document_id") == chunk.get("document_id")), {})
                hits.append({
                    "score": float(score),
                    "chunk_id": chunk["chunk_id"],
                    "chunk_text": chunk["chunk_text"],
                    "document_id": chunk.get("document_id", "unknown"),
                    "title": chunk.get("title", doc_lookup.get("title", "unknown")),
                    "country": chunk.get("country", doc_lookup.get("country", "unknown")),
                    "category": chunk.get("category", doc_lookup.get("category", "visa_policy")),
                    "source_type": chunk.get("source_type", doc_lookup.get("source_type", "unknown")),
                    "url": doc_lookup.get("url", ""),
                    "is_primary": collection == "primary",
                })
        return hits[:top_k]
