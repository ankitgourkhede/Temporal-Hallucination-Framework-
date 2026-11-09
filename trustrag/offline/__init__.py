"""Offline ingestion, indexing, and source metadata pipeline for Phase 1."""

from trustrag.offline.change_detect import compute_diff_magnitude, hash_content
from trustrag.offline.embed_index import OfflineFAISSIndex
from trustrag.offline.ingest import ingest_corpus
from trustrag.offline.metadata import DocumentMetadata, init_metadata_store
from trustrag.offline.reputation import SourceReputationStore

__all__ = [
    "OfflineFAISSIndex",
    "DocumentMetadata",
    "SourceReputationStore",
    "compute_diff_magnitude",
    "hash_content",
    "ingest_corpus",
    "init_metadata_store",
]
