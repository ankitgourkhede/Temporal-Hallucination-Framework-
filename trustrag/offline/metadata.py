"""Pydantic metadata models and a lightweight SQLite metadata store.

This module gives each document a provenance record and a reproducible identity.
The project can later attach evidence, authority, and trust features to the same
record without re-deriving the document source.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from trustrag.offline.change_detect import compute_diff_magnitude


class DocumentMetadata(BaseModel):
    """Metadata captured for a single document snapshot."""

    document_id: str
    title: str
    url: str
    source_domain: str
    source_type: str
    authority_level: float = Field(..., ge=0.0, le=1.0)
    last_verified_ts: str
    published_ts: str
    crawl_timestamp: str
    diff_magnitude: float = Field(..., ge=0.0, le=1.0)
    content_hash: str
    embedding_version: str
    category: str
    country: str
    synthetic: bool = False
    content: str = ""
    doc_key: str = ""
    is_primary: bool = False
    nationality_scope: str = "UNKNOWN"
    content_date: str | None = None
    status: str = "active"
    current_version: int = 1
    snapshot_date: str = ""


def init_metadata_store(db_path: str | Path) -> sqlite3.Connection:
    """Create or migrate the SQLite metadata store used by offline and online phases."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            document_id TEXT PRIMARY KEY,
            doc_key TEXT,
            title TEXT,
            url TEXT,
            source_domain TEXT,
            source_type TEXT,
            authority_level REAL,
            is_primary INTEGER DEFAULT 0,
            nationality_scope TEXT DEFAULT 'UNKNOWN',
            country TEXT,
            category TEXT,
            content_date TEXT,
            last_verified_ts TEXT,
            published_ts TEXT,
            crawl_timestamp TEXT,
            status TEXT DEFAULT 'active',
            synthetic INTEGER DEFAULT 0,
            current_version INTEGER DEFAULT 1,
            diff_magnitude REAL,
            content_hash TEXT,
            embedding_version TEXT,
            content TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            document_id TEXT,
            version INTEGER DEFAULT 1,
            chunk_index INTEGER DEFAULT 0,
            heading_path TEXT DEFAULT '',
            text TEXT,
            n_words INTEGER DEFAULT 0,
            faiss_collection TEXT DEFAULT 'secondary',
            faiss_row INTEGER,
            embedding_version TEXT,
            chunk_text TEXT,
            title TEXT,
            country TEXT,
            category TEXT,
            source_type TEXT,
            authority_level REAL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS document_versions (
            document_id TEXT NOT NULL,
            version INTEGER NOT NULL,
            content_hash TEXT NOT NULL,
            text TEXT NOT NULL,
            snapshot_date TEXT NOT NULL,
            PRIMARY KEY (document_id, version)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS change_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id TEXT NOT NULL,
            category TEXT,
            country TEXT,
            diff_magnitude REAL NOT NULL,
            detected_at TEXT NOT NULL,
            old_hash TEXT,
            new_hash TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS reputation (
            source_domain TEXT PRIMARY KEY,
            n INTEGER NOT NULL DEFAULT 0,
            correct INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT,
            split_used TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS live_cache (
            url TEXT PRIMARY KEY,
            fetched_at TEXT NOT NULL,
            hash TEXT NOT NULL,
            payload TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS pending_changes (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id TEXT,
            url TEXT NOT NULL,
            old_hash TEXT,
            new_hash TEXT NOT NULL,
            detected_at TEXT NOT NULL
        )
        """
    )
    _migrate_columns(
        conn,
        "documents",
        {
            "doc_key": "TEXT",
            "is_primary": "INTEGER DEFAULT 0",
            "nationality_scope": "TEXT DEFAULT 'UNKNOWN'",
            "country": "TEXT",
            "category": "TEXT",
            "content_date": "TEXT",
            "status": "TEXT DEFAULT 'active'",
            "current_version": "INTEGER DEFAULT 1",
        },
    )
    _migrate_columns(
        conn,
        "chunks",
        {
            "version": "INTEGER DEFAULT 1",
            "chunk_index": "INTEGER DEFAULT 0",
            "heading_path": "TEXT DEFAULT ''",
            "text": "TEXT",
            "n_words": "INTEGER DEFAULT 0",
            "faiss_collection": "TEXT DEFAULT 'secondary'",
            "faiss_row": "INTEGER",
            "embedding_version": "TEXT",
        },
    )
    for row in conn.execute("SELECT source_domain FROM documents WHERE source_domain IS NOT NULL"):
        conn.execute("INSERT OR IGNORE INTO reputation (source_domain) VALUES (?)", (row[0],))
    conn.commit()
    return conn


def _migrate_columns(conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    for name, declaration in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {declaration}")


def upsert_document(conn: sqlite3.Connection, metadata: DocumentMetadata) -> None:
    """Insert a document snapshot, refreshing unchanged documents without versioning."""
    existing = conn.execute(
        "SELECT content_hash, content, current_version FROM documents WHERE document_id = ?",
        (metadata.document_id,),
    ).fetchone()
    current_version = int(existing[2] or 1) if existing else 1
    changed = existing is not None and existing[0] != metadata.content_hash
    if changed:
        current_version += 1
        old_hash = str(existing[0] or "")
        diff = compute_diff_magnitude(existing[1] or "", metadata.content)
        if diff >= 0.02 and metadata.is_primary:
            conn.execute(
                "INSERT INTO change_events (document_id, category, country, diff_magnitude, detected_at, old_hash, new_hash) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    metadata.document_id,
                    metadata.category,
                    metadata.country,
                    diff,
                    metadata.snapshot_date or metadata.last_verified_ts,
                    old_hash,
                    metadata.content_hash,
                ),
            )
    else:
        diff = float(metadata.diff_magnitude)

    snapshot_date = metadata.snapshot_date or metadata.last_verified_ts
    conn.execute(
        """
        INSERT INTO documents (
            document_id, doc_key, title, url, source_domain, source_type, authority_level,
            is_primary, nationality_scope, country, category, content_date, last_verified_ts, published_ts,
            crawl_timestamp, status, synthetic, current_version, diff_magnitude,
            content_hash, embedding_version, content
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(document_id) DO UPDATE SET
            doc_key=excluded.doc_key, title=excluded.title, url=excluded.url,
            source_domain=excluded.source_domain, source_type=excluded.source_type,
            authority_level=excluded.authority_level, is_primary=excluded.is_primary,
            nationality_scope=excluded.nationality_scope,
            country=excluded.country, category=excluded.category, content_date=excluded.content_date,
            last_verified_ts=excluded.last_verified_ts, published_ts=excluded.published_ts,
            crawl_timestamp=excluded.crawl_timestamp, status=excluded.status,
            synthetic=excluded.synthetic, current_version=excluded.current_version,
            diff_magnitude=excluded.diff_magnitude, content_hash=excluded.content_hash,
            embedding_version=excluded.embedding_version, content=excluded.content
        """,
        (
            metadata.document_id, metadata.doc_key, metadata.title, metadata.url,
            metadata.source_domain, metadata.source_type, metadata.authority_level,
            int(metadata.is_primary), metadata.nationality_scope, metadata.country, metadata.category, metadata.content_date,
            metadata.last_verified_ts, metadata.published_ts, metadata.crawl_timestamp,
            metadata.status, int(metadata.synthetic), current_version, diff,
            metadata.content_hash, metadata.embedding_version, metadata.content,
        ),
    )
    if existing is None or changed:
        conn.execute(
            "INSERT OR REPLACE INTO document_versions (document_id, version, content_hash, text, snapshot_date) "
            "VALUES (?, ?, ?, ?, ?)",
            (metadata.document_id, current_version, metadata.content_hash, metadata.content, snapshot_date),
        )
    conn.execute(
        "INSERT OR IGNORE INTO reputation (source_domain, n, correct) VALUES (?, 0, 0)",
        (metadata.source_domain,),
    )
    conn.commit()


def fetch_document_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Return all row dictionaries from the metadata table."""
    rows = conn.execute(
        "SELECT document_id, doc_key, title, url, source_domain, source_type, authority_level, is_primary, "
        "nationality_scope, country, category, content_date, last_verified_ts, published_ts, crawl_timestamp, status, synthetic, "
        "current_version, diff_magnitude, content_hash, embedding_version, content "
        "FROM documents ORDER BY rowid"
    ).fetchall()
    return [
        {
            "document_id": row[0],
            "doc_key": row[1],
            "title": row[2],
            "url": row[3],
            "source_domain": row[4],
            "source_type": row[5],
            "authority_level": row[6],
            "is_primary": bool(row[7]),
            "nationality_scope": row[8],
            "country": row[9],
            "category": row[10],
            "content_date": row[11],
            "last_verified_ts": row[12],
            "published_ts": row[13],
            "crawl_timestamp": row[14],
            "status": row[15],
            "synthetic": bool(row[16]),
            "current_version": row[17],
            "diff_magnitude": row[18],
            "content_hash": row[19],
            "embedding_version": row[20],
            "content": row[21],
        }
        for row in rows
    ]


def upsert_chunk(
    conn: sqlite3.Connection,
    chunk_id: str,
    document_id: str,
    chunk_text: str,
    title: str,
    country: str,
    category: str,
    source_type: str,
    authority_level: float,
    *,
    version: int = 1,
    chunk_index: int = 0,
    heading_path: str = "",
    faiss_collection: str = "secondary",
    faiss_row: int | None = None,
    embedding_version: str = "",
) -> None:
    """Store chunk metadata in SQLite so vector search results can resolve document provenance."""
    conn.execute(
        "INSERT OR REPLACE INTO chunks (chunk_id, document_id, version, chunk_index, heading_path, text, n_words, "
        "faiss_collection, faiss_row, embedding_version, chunk_text, title, country, category, source_type, authority_level) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            chunk_id, document_id, version, chunk_index, heading_path, chunk_text,
            len(chunk_text.split()), faiss_collection, faiss_row, embedding_version,
            chunk_text, title, country, category, source_type, authority_level,
        ),
    )
    conn.commit()


def fetch_chunk_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Return chunk metadata rows ordered by chunk index."""
    rows = conn.execute(
        "SELECT chunk_id, document_id, version, chunk_index, heading_path, text, n_words, faiss_collection, "
        "faiss_row, embedding_version, title, country, category, source_type, authority_level "
        "FROM chunks WHERE 1=1 ORDER BY rowid"
    ).fetchall()
    return [
        {
            "chunk_id": row[0],
            "document_id": row[1],
            "version": row[2],
            "chunk_index": row[3],
            "heading_path": row[4],
            "text": row[5],
            "chunk_text": row[5],
            "n_words": row[6],
            "faiss_collection": row[7],
            "faiss_row": row[8],
            "embedding_version": row[9],
            "title": row[10],
            "country": row[11],
            "category": row[12],
            "source_type": row[13],
            "authority_level": row[14],
        }
        for row in rows
    ]
