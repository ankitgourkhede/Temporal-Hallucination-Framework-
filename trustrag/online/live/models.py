"""Structured document returned by a live advisory adapter."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class LiveDocument:
    url: str
    title: str
    source_domain: str
    source_type: str
    nationality_scope: str
    country: str
    category: str
    content_date: str | None
    fetched_at: str
    text: str
    section: str
    raw_hash: str

    def as_dict(self) -> dict:
        return asdict(self)