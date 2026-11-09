"""Change detection utilities for offline document versioning.

The objective is to decide whether a document changed materially enough to warrant a
new embedding, a fresh metadata snapshot, and a potentially different trust score.
"""

from __future__ import annotations

import difflib
import hashlib


def hash_content(text: str) -> str:
    """Return a deterministic SHA-256 hash for a text snapshot."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_diff_magnitude(previous_text: str | None, current_text: str) -> float:
    """Return a normalised difference value in the range [0, 1].

    A high value means the document changed substantially and should be re-indexed,
    while a low value suggests the same underlying information still holds.
    """
    if previous_text is None:
        return 1.0
    ratio = difflib.SequenceMatcher(None, previous_text, current_text).ratio()
    return round(1.0 - ratio, 6)
