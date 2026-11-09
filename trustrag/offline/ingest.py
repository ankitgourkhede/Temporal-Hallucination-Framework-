"""Seed corpus ingestion for the offline knowledge layer.

The project is designed to ingest a static corpus first and index it deterministically
before any online retrieval runs. This keeps the online pipeline fully evidence-bound
and prevents unverified web access from contaminating the answer generation.
"""

from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml

from trustrag.offline.change_detect import compute_diff_magnitude, hash_content
from trustrag.offline.metadata import DocumentMetadata


def _load_payload(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {"text": value}
    for sidecar in (path.with_suffix(".meta.json"), Path(f"{path}.meta.json")):
        if sidecar.exists():
            value = json.loads(sidecar.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                return value
    return {}


def _clean_content(path: Path, payload: dict[str, Any]) -> str:
    if path.suffix.lower() == ".json":
        text = payload.get("text", payload.get("content", ""))
    else:
        raw = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".html":
            try:
                import trafilatura

                text = trafilatura.extract(raw, include_comments=False, include_tables=True) or ""
            except ImportError:
                try:
                    from bs4 import BeautifulSoup

                    text = BeautifulSoup(raw, "html.parser").get_text("\n")
                except ImportError:
                    text = re.sub(r"<[^>]+>", " ", raw)
        else:
            text = raw
    text = unicodedata.normalize("NFKC", str(text))
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def _infer_source_type(url: str, payload: dict[str, Any] | None) -> str:
    if payload and payload.get("source_type"):
        explicit = str(payload["source_type"]).lower()
        return {
            "blog": "travel_blog",
            "forum": "forum_social",
            "iata": "intl_body",
            "who": "intl_body",
        }.get(explicit, explicit)
    parsed = urlsplit(url)
    domain = parsed.netloc.lower()
    path = parsed.path.lower()
    if any(suffix in domain for suffix in (".gov", ".gov.in", ".gov.vn", ".gc.ca", ".gov.uk")):
        return "government"
    if any(term in f"{domain} {path}" for term in ("embassy", "consulate", "high-commission")):
        return "embassy"
    if any(term in domain for term in ("iata.org", "who.int", "icao.int", "un.org")):
        return "intl_body"
    if any(term in domain for term in ("airline", "airways", "qatarairways", "singaporeair")):
        return "airline"
    if any(term in domain for term in ("medium.com", "blogspot", "wordpress")) or "/blog/" in path:
        return "travel_blog"
    if any(term in domain for term in ("reddit.com", "quora.com", "tripadvisor.")):
        return "forum_social"
    if any(term in domain for term in ("news", "bbc.", "reuters.com", "apnews.com")):
        return "news"
    return "unknown"


def _infer_authority(source_type: str) -> float:
    authority_map = {
        "government": 1.0,
        "embassy": 0.95,
        "intl_body": 0.9,
        "airline": 0.8,
        "news": 0.6,
        "travel_blog": 0.35,
        "forum_social": 0.2,
        "unknown": 0.3,
    }
    return authority_map.get(source_type, 0.3)


def _parse_content_date(
    text: str, payload: dict[str, Any], snapshot_date: str | None = None
) -> str | None:
    candidates = [
        str(payload[key])
        for key in ("content_date", "effective_date", "published_ts", "published_date")
        if payload.get(key)
    ]
    candidates.extend(re.findall(
        r"(?:last\s+updated|updated\s+on|effective\s+from|dated)\s*[:\-]?\s*"
        r"([A-Za-z]+\s+\d{1,2},?\s+\d{4}|\d{1,2}\s+[A-Za-z]+\s+\d{4}|\d{4}-\d{2}-\d{2})",
        text,
        flags=re.IGNORECASE,
    ))
    anchor = datetime.fromisoformat((snapshot_date or datetime.now(timezone.utc).date().isoformat())[:10]).date()
    parsed_dates = []
    formats = ("%Y-%m-%d", "%d %B %Y", "%B %d, %Y", "%B %d %Y", "%d %b %Y", "%b %d, %Y")
    for candidate in candidates:
        parsed = None
        for date_format in formats:
            try:
                parsed = datetime.strptime(candidate.strip(), date_format).date()
                break
            except ValueError:
                continue
        if parsed is None:
            try:
                import dateparser

                value = dateparser.parse(candidate)
                parsed = value.date() if value else None
            except ImportError:
                parsed = None
        if parsed is not None and parsed <= anchor:
            parsed_dates.append(parsed)
    return max(parsed_dates).isoformat() if parsed_dates else None


def _canonical_category(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    aliases = {
        "visa_policy": "visa_policy",
        "vaccination_requirement": "vaccination_requirement",
        "health_entry": "vaccination_requirement",
        "customs_regulation": "customs_regulation",
    }
    return aliases.get(normalized, normalized or "visa_policy")


def _nationality_scope(text: str, payload: dict[str, Any]) -> str:
    scope = payload.get("nationality_scope")
    if isinstance(scope, list):
        return ",".join(str(item).upper() for item in scope)
    if scope:
        return str(scope).upper()
    lowered = text.lower()
    for terms, code in (
        (("indian", "from india", "india passport"), "IN"),
        (("british", "uk nationals", "united kingdom passport"), "GB"),
        (("american", "us nationals", "united states passport"), "US"),
        (("canadian",), "CA"),
        (("australian",), "AU"),
    ):
        if any(term in lowered for term in terms):
            return code
    return "UNKNOWN"


def ingest_corpus(
    directory: str | Path,
    authority_path: str | Path | None = None,
) -> list[DocumentMetadata]:
    """Load all seed corpus files and convert them into metadata objects."""
    corpus_dir = Path(directory)
    authority_file = Path(authority_path) if authority_path else Path(__file__).parents[1] / "policies" / "authority.yaml"
    authority_config = yaml.safe_load(authority_file.read_text(encoding="utf-8")) or {}
    authority_map = authority_config.get("tiers", authority_config)
    domain_overrides = authority_config.get("domain_overrides", {})
    documents: list[DocumentMetadata] = []
    if not corpus_dir.exists():
        return documents

    for path in sorted(corpus_dir.rglob("*")):
        if path.is_dir() or path.suffix.lower() not in {".json", ".md", ".html", ".txt"}:
            continue

        payload = _load_payload(path)
        text = _clean_content(path, payload)
        if not text.strip():
            continue

        title = str(payload.get("title") or path.stem.replace("_", " ").title())
        country = str(payload.get("country") or "Unknown")
        category = _canonical_category(str(payload.get("category") or "visa_policy"))
        url = str(payload.get("url") or path.resolve().as_uri())
        source_domain = str(payload.get("source_domain") or urlsplit(url).hostname or path.parent.name or "local")
        source_domain = source_domain.lower()
        source_type = _infer_source_type(url, payload)
        for domain, override in domain_overrides.items():
            if source_domain == domain or source_domain.endswith(f".{domain}"):
                source_type = str(override).lower()
                break
        source_type = {"blog": "travel_blog", "forum": "forum_social"}.get(source_type, source_type)
        authority = float(authority_map.get(source_type, 0.3))
        snapshot_date = str(payload.get("snapshot_date") or payload.get("as_of_date") or datetime.now(timezone.utc).date().isoformat())
        content_date = _parse_content_date(text, payload, snapshot_date)
        doc_key = str(payload.get("doc_key") or f"{path.stem}_{hash_content(url)[:10]}")
        doc = DocumentMetadata(
            document_id=f"doc-{hash_content(doc_key)[:20]}",
            doc_key=doc_key,
            title=title,
            url=url,
            source_domain=source_domain,
            source_type=source_type,
            authority_level=authority,
            is_primary=source_type in {"government", "embassy", "intl_body"},
            nationality_scope=_nationality_scope(text, payload),
            last_verified_ts=snapshot_date,
            published_ts=str(payload.get("published_ts") or content_date or snapshot_date),
            crawl_timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            diff_magnitude=1.0,
            content_hash=hash_content(text),
            embedding_version="BAAI/bge-small-en-v1.5|norm=l2|v1",
            category=category,
            country=country,
            synthetic=bool(payload.get("synthetic", True)),
            content=text,
            content_date=content_date,
            snapshot_date=snapshot_date,
        )
        documents.append(doc)

    return documents
