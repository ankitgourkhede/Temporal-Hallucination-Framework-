"""Demo-only parallel live fetcher with caching, robots checks, and stale flags."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from html import unescape
from pathlib import Path
from typing import Any, Callable
from urllib.error import URLError
from urllib.parse import urljoin, urlsplit
from urllib.request import Request, urlopen
from urllib.robotparser import RobotFileParser

import yaml
from bs4 import BeautifulSoup

from trustrag.online.live.govuk import GovUkSource
from trustrag.online.live.models import LiveDocument


class LiveCache:
    def __init__(self, database_path: str | Path):
        self.path = Path(database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS live_cache (url TEXT PRIMARY KEY, fetched_at TEXT NOT NULL, hash TEXT NOT NULL, payload TEXT NOT NULL)"
            )

    def get(self, url: str, ttl_seconds: int, now: datetime | None = None) -> tuple[str, str] | None:
        with sqlite3.connect(self.path) as connection:
            row = connection.execute("SELECT fetched_at, payload FROM live_cache WHERE url = ?", (url,)).fetchone()
        if row is None:
            return None
        fetched_at = datetime.fromisoformat(row[0])
        current = now or datetime.now(timezone.utc)
        if (current - fetched_at).total_seconds() >= ttl_seconds:
            return None
        return row[1], row[0]

    def put(self, url: str, payload: str, fetched_at: str | None = None) -> None:
        fetched_at = fetched_at or datetime.now(timezone.utc).isoformat()
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "INSERT INTO live_cache (url, fetched_at, hash, payload) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(url) DO UPDATE SET fetched_at=excluded.fetched_at, hash=excluded.hash, payload=excluded.payload",
                (url, fetched_at, digest, payload),
            )


class LiveFetcher:
    def __init__(
        self,
        config_path: str | Path,
        *,
        http_get: Callable[[str, float], bytes] | None = None,
        robots_allowed: Callable[[str], bool] | None = None,
        cache: LiveCache | None = None,
        min_interval_s: float = 1.0,
    ):
        self.config_path = Path(config_path)
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        paths = self.config["paths"]
        self.metadata_path = Path(paths["metadata_db"])
        self.live_config = self.config.get("live", {})
        self.http_get = http_get or self._http_get
        self.robots_allowed = robots_allowed or self._robots_allowed
        self.cache = cache or LiveCache(self.metadata_path)
        self.min_interval_s = min_interval_s
        self._last_request: dict[str, float] = {}
        self._rate_lock = threading.Lock()
        self._cache_times: dict[str, str] = {}

    def _http_get(self, url: str, timeout_s: float) -> bytes:
        host = urlsplit(url).netloc
        with self._rate_lock:
            remaining = self.min_interval_s - (time.monotonic() - self._last_request.get(host, 0.0))
            if remaining > 0:
                time.sleep(remaining)
            self._last_request[host] = time.monotonic()
        request = Request(url, headers={"User-Agent": "TrustRAG-research/0.1", "Accept": "application/json,text/html"})
        with urlopen(request, timeout=timeout_s) as response:
            return response.read()

    def _robots_allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt"
        parser = RobotFileParser()
        parser.set_url(robots_url)
        try:
            payload = self.http_get(robots_url, 3.0).decode("utf-8", errors="replace")
        except (URLError, TimeoutError, OSError):
            return False
        parser.parse(payload.splitlines())
        return parser.can_fetch("TrustRAG-research/0.1", url)

    def fetch(self, policy: dict[str, Any], *, mode: str = "demo", live_requested: bool = True) -> tuple[list[LiveDocument], list[str]]:
        if mode == "benchmark" or not live_requested or not self.live_config.get("enabled", False):
            return [], []
        destination = str(policy.get("entities", {}).get("destination") or "")
        nationality = str(policy.get("entities", {}).get("nationality") or "")
        category = str(policy.get("category") or "")
        if not destination or category not in {"visa_policy", "vaccination_requirement", "customs_regulation"}:
            return [], []

        tasks = []
        if self.live_config.get("govuk", {}).get("enabled", True):
            tasks.append(("govuk_fcdo", self._fetch_govuk, (destination, nationality, category)))
        for entry in self._whitelist_entries(destination, category):
            tasks.append(("whitelist", self._fetch_whitelist, (entry, destination, category)))
        if not tasks:
            return [], []

        timeout_s = float(self.live_config.get("per_adapter_timeout_s", 5.0))
        total_timeout_s = float(self.live_config.get("total_timeout_s", 6.0))
        documents: list[LiveDocument] = []
        flags: list[str] = []
        executor = ThreadPoolExecutor(max_workers=min(4, len(tasks)))
        started = time.monotonic()
        futures = {executor.submit(function, *arguments, timeout_s): name for name, function, arguments in tasks}
        try:
            for future in as_completed(futures, timeout=total_timeout_s):
                name = futures[future]
                try:
                    documents.extend(future.result())
                except Exception:
                    flags.append(f"live_check_failed:{name}")
        except TimeoutError:
            flags.extend(f"live_check_failed:{name}" for future, name in futures.items() if not future.done())
            for future in futures:
                future.cancel()
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        stale = self._record_stale_documents(documents)
        if stale:
            flags.append("index_stale")
        return documents, list(dict.fromkeys(flags))

    def _ttl(self, category: str) -> int:
        defaults = self.live_config.get("cache_ttl_s", {})
        return int(defaults.get("visa", 43_200) if category == "visa_policy" else defaults.get("other", 86_400))

    def _cached_get(self, url: str, timeout_s: float, category: str, *, check_robots: bool) -> bytes:
        cached = self.cache.get(url, self._ttl(category))
        if cached is not None:
            payload, fetched_at = cached
            self._cache_times[url] = fetched_at
            return payload.encode("utf-8")
        if check_robots and not self.robots_allowed(url):
            raise PermissionError(f"robots.txt disallows {url}")
        raw = self.http_get(url, timeout_s)
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("utf-8", errors="replace")
        fetched_at = datetime.now(timezone.utc).isoformat()
        self.cache.put(url, text, fetched_at)
        self._cache_times[url] = fetched_at
        return text.encode("utf-8")

    def _fetch_govuk(self, destination: str, nationality: str, category: str, timeout_s: float) -> list[LiveDocument]:
        adapter = GovUkSource(
            http_get=lambda url, timeout: self._cached_get(url, timeout, category, check_robots=False)
        )
        documents = adapter.fetch(destination, nationality, category, timeout_s)
        for document in documents:
            document.fetched_at = self._cache_times.get(
                GovUkSource.base_url.format(GovUkSource._find_country_slug(destination, adapter._index or {})),
                document.fetched_at,
            )
        return documents

    def _whitelist_entries(self, destination: str, category: str) -> list[dict[str, Any]]:
        whitelist = self.live_config.get("whitelist", {})
        destination_entries = whitelist.get(destination, {})
        names = {category, category.replace("_", " ").title()}
        entries = []
        for name in names:
            values = destination_entries.get(name, [])
            if isinstance(values, dict):
                values = [values]
            entries.extend(values)
        return entries

    def _fetch_whitelist(self, entry: dict[str, Any], destination: str, category: str, timeout_s: float) -> list[LiveDocument]:
        url = str(entry["url"])
        raw = self._cached_get(url, timeout_s, category, check_robots=True)
        html = raw.decode("utf-8", errors="replace")
        text = ""
        try:
            import trafilatura

            text = trafilatura.extract(html, include_comments=False, include_tables=True) or ""
        except Exception:
            text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", unescape(text)).strip()
        if not text:
            raise ValueError(f"No main text could be extracted from {url}")
        fetched_at = self._cache_times.get(url, datetime.now(timezone.utc).isoformat())
        content_date = self._extract_date(text)
        return [
            LiveDocument(
                url=url,
                title=str(entry.get("title", destination + " official travel requirements")),
                source_domain=urlsplit(url).hostname or "unknown",
                source_type=str(entry.get("source_type", "government")),
                nationality_scope=entry.get("nationality_scope", "UNKNOWN"),
                country=destination,
                category=category,
                content_date=content_date,
                fetched_at=fetched_at,
                text=text,
                section=category,
                raw_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            )
        ]

    @staticmethod
    def _extract_date(text: str) -> str | None:
        match = re.search(r"(?:last\s+updated|updated\s+on)\s*[:\-]?\s*(\d{4}-\d{2}-\d{2})", text, re.IGNORECASE)
        return match.group(1) if match else None

    def _record_stale_documents(self, documents: list[LiveDocument]) -> bool:
        stale = False
        connection = sqlite3.connect(self.metadata_path)
        for document in documents:
            base_url = document.url.split("#", 1)[0]
            row = connection.execute(
                "SELECT document_id, content_hash FROM documents WHERE url = ?", (base_url,)
            ).fetchone()
            if row is None or row[1] == document.raw_hash:
                continue
            connection.execute(
                "INSERT INTO pending_changes (document_id, url, old_hash, new_hash, detected_at) VALUES (?, ?, ?, ?, ?)",
                (row[0], base_url, row[1], document.raw_hash, document.fetched_at),
            )
            stale = True
        connection.commit()
        connection.close()
        return stale