"""Google Programmable Search JSON API fallback for missing offline/live evidence."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Callable
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

from trustrag.online.live.models import LiveDocument


class GoogleSearchFallback:
    endpoint = "https://www.googleapis.com/customsearch/v1"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        search_engine_id: str | None = None,
        http_get: Callable[[str, float], bytes] | None = None,
    ):
        self.api_key = api_key if api_key is not None else os.environ.get("GOOGLE_CSE_API_KEY", "")
        self.search_engine_id = (
            search_engine_id if search_engine_id is not None else os.environ.get("GOOGLE_CSE_CX", "")
        )
        self.http_get = http_get or self._http_get

    @staticmethod
    def _http_get(url: str, timeout_s: float) -> bytes:
        request = Request(url, headers={"Accept": "application/json", "User-Agent": "TrustRAG-research/0.1"})
        with urlopen(request, timeout=timeout_s) as response:
            return response.read()

    def search(
        self,
        *,
        query: str,
        destination: str,
        nationality: str,
        category: str,
        limit: int = 5,
        timeout_s: float = 5.0,
    ) -> tuple[list[LiveDocument], list[str]]:
        if not self.api_key or not self.search_engine_id:
            return [], ["google_fallback_unavailable:missing_credentials"]
        search_query = f"{query} {destination} official government travel requirements"
        url = self.endpoint + "?" + urlencode(
            {"key": self.api_key, "cx": self.search_engine_id, "q": search_query, "num": min(10, max(1, limit))}
        )
        try:
            payload = json.loads(self.http_get(url, timeout_s).decode("utf-8"))
        except Exception as error:
            return [], [f"google_fallback_failed:{type(error).__name__}"]
        documents = self.parse_results(payload, category=category, destination=destination, nationality=nationality)
        return documents, [] if documents else ["google_fallback_no_results"]

    @classmethod
    def parse_results(
        cls, payload: dict[str, Any], *, category: str, destination: str, nationality: str
    ) -> list[LiveDocument]:
        fetched_at = datetime.now(timezone.utc).isoformat()
        documents = []
        for result in payload.get("items", []):
            link = str(result.get("link", ""))
            snippet = str(result.get("snippet", "")).strip()
            if not link or not snippet:
                continue
            domain = urlsplit(link).hostname or "unknown"
            source_type = cls._source_type(domain)
            text = " ".join(snippet.split())
            documents.append(
                LiveDocument(
                    url=link,
                    title=str(result.get("title", domain)),
                    source_domain=domain,
                    source_type=source_type,
                    nationality_scope="UNKNOWN",
                    country=destination,
                    category=category,
                    content_date=None,
                    fetched_at=fetched_at,
                    text=text,
                    section="google_search_snippet",
                    raw_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                )
            )
        return documents

    @staticmethod
    def _source_type(domain: str) -> str:
        domain = domain.lower()
        if domain.endswith((".gov", ".gov.uk", ".gov.sg", ".gov.vn", ".gov.in", ".go.jp")):
            return "government"
        if "embassy" in domain or "consulate" in domain:
            return "embassy"
        if domain.endswith(".who.int") or domain.endswith(".iata.org"):
            return "intl_body"
        return "unknown"