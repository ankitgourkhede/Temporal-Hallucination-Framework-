"""GOV.UK FCDO travel-advice content API adapter with passport scope labels."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from html import unescape
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup

from trustrag.online.live.models import LiveDocument


class GovUkSource:
    name = "govuk_fcdo"
    source_type = "foreign_gov_advisory"
    index_url = "https://www.gov.uk/api/content/foreign-travel-advice"
    base_url = "https://www.gov.uk/api/content/foreign-travel-advice/{}"

    def __init__(self, http_get=None):
        self.http_get = http_get or self._http_get
        self._index = None

    @staticmethod
    def _http_get(url: str, timeout_s: float) -> bytes:
        request = Request(url, headers={"Accept": "application/json", "User-Agent": "TrustRAG-research/0.1"})
        with urlopen(request, timeout=timeout_s) as response:
            return response.read()

    def supports(self, category: str, destination: str, nationality: str) -> bool:
        return category in {"visa_policy", "vaccination_requirement", "customs_regulation"} and bool(destination)

    def fetch(self, destination: str, nationality: str, category: str, timeout_s: float = 5.0) -> list[LiveDocument]:
        if not self.supports(category, destination, nationality):
            return []
        if self._index is None:
            self._index = json.loads(self.http_get(self.index_url, timeout_s).decode("utf-8"))
        country_slug = self._find_country_slug(destination, self._index)
        if not country_slug:
            return []
        payload = json.loads(self.http_get(self.base_url.format(country_slug), timeout_s).decode("utf-8"))
        return self.parse_page(payload, category)

    @staticmethod
    def _find_country_slug(destination: str, index_payload: dict[str, Any]) -> str | None:
        wanted = destination.casefold().strip()

        def find_country(value: Any) -> str | None:
            if isinstance(value, dict):
                country = value.get("country") or value.get("details", {}).get("country", {})
                if isinstance(country, dict):
                    names = [country.get("name", ""), *country.get("synonyms", [])]
                    if any(str(name).casefold().strip() == wanted for name in names):
                        return str(country.get("slug", "")) or None
                for child in value.values():
                    match = find_country(child)
                    if match:
                        return match
            elif isinstance(value, list):
                for child in value:
                    match = find_country(child)
                    if match:
                        return match
            return None

        return find_country(index_payload)

    @classmethod
    def parse_page(cls, payload: dict[str, Any], category: str) -> list[LiveDocument]:
        details = payload.get("details", {})
        country = details.get("country", {}).get("name", "")
        parts = details.get("parts", [])
        content_date = payload.get("public_updated_at")
        fetched_at = datetime.now(timezone.utc).isoformat()
        title = str(payload.get("title", "FCDO travel advice"))
        documents = []
        for part in parts:
            slug = str(part.get("slug", "")).casefold()
            body = str(part.get("body", ""))
            if category == "visa_policy" and slug == "entry-requirements":
                text = cls._html_text(body)
                scope = "GB"
                section = slug
            elif category == "vaccination_requirement" and slug == "health":
                text = cls._html_text(body)
                scope = "ALL"
                section = slug
            elif category == "customs_regulation":
                heading_terms = {
                    "entry-requirements": ("customs rules", "bringing", "importing", "customs"),
                    "health": ("medication", "medicine"),
                    "safety-and-security": ("local laws and customs", "customs"),
                    "local-laws-and-customs": ("customs", "medicine"),
                }.get(slug)
                if not heading_terms:
                    continue
                sections = cls._extract_subsections(body, heading_terms)
                for heading, text in sections:
                    documents.append(
                        cls._document(
                            payload, title, country, category, content_date, fetched_at,
                            f"{slug}:{heading}", text, "ALL",
                        )
                    )
                continue
            else:
                continue
            if text:
                documents.append(cls._document(payload, title, country, category, content_date, fetched_at, section, text, scope))
        return documents

    @staticmethod
    def _html_text(body: str) -> str:
        soup = BeautifulSoup(body, "html.parser")
        return re.sub(r"\s+", " ", unescape(soup.get_text(" ", strip=True))).strip()

    @classmethod
    def _extract_subsections(cls, body: str, heading_terms: tuple[str, ...]) -> list[tuple[str, str]]:
        soup = BeautifulSoup(body, "html.parser")
        results = []
        headings = soup.find_all(["h2", "h3"])
        for heading in headings:
            heading_text = heading.get_text(" ", strip=True).casefold()
            if not any(term in heading_text for term in heading_terms):
                continue
            fragments = []
            for sibling in heading.find_next_siblings():
                if sibling.name in {"h2", "h3"}:
                    break
                fragments.append(sibling.get_text(" ", strip=True))
            text = re.sub(r"\s+", " ", " ".join(fragments)).strip()
            if text:
                results.append((heading_text, text))
        return results

    @classmethod
    def _document(cls, payload, title, country, category, content_date, fetched_at, section, text, scope):
        canonical_url = str(payload.get("web_url", ""))
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return LiveDocument(
            url=f"{canonical_url}#{section}",
            title=f"{title}: {section}",
            source_domain="www.gov.uk",
            source_type=cls.source_type,
            nationality_scope=scope,
            country=country,
            category=category,
            content_date=str(content_date)[:10] if content_date else None,
            fetched_at=fetched_at,
            text=text,
            section=section,
            raw_hash=digest,
        )