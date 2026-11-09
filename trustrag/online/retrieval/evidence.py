"""Evidence objects and normalization helpers for extracted claims."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class EvidenceObject:
    evidence_id: str
    claim: str
    slot: str
    value_raw: str
    value_norm: Any
    document_id: str
    source_domain: str
    source_type: str
    is_primary: bool
    content_date: str | None
    last_verified_ts: str | None
    crawl_timestamp: str | None
    extraction_confidence: float
    hedged: bool
    chunk_text: str
    relevance: float
    destination: str
    title: str = ""
    url: str = ""
    nationality_scope: str = "UNKNOWN"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_days(text: str) -> int | None:
    match = re.search(r"\b(\d{1,3})\s+days?\b", text, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"\b(one|1|three|3|six|6)\s+months?\b", text, flags=re.IGNORECASE)
    if match:
        values = {"one": 30, "1": 30, "three": 90, "3": 90, "six": 180, "6": 180}
        return values[match.group(1).lower()]
    return None


def normalize_bool(text: str) -> bool | None:
    value = text.lower()
    if re.search(r"\b(not required|not generally required|no visa required|visa-free|exempt)\b", value):
        return False
    if re.search(r"\b(required|mandatory|must obtain|must have)\b", value):
        return True
    return None


def normalize_enum(text: str) -> str | None:
    value = text.lower()
    if re.search(r"\b(e[- ]?visa|electronic visa)\b", value):
        return "evisa"
    if re.search(r"\bvisa[- ]free|visa exemption\b", value):
        return "visa_free"
    if re.search(r"\bvisa on arrival\b", value):
        return "visa_on_arrival"
    if re.search(r"\bembassy visa\b", value):
        return "embassy_visa"
    return None


def normalize_money(text: str) -> tuple[float, str] | None:
    match = re.search(
        r"(?:(USD|EUR|GBP|INR|SGD|THB|AUD|CAD|JPY|VND)\s*|([$€£₹])\s*)"
        r"([\d,]+(?:\.\d{1,2})?)\s*(USD|EUR|GBP|INR|SGD|THB|AUD|CAD|JPY|VND)?",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    code = (match.group(1) or match.group(4) or "").upper()
    symbol_currency = {"$": "USD", "€": "EUR", "£": "GBP", "₹": "INR"}
    code = code or symbol_currency.get(match.group(2), "")
    if not code:
        return None
    return float(match.group(3).replace(",", "")), code


def scope_matches(slot: str, scope: str | list[str] | None, nationality: str | None) -> bool:
    dependent_slots = {"visa_required", "visa_type", "max_stay_days"}
    if slot not in dependent_slots:
        return True
    if scope is None:
        return True
    allowed = scope if isinstance(scope, list) else str(scope).replace("[", "").replace("]", "").replace("'", "").replace('"', "").split(",")
    allowed = {str(value).strip().upper() for value in allowed}
    if "ALL" in allowed or "UNKNOWN" in allowed:
        return True
    nationality_codes = {
        "INDIA": "IN", "UNITED KINGDOM": "GB", "BRITISH": "GB",
        "UNITED STATES": "US", "AMERICAN": "US", "CANADA": "CA",
        "AUSTRALIA": "AU", "SINGAPORE": "SG", "THAILAND": "TH",
        "MALAYSIA": "MY", "INDONESIA": "ID", "JAPAN": "JP", "CHINA": "CN",
        "NEPAL": "NP", "KENYA": "KE",
    }
    code = nationality_codes.get(str(nationality or "").upper(), str(nationality or "").upper())
    return code in allowed


def extract_claims(
    *,
    chunk: dict[str, Any],
    document: dict[str, Any],
    slots: list[str],
    destination: str,
    item: str | None = None,
) -> list[dict[str, Any]]:
    text = str(chunk.get("text", chunk.get("chunk_text", "")))
    lowered = text.lower()
    values: list[tuple[str, Any, str]] = []
    if "max_stay_days" in slots:
        days = normalize_days(text)
        if days is not None and re.search(r"\b(stay|duration|days?)\b", lowered):
            values.append(("max_stay_days", days, str(days)))
    if "visa_required" in slots:
        value = normalize_bool(text)
        if value is not None and "visa" in lowered:
            values.append(("visa_required", value, str(value).lower()))
    if "visa_type" in slots:
        enum_value = normalize_enum(text)
        if enum_value:
            values.append(("visa_type", enum_value, enum_value))
    if "passport_validity_months" in slots:
        months = re.search(r"passport.{0,50}?\b(\d{1,2})\s+months?\b", text, flags=re.IGNORECASE)
        if months:
            values.append(("passport_validity_months", int(months.group(1)), months.group(0)))
    if "vaccine_required" in slots:
        value = normalize_bool(text)
        if value is not None and re.search(r"vaccine|vaccination|health proof|certificate", lowered):
            values.append(("vaccine_required", value, str(value).lower()))
    if "vaccine_name" in slots:
        for vaccine in ("yellow fever", "polio", "covid-19", "covid"):
            if vaccine in lowered:
                values.append(("vaccine_name", vaccine, vaccine))
                break
    if "certificate_required" in slots and re.search(r"certificate|proof", lowered):
        value = normalize_bool(text)
        if value is not None:
            values.append(("certificate_required", value, str(value).lower()))
    item_present = False
    if item:
        item_terms = {item.lower()}
        if item.lower() in {"medicine", "medication"}:
            item_terms.update({"medicine", "medicines", "medication", "medications"})
        item_present = any(term in lowered for term in item_terms)
    if "item_allowed" in slots and item and item_present:
        if re.search(r"prohibited|not allowed|banned|illegal to bring", lowered):
            values.append(("item_allowed", "prohibited", item))
        elif re.search(
            r"restricted|restrictions|declare|need approval|approval.{0,20}needed|requires? approval|controlled goods",
            lowered,
        ):
            values.append(("item_allowed", "restricted", item))
        elif re.search(r"allowed|may bring|can carry", lowered):
            values.append(("item_allowed", "allowed", item))
    if "declaration_required" in slots:
        value = normalize_bool(text)
        if value is not None and re.search(r"declare|declaration", lowered):
            values.append(("declaration_required", value, str(value).lower()))
    if "duty_free_limit" in slots:
        money = normalize_money(text)
        if money is not None:
            values.append(("duty_free_limit", money, str(money[0])))

    hedged_terms = ("may", "might", "usually", "subject to", "at the discretion", "depends", "typically", "generally")
    output = []
    for index, (slot, normalized, raw) in enumerate(values):
        claim = text.strip()
        output.append(
            EvidenceObject(
                evidence_id=f"{chunk['chunk_id']}:{slot}:{index}",
                claim=claim,
                slot=slot,
                value_raw=raw,
                value_norm=normalized,
                document_id=str(document["document_id"]),
                source_domain=str(document.get("source_domain", "unknown")),
                source_type=str(document.get("source_type", "unknown")),
                is_primary=bool(document.get("is_primary")),
                content_date=document.get("content_date"),
                last_verified_ts=document.get("last_verified_ts"),
                crawl_timestamp=document.get("crawl_timestamp"),
                extraction_confidence=1.0,
                hedged=any(term in lowered for term in hedged_terms),
                chunk_text=text,
                relevance=float(chunk.get("score", 0.0)),
                destination=destination,
                title=str(document.get("title", "")),
                url=str(document.get("url", "")),
                nationality_scope=str(document.get("nationality_scope", "UNKNOWN")),
            ).as_dict()
        )
    return output