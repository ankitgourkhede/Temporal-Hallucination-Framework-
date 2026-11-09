"""HTML building blocks for the TrustRAG answer card (build prompt "Interfaces" + Phase 4 spec §7).

Kept free of Streamlit imports so every piece can be unit-tested. ``ui/app.py`` renders these strings with
``st.markdown(..., unsafe_allow_html=True)``. Every value coming from documents or the LLM is HTML-escaped.
"""

from __future__ import annotations

import html
import re
from typing import Any

CARD_CSS = """
<style>
.tr-answer { font-size: 1.02rem; line-height: 1.65; margin: 0 0 .6rem; }
.tr-answer mark { background: #fde9b8; color: #6b4a00; padding: 0 2px; border-radius: 3px; }
.tr-badges { display: flex; flex-wrap: wrap; gap: 6px; margin: .2rem 0 .9rem; }
.tr-badge { font-size: .80rem; padding: 3px 10px; border-radius: 6px; white-space: nowrap; }
.tr-ok { background: #e3f4e6; color: #1f6b33; }
.tr-warn { background: #fdf0d5; color: #7a5200; }
.tr-bad { background: #fbe3e3; color: #8f2222; }
.tr-info { background: #e5effb; color: #154a85; }
.tr-muted { background: #eef0f2; color: #4a5057; }
.tr-label { font-size: .80rem; color: #5f6670; margin: .2rem 0 .3rem; }
.tr-sources { width: 100%; border-collapse: collapse; font-size: .85rem; }
.tr-sources th { text-align: left; font-weight: 500; color: #7a828b; padding: 4px 6px; border-bottom: 1px solid #e3e6ea; }
.tr-sources td { padding: 6px; border-bottom: 1px solid #eef0f2; vertical-align: middle; }
.tr-sources tr.tr-weak td { color: #9aa1a8; }
.tr-sources a { color: inherit; text-decoration: underline dotted; }
.tr-bar { height: 7px; border-radius: 4px; background: #2f6fd0; }
.tr-weak .tr-bar { background: #b9bec4; }
.tr-bar-track { width: 100%; min-width: 70px; background: #eef0f2; border-radius: 4px; }
.tr-conflict { background: #fdf0d5; color: #6b4a00; border-radius: 8px; padding: 8px 12px; font-size: .86rem; margin-top: .8rem; }
.tr-tag { font-size: .72rem; color: #5f6670; background: #eef0f2; border-radius: 4px; padding: 1px 6px; margin-left: 4px; }
</style>
"""

STATUS_STYLE = {
    "VERIFIED": ("tr-ok", "Verified"),
    "PARTIAL": ("tr-warn", "Partially verified"),
    "VERIFICATION_REQUIRED": ("tr-bad", "Verification required"),
    "UNVERIFIED": ("tr-muted", "Not verified (baseline)"),
    "CLARIFICATION": ("tr-info", "Needs clarification"),
    "OUT_OF_SCOPE": ("tr-muted", "Out of scope"),
    "SYSTEM_UNAVAILABLE": ("tr-bad", "System unavailable"),
}
SOURCE_TYPE_LABELS = {
    "government": "Government", "embassy": "Embassy", "intl_body": "Intl. body",
    "foreign_gov_advisory": "Foreign govt advisory", "commercial_aggregator": "Aggregator",
    "airline": "Airline", "news": "News", "travel_blog": "Blog", "forum_social": "Forum", "unknown": "Unknown",
}
SLOT_LABELS = {
    "max_stay_days": "max stay", "visa_required": "visa required", "visa_type": "visa type",
    "passport_validity_months": "passport validity", "vaccine_required": "vaccine required",
    "vaccine_name": "vaccine", "certificate_required": "certificate required", "item_allowed": "item allowed",
    "duty_free_limit": "duty-free limit", "declaration_required": "declaration required",
}
WEAK_TRUST = 0.30


def _e(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _badge(css: str, text: str) -> str:
    return f'<span class="tr-badge {css}">{_e(text)}</span>'


def _format_value(slot: str, value: Any) -> str:
    if isinstance(value, dict) and "value" in value:
        value = value["value"]
    if slot == "max_stay_days" and isinstance(value, (int, float)):
        return f"{int(value)} days"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return f"{value[0]} {value[1]}"
    return str(value).replace("_", " ")


def answer_html(response: dict[str, Any]) -> str:
    """Answer text; for PARTIAL answers the unsupported/contradicted claims are highlighted."""
    text = _e(response.get("answer", ""))
    for claim in response.get("unsupported_claims") or []:
        claim_text = _e(claim.get("text", "")).strip()
        if claim_text and claim_text in text:
            label = _e(claim.get("label", "UNSUPPORTED").lower())
            text = text.replace(claim_text, f'<mark title="{label}">{claim_text}</mark>', 1)
    text = re.sub(r"\[(E\d+)\]", r'<sup>[\1]</sup>', text)
    return f'<div class="tr-answer">{text}</div>'


def badges_html(response: dict[str, Any]) -> str:
    status = str(response.get("verification_status", ""))
    css, label = STATUS_STYLE.get(status, ("tr-muted", status.replace("_", " ").title()))
    faithfulness = response.get("faithfulness")
    if faithfulness is not None and status in {"VERIFIED", "PARTIAL", "VERIFICATION_REQUIRED"}:
        label = f"{label} · faithfulness {float(faithfulness):.2f}"
    parts = [_badge(css, label)]
    if status not in {"CLARIFICATION", "OUT_OF_SCOPE", "SYSTEM_UNAVAILABLE"}:
        parts.append(_badge("tr-info", f"Confidence {float(response.get('confidence') or 0):.0%}"))
    freshness = response.get("freshness") or {}
    age = freshness.get("age_days")
    if age is not None:
        parts.append(_badge("tr-muted", "Updated today" if age == 0 else f"Updated {age} day{'s' if age != 1 else ''} ago"))
    if response.get("category"):
        category = str(response["category"]).replace("_", " ").capitalize()
        volatility = (response.get("volatility") or "").lower()
        parts.append(_badge("tr-muted", f"{category} · {volatility} volatility" if volatility else category))
    if response.get("agreement") and response.get("sources") and status != "UNVERIFIED":
        parts.append(_badge("tr-muted", f"Agreement {str(response['agreement']).lower()}"))
    if response.get("evidence_strategy") and response.get("system") != "S0":
        parts.append(_badge("tr-muted", str(response["evidence_strategy"]).replace("_", " ").lower()))
    if (response.get("trace") or {}).get("live_documents"):
        parts.append(_badge("tr-info", "Live-checked"))
    return f'<div class="tr-badges">{"".join(parts)}</div>'


def sources_html(sources: list[dict[str, Any]], *, trust_scored: bool = True) -> str:
    """Sources ranked by trust, with a trust bar; weak sources are greyed out."""
    if not sources:
        return ""
    header_metric = "Trust" if trust_scored else "Relevance"
    rows = []
    for source in sources:
        score = max(0.0, min(1.0, float(source.get("trust") or 0.0)))
        weak = trust_scored and source.get("trust_scored", True) and (
            score < WEAK_TRUST or source.get("in_context") is False
        )
        title = _e(source.get("title") or source.get("url") or "Source")
        url = source.get("url")
        title_html = f'<a href="{_e(url)}" target="_blank">{title}</a>' if url else title
        tags = []
        if source.get("live"):
            tags.append("live")
        if source.get("search_fallback"):
            tags.append("search snippet")
        if source.get("in_context") is False:
            tags.append("not used: low trust")
        scope = source.get("nationality_scope")
        if scope and scope not in {"ALL", "UNKNOWN"}:
            tags.append(f"advice for {scope} nationals")
        tag_html = "".join(f'<span class="tr-tag">{_e(tag)}</span>' for tag in tags)
        evidence_id = _e(source.get("evidence_id") or "")
        source_type = SOURCE_TYPE_LABELS.get(str(source.get("source_type")), _e(source.get("source_type")))
        score_text = f"{score:.2f}" if source.get("trust_scored", True) else "n/a"
        rows.append(
            f'<tr class="{"tr-weak" if weak else ""}">'
            f"<td>{'[' + evidence_id + '] ' if evidence_id else ''}{title_html}{tag_html}</td>"
            f"<td>{_e(source_type)}</td><td>{score_text}</td>"
            f'<td><div class="tr-bar-track"><div class="tr-bar" style="width:{score * 100:.0f}%"></div></div></td>'
            "</tr>"
        )
    return (
        f'<div class="tr-label">Sources ranked by {header_metric.lower()}</div>'
        f'<table class="tr-sources"><tr><th>Source</th><th>Type</th><th>{header_metric}</th><th></th></tr>'
        + "".join(rows) + "</table>"
    )


def conflicts_html(conflicts: list[dict[str, Any]]) -> str:
    """One plain-language line per conflicting slot: chosen value vs the runner-up, with trust support."""
    lines = []
    for conflict in conflicts or []:
        slot = str(conflict.get("slot", ""))
        support = conflict.get("support") or {}
        chosen = conflict.get("chosen")
        rival = conflict.get("value") or {}
        rival_value = rival.get("value") if isinstance(rival, dict) else rival
        chosen_support = support.get(str(chosen))
        rival_support = rival.get("support") if isinstance(rival, dict) else support.get(str(rival_value))
        chosen_text = _format_value(slot, chosen)
        rival_text = _format_value(slot, rival_value)
        support_text = ""
        if chosen_support is not None and rival_support is not None:
            support_text = f" (trust-weighted support {float(chosen_support):.2f} vs {float(rival_support):.2f})"
        lines.append(
            f"⚠ Conflict on {_e(SLOT_LABELS.get(slot, slot.replace('_', ' ')))}: "
            f"<b>{_e(chosen_text)}</b> vs {_e(rival_text)}{_e(support_text)}. "
            f"The better-supported value was chosen."
        )
    if not lines:
        return ""
    return '<div class="tr-conflict">' + "<br>".join(lines) + "</div>"


def trust_breakdown_rows(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-source feature table for the 'Trust breakdown' expander (strongest evidence per source)."""
    evidence = ((response.get("trace") or {}).get("trust") or {}).get("evidence") or []
    best: dict[str, dict[str, Any]] = {}
    for item in evidence:
        key = str(item.get("source_domain"))
        if key not in best or float(item.get("Trust") or 0) > float(best[key].get("Trust") or 0):
            best[key] = item
    rows = []
    for item in sorted(best.values(), key=lambda value: float(value.get("Trust") or 0), reverse=True):
        rows.append({
            "Source": item.get("source_domain"),
            "Slot": item.get("slot"),
            "Value": _format_value(str(item.get("slot")), item.get("value_norm")),
            "A authority": round(float(item.get("A") or 0), 3),
            "F freshness": round(float(item.get("F") or 0), 3),
            "G agreement": round(float(item.get("G") or 0), 3),
            "QE quality": round(float(sum((item.get("qe_components") or {}).values()) / 4 if item.get("qe_components") else item.get("QE") or 0), 3),
            "SR reputation": round(float(item.get("SR") or 0), 3),
            "CP conflict": round(float(item.get("CP") or 0), 3),
            "Trust": round(float(item.get("Trust") or 0), 3),
        })
    return rows
