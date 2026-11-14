"""Streamlit chat client for the shared TrustRAG answer pipeline."""

from __future__ import annotations

import html
import re
from datetime import date
from pathlib import Path
from typing import Any

import streamlit as st
import yaml

from trustrag.offline.embed_index import EmbeddingModelUnavailable
from trustrag.pipeline import TrustRAGPipeline

st.set_page_config(page_title="TrustRAG Travel Advisory", page_icon="🛡", layout="wide")
_css = Path(__file__).resolve().parent.parent / "travel_theme.css"
if _css.exists():
    st.markdown(f"<style>{_css.read_text()}</style>", unsafe_allow_html=True)

SYSTEM_LABELS = {
    "S0": "Vanilla RAG",
    "S1": "Standard Trust RAG (static)",
    "S2": "TrustRAG (policy weights)",
    "S3": "TrustRAG (learned weights)",
}
KNOWLEDGE_BASES = {
    "trustrag/config.benchmark.yaml": "Benchmark corpus (75 docs, 15 countries)",
    "trustrag/config.yaml": "Demo corpus (4 docs)",
}
LLM_BACKENDS = ["mock", "ollama", "ollama_cloud", "openai", "anthropic"]
CATEGORY_NAMES = {
    "visa_policy": "Visa policy",
    "vaccination_requirement": "Health entry",
    "customs_regulation": "Customs",
}
SLOT_NAMES = {
    "max_stay_days": "max stay", "visa_required": "visa needed", "visa_type": "visa type",
    "passport_validity_months": "passport validity", "vaccine_required": "vaccine needed",
    "vaccine_name": "vaccine", "certificate_required": "certificate needed",
    "item_allowed": "item rule", "duty_free_limit": "duty-free allowance",
    "declaration_required": "declaration needed",
}
SOURCE_TYPES = {
    "government": "Government", "embassy": "Embassy", "intl_body": "Intl. body", "airline": "Airline",
    "news": "News", "travel_blog": "Blog", "forum_social": "Forum", "unknown": "Unknown",
}

ICON_CHECK = '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="m8 12.5 2.7 2.7L16 9.8"/></svg>'
ICON_CLOCK = '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>'
ICON_WARN = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3.5 2.5 20h19z"/><path d="M12 10v4.5M12 17.2v.3"/></svg>'
ICON_INFO = '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M12 11v5.5M12 7.8v.3"/></svg>'
ICON_PLANE = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 18h18"/><path d="m4.5 13.5 3 1.2 4.6-2.1-6-4.6 1.8-.8 7.6 3.3 3.7-1.7a1.6 1.6 0 0 1 1.3 2.9l-12.4 5.6-3.6-1.5z"/></svg>'
ICON_COMPASS = '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="m15.5 8.5-2 5-5 2 2-5z"/></svg>'
ISO3 = {
    "India": "IND", "Vietnam": "VNM", "Thailand": "THA", "Singapore": "SGP", "Malaysia": "MYS", "Indonesia": "IDN",
    "Japan": "JPN", "Nepal": "NPL", "Kenya": "KEN", "United Kingdom": "GBR", "United States": "USA",
    "Australia": "AUS", "Canada": "CAN", "China": "CHN", "Sri Lanka": "LKA", "Philippines": "PHL",
}
FACT_SLOTS = {
    "visa_policy": ["max_stay_days", "passport_validity_months", "visa_type", "visa_required"],
    "vaccination_requirement": ["vaccine_required", "certificate_required", "vaccine_name"],
    "customs_regulation": ["duty_free_limit", "item_allowed", "declaration_required"],
}
FACT_LABELS = {
    "max_stay_days": "MAX STAY", "passport_validity_months": "PASSPORT", "visa_type": "VISA TYPE",
    "visa_required": "VISA NEEDED", "vaccine_required": "VACCINE NEEDED", "certificate_required": "CERTIFICATE",
    "vaccine_name": "VACCINE", "duty_free_limit": "DUTY-FREE", "item_allowed": "ITEM RULE",
    "declaration_required": "DECLARE",
}
ICON_SHIELD = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3 4.5 6v5.5c0 4.6 3.1 8.2 7.5 9.5 4.4-1.3 7.5-4.9 7.5-9.5V6z"/></svg>'

st.markdown(
    """
    <style>
    /* Custom answer-card components, styled with the departure-board palette from travel_theme.css. */
    .tr-brand { display: flex; align-items: center; gap: .65rem; margin: .1rem 0 1rem; }
    .tr-brand .tr-logo { width: 40px; height: 40px; border-radius: 9px; background: var(--db-amber); display: grid; place-items: center; flex: none; }
    .tr-brand .tr-logo svg { width: 24px; height: 24px; fill: none; stroke: #121211; stroke-width: 1.8; stroke-linecap: round; stroke-linejoin: round; }
    .tr-brand b { display: block; color: var(--db-text) !important; font-size: 1.2rem; font-weight: 600; line-height: 1.2; }
    .tr-brand small { display: block; font-family: var(--db-mono) !important; color: var(--db-amber) !important; font-size: .72rem !important; letter-spacing: .1em; }
    .tr-side-note { font-family: var(--db-mono); color: var(--db-green) !important; font-size: .78rem; margin-top: -.3rem; }
    .tr-side-note.plain { color: var(--db-dim) !important; font-family: 'Inter', sans-serif; }

    .tr-legend { color: var(--db-muted); font-size: .95rem; margin: -.4rem 0 1.2rem; line-height: 1.55; }
    .tr-user { display: flex; justify-content: flex-end; margin: .4rem 0 .9rem; }
    .tr-user span { background: var(--db-amber); color: #121211; font-weight: 500; border-radius: 16px 16px 4px 16px; padding: .6rem 1.1rem; max-width: 75%; font-size: 1rem; }
    .tr-turn { display: flex; gap: .8rem; align-items: flex-start; margin-bottom: 1.3rem; }
    .tr-avatar { width: 36px; height: 36px; flex: none; border-radius: 50%; background: var(--db-amber-bg); display: grid; place-items: center; margin-top: .2rem; }
    .tr-avatar svg { width: 18px; height: 18px; fill: none; stroke: var(--db-amber); stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }
    .tr-card { flex: 1; min-width: 0; background: var(--db-panel); border: 1px solid var(--db-line); border-radius: 4px 16px 16px 16px; padding: 1.1rem 1.25rem 1rem; color: var(--db-text); animation: db-in .35s ease-out; }
    .tr-facts { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); background: var(--db-board); border: 1px solid var(--db-line); border-radius: 8px; margin-bottom: 1rem; overflow: hidden; }
    .tr-facts div { padding: .6rem .85rem; border-right: 1px solid var(--db-line); }
    .tr-facts div:last-child { border-right: 0; }
    .tr-facts span { display: block; font-family: var(--db-mono); font-size: .68rem; letter-spacing: .08em; color: var(--db-dim); }
    .tr-facts b { display: block; font-family: var(--db-mono); font-size: 1rem; font-weight: 600; letter-spacing: .04em; color: var(--db-amber-soft); text-transform: uppercase; }
    .tr-answer { font-size: 1.05rem; line-height: 1.65; margin: 0 0 .9rem; color: var(--db-text) !important; }
    .tr-answer strong { color: var(--db-amber-soft); font-weight: 600; }
    .tr-cite { color: var(--db-dim); font-size: .88em; white-space: nowrap; }

    .tr-chips { display: flex; flex-wrap: wrap; gap: .45rem; margin-bottom: .5rem; }
    .tr-chip { display: inline-flex; align-items: center; gap: .35rem; border-radius: 4px; border: 1px solid; padding: .22rem .6rem; font-size: 12px; font-weight: 500; line-height: 1.4; }
    .tr-chip svg { width: 14px; height: 14px; fill: none; stroke: currentColor; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }
    .tr-good { background: #0F2E24; color: #5DCAA5; border-color: #1D9E75; }
    .tr-blue { background: #13253A; color: #85B7EB; border-color: #185FA5; }
    .tr-vol, .tr-warn { background: #2A2416; color: #FAC775; border-color: #BA7517; }
    .tr-bad { background: #2E1514; color: #F09492; border-color: #A33231; }
    .tr-plain { background: #262624; color: #B4B2A9; border-color: #3A3A37; }

    .tr-section { font-family: var(--db-mono); text-transform: uppercase; letter-spacing: .08em; font-size: .72rem; color: var(--db-dim); margin: 1.1rem 0 .3rem; }
    table.tr-sources { width: 100%; border-collapse: collapse; font-size: .93rem; }
    table.tr-sources th { text-align: left; font-family: var(--db-mono); text-transform: uppercase; letter-spacing: .06em; font-size: .7rem; font-weight: 400; color: var(--db-dim); padding: .45rem .35rem; border-bottom: 1px solid var(--db-line-2); }
    table.tr-sources td { padding: .55rem .35rem; border-bottom: 1px solid var(--db-line); vertical-align: middle; color: var(--db-text); }
    table.tr-sources tr:hover td { background: #222220; }
    table.tr-sources tr:last-child td { border-bottom: 0; }
    table.tr-sources td.num { font-family: var(--db-mono); font-variant-numeric: tabular-nums; width: 4rem; }
    table.tr-sources a { color: var(--db-text); text-decoration: none; }
    table.tr-sources a:hover { color: var(--db-amber-soft); text-decoration: underline; }
    table.tr-sources tr.uncited td { color: var(--db-muted); }
    .tr-tag { font-family: var(--db-mono); font-size: .66rem; letter-spacing: .06em; text-transform: uppercase; color: var(--db-green); margin-left: .35rem; }
    .tr-bar { width: 7rem; height: 6px; border-radius: 99px; }
    .tr-bar span { display: block; height: 6px; border-radius: 99px; min-width: 10px; }

    .tr-callout { display: flex; gap: .55rem; align-items: flex-start; padding: .7rem .9rem; margin-top: .9rem; font-size: .93rem; line-height: 1.5; border-left: 4px solid; }
    .tr-callout svg { width: 16px; height: 16px; flex: none; margin-top: .2rem; fill: none; stroke: currentColor; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }
    .tr-callout.tr-warn { background: var(--db-amber-bg); color: var(--db-amber-soft); border-color: var(--db-amber); }
    .tr-callout.tr-plain { background: #1F1F1D; color: var(--db-muted); border-color: var(--db-line-2); }
    .tr-callout strong { color: inherit; }

    details.tr-breakdown { margin-top: .9rem; color: var(--db-muted); font-size: .9rem; }
    details.tr-breakdown summary { cursor: pointer; line-height: 1.5; }
    details.tr-breakdown table { width: 100%; border-collapse: collapse; margin-top: .5rem; font-size: .82rem; }
    details.tr-breakdown th, details.tr-breakdown td { text-align: left; padding: .3rem .35rem; border-bottom: 1px solid var(--db-line); font-family: var(--db-mono); font-variant-numeric: tabular-nums; color: var(--db-muted); }
    details.tr-breakdown th { font-weight: 400; color: var(--db-dim); }

    .tr-compare { background: var(--db-panel); border: 1px solid var(--db-line); border-radius: 10px; padding: .9rem 1rem; height: 100%; color: var(--db-text); }
    .tr-compare h4 { font-family: var(--db-mono); font-size: .8rem; letter-spacing: .05em; text-transform: uppercase; color: var(--db-amber-soft); margin: 0 0 .5rem; font-weight: 600; }
    .tr-compare p { font-size: .9rem; line-height: 1.5; margin: 0 0 .6rem; color: var(--db-text) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------- helpers

def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def answer_html(text: str) -> str:
    """Escape model output, then add emphasis for key values and soften citation markers."""
    safe = esc(text)
    safe = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe)
    safe = re.sub(
        r"(?<![\w>])((?:USD|EUR|GBP|INR|SGD|THB|AUD|CAD|JPY|VND)\s?[\d,]+|\d[\d,]*\s(?:days?|months?))(?![\w<])",
        r"<strong>\1</strong>",
        safe,
    )
    safe = re.sub(r"((?:\s*\[E\d+\])+)", lambda m: f' <span class="tr-cite">{m.group(1).strip()}</span>', safe)
    return safe


def status_chip(response: dict[str, Any]) -> str:
    status = response.get("verification_status", "")
    faith = response.get("faithfulness")
    faith_text = f" · faithfulness {faith:.2f}" if isinstance(faith, (int, float)) else ""
    styles = {
        "VERIFIED": ("tr-good", ICON_CHECK, f"Verified{faith_text}"),
        "PARTIAL": ("tr-warn", ICON_WARN, f"Partly verified{faith_text}"),
        "VERIFICATION_REQUIRED": ("tr-bad", ICON_WARN, "Needs checking with official sources"),
        "UNVERIFIED": ("tr-plain", ICON_INFO, "Unverified baseline answer"),
        "CLARIFICATION": ("tr-blue", ICON_INFO, "Needs a detail from you"),
        "OUT_OF_SCOPE": ("tr-plain", ICON_INFO, "Outside visa, health and customs"),
        "SYSTEM_UNAVAILABLE": ("tr-plain", ICON_INFO, "System unavailable"),
    }
    css, icon, label = styles.get(status, ("tr-plain", ICON_INFO, status.title()))
    return f'<span class="tr-chip {css}">{icon}{esc(label)}</span>'


def age_text(days: int | None) -> str | None:
    if days is None:
        return None
    if days < 1:
        return "today"
    if days < 60:
        return f"{days} day{'s' if days != 1 else ''} ago"
    if days < 730:
        return f"{round(days / 30.4)} months ago"
    return f"{days / 365:.1f} years ago"


def chips_html(response: dict[str, Any]) -> str:
    chips = [status_chip(response)]
    if response.get("verification_status") not in {"CLARIFICATION", "OUT_OF_SCOPE", "SYSTEM_UNAVAILABLE"}:
        chips.append(f'<span class="tr-chip tr-blue">Confidence {float(response.get("confidence", 0)):.0%}</span>')
    second = []
    freshness = response.get("freshness") or {}
    days = freshness.get("age_days")
    if days is None and freshness.get("most_recent_source"):
        try:  # the UI asks about today; benchmark runs pass an explicit as-of date instead
            days = (date.today() - date.fromisoformat(str(freshness["most_recent_source"])[:10])).days
        except ValueError:
            days = None
    age = age_text(days)
    if age:
        second.append(f'<span class="tr-chip tr-plain">{ICON_CLOCK}Official source updated {esc(age)}</span>')
    category = CATEGORY_NAMES.get(response.get("category") or "")
    if category:
        volatility = (response.get("volatility") or "").lower()
        suffix = f" · {volatility} volatility" if volatility else ""
        second.append(f'<span class="tr-chip tr-vol">{ICON_WARN}{esc(category + suffix)}</span>')
    latency = response.get("latency_ms")
    if isinstance(latency, (int, float)):
        second.append(f'<span class="tr-chip tr-plain">{ICON_CLOCK}Answered in {latency / 1000:.1f} s</span>')
    rows = f'<div class="tr-chips">{"".join(chips)}</div>'
    if second:
        rows += f'<div class="tr-chips">{"".join(second)}</div>'
    return rows


def trust_colour(trust: float) -> str:
    if trust >= 0.75:
        return "#5DCAA5"
    if trust >= 0.50:
        return "#EF9F27"
    return "#E24B4A"


def sources_html(response: dict[str, Any]) -> str:
    sources = response.get("sources") or []
    if not sources:
        return ""
    rows = []
    for source in sorted(sources, key=lambda item: float(item.get("trust", 0)), reverse=True):
        trust = float(source.get("trust", 0))
        label = f"[{source['evidence_id']}] " if source.get("evidence_id") else ""
        title = esc(label + str(source.get("title") or source.get("url", "Source")))
        url = str(source.get("url") or "")
        link = f'<a href="{esc(url)}" target="_blank" rel="noopener">{title}</a>' if url.startswith("http") else title
        origin = ""
        if source.get("live"):
            origin = '<span class="tr-tag">live</span>'
        elif source.get("search_fallback"):
            origin = '<span class="tr-tag">web</span>'
        width = max(4, min(100, round(trust * 100)))
        colour = trust_colour(trust)
        rows.append(
            f'<tr class="{"" if source.get("cited", True) else "uncited"}"><td>{link}{origin}</td>'
            f'<td>{esc(SOURCE_TYPES.get(source.get("source_type", "unknown"), source.get("source_type", "")))}</td>'
            f'<td class="num" style="color:{colour}">{trust:.2f}</td>'
            f'<td><div class="tr-bar"><span style="width:{width}%;background:{colour}"></span></div></td></tr>'
        )
    heading = "Sources ranked by trust" if response.get("system") != "S0" else "Sources ranked by similarity"
    return (
        f'<div class="tr-section">{heading}</div>'
        '<table class="tr-sources"><thead><tr><th>Source</th><th>Type</th><th>Trust</th><th></th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
    )


def format_value(slot: str, value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return f"{value[1]} {float(value[0]):,.0f}"
    if slot == "max_stay_days":
        return f"{value} days"
    if slot == "passport_validity_months":
        return f"{value} months"
    return str(value).replace("_", " ")


def conflicts_html(response: dict[str, Any]) -> str:
    notes = []
    for conflict in response.get("conflicts") or []:
        slot = conflict.get("slot", "")
        support = {str(key): float(value) for key, value in (conflict.get("support") or {}).items()}
        total = sum(support.values()) or 1.0
        chosen = conflict.get("chosen")
        other = (conflict.get("value") or {}).get("value")
        chosen_share = support.get(str(chosen), 0.0) / total
        other_share = support.get(str(other), 0.0) / total
        notes.append(
            f"<strong>{esc(SLOT_NAMES.get(slot, slot))}</strong>: {esc(format_value(slot, chosen))} "
            f"({chosen_share:.0%} of trust-weighted support) chosen over {esc(format_value(slot, other))} ({other_share:.0%})"
        )
    if not notes:
        return ""
    body = "<br>".join(notes)
    return f'<div class="tr-callout tr-warn">{ICON_WARN}<div>Sources disagree. {body}.</div></div>'


def notices_html(response: dict[str, Any], backend: str) -> str:
    flags = response.get("flags") or []
    notes = []
    fallback = next((flag for flag in flags if flag.startswith("llm_fallback:")), None)
    if fallback:
        notes.append(f"The {esc(backend)} model was unavailable ({esc(fallback.split(':')[-1])}), so this answer was "
                     "written by the built-in template generator from the same verified facts.")
    if "index_stale" in flags:
        notes.append("An official live page differs from the saved copy; the live version was used and queued for reindexing.")
    if any(flag.startswith("live_check_failed") for flag in flags):
        notes.append("One configured official site could not be checked live; saved evidence was used.")
    if response.get("trace", {}).get("live_documents"):
        notes.append(f"Live-checked {int(response['trace']['live_documents'])} official document(s).")
    if response.get("unsupported_claims"):
        notes.append("Some sentences could not be matched to the cited evidence.")
    return "".join(f'<div class="tr-callout tr-plain">{ICON_INFO}<div>{note}</div></div>' for note in notes)


def breakdown_html(response: dict[str, Any]) -> str:
    evidence = response.get("trace", {}).get("trust", {}).get("evidence", []) or []
    if not evidence:
        return ""
    context = response.get("trace", {}).get("context_evidence", []) or []
    top_source = next((s for s in sorted(response.get("sources") or [], key=lambda s: float(s.get("trust", 0)), reverse=True)
                       if s.get("cited")), None)
    top = None
    label = ""
    if top_source:
        ctx = next((c for c in context if c.get("evidence_id") == top_source.get("evidence_id")), None)
        if ctx:
            top = next((e for e in evidence if str(e.get("evidence_id", "")).startswith(str(ctx.get("document_id")))
                        and e.get("slot") == ctx.get("slot")), None)
            label = f" for {top_source.get('evidence_id')}"
    top = top or max(evidence, key=lambda e: float(e.get("Trust") or 0))

    def f(value: Any) -> str:
        return f"{float(value):.2f}" if isinstance(value, (int, float)) else "–"

    summary = (f"Trust breakdown{esc(label)}: authority {f(top.get('A'))} · freshness {f(top.get('F'))} · "
               f"agreement {f(top.get('G'))} · quality {f(top.get('QE'))} · reputation {f(top.get('SR'))} · "
               f"conflict penalty {f(top.get('CP'))}")
    rows = "".join(
        f"<tr><td>{esc(e.get('source_domain', ''))}</td><td>{esc(SLOT_NAMES.get(e.get('slot', ''), e.get('slot', '')))}</td>"
        f"<td>{esc(format_value(e.get('slot', ''), e.get('value_norm')))}</td>"
        f"<td>{f(e.get('A'))}</td><td>{f(e.get('F'))}</td><td>{f(e.get('G'))}</td><td>{f(e.get('QE'))}</td>"
        f"<td>{f(e.get('SR'))}</td><td>{f(e.get('CP'))}</td><td><strong>{f(e.get('Trust'))}</strong></td></tr>"
        for e in sorted(evidence, key=lambda e: float(e.get("Trust") or 0), reverse=True)
    )
    return (
        f'<details class="tr-breakdown"><summary>{summary}</summary>'
        '<table><thead><tr><th>Source</th><th>Fact</th><th>Value</th><th>A</th><th>F</th><th>G</th><th>QE</th>'
        f'<th>SR</th><th>CP</th><th>Trust</th></tr></thead><tbody>{rows}</tbody></table></details>'
    )


def facts_html(response: dict[str, Any]) -> str:
    """Display-only strip built from fields the pipeline already returns (entities, resolved slots)."""
    if response.get("verification_status") in {"CLARIFICATION", "OUT_OF_SCOPE", "SYSTEM_UNAVAILABLE", "VERIFICATION_REQUIRED"}:
        return ""
    trace = response.get("trace") or {}
    entities = (trace.get("policy") or {}).get("entities") or {}
    resolved = {item.get("slot"): item.get("value") for item in trace.get("resolved_slots", []) or []}
    cells = []
    origin, destination = entities.get("nationality"), entities.get("destination")
    if origin and destination:
        cells.append(("ROUTE", f"{ISO3.get(origin, origin)} \u2192 {ISO3.get(destination, destination)}"))
    for slot in FACT_SLOTS.get(response.get("category") or "", []):
        if slot in resolved and resolved[slot] is not None and len(cells) < 3:
            cells.append((FACT_LABELS[slot], format_value(slot, resolved[slot])))
    if len(cells) < 2:
        return ""
    return '<div class="tr-facts">' + "".join(f"<div><span>{esc(k)}</span><b>{esc(v)}</b></div>" for k, v in cells) + "</div>"


def render_answer(response: dict[str, Any], backend: str, show_trace: bool) -> None:
    card = (
        f'<div class="tr-turn"><div class="tr-avatar">{ICON_COMPASS}</div>'
        f'<div class="tr-card">{facts_html(response)}<p class="tr-answer">{answer_html(response.get("answer", ""))}</p>'
        f"{chips_html(response)}{sources_html(response)}{conflicts_html(response)}"
        f"{notices_html(response, backend)}{breakdown_html(response)}</div></div>"
    )
    st.markdown(card, unsafe_allow_html=True)
    if show_trace:
        with st.expander("Pipeline trace"):
            st.json(response.get("trace", {}))


def render_user(text: str) -> None:
    st.markdown(f'<div class="tr-user"><span>{esc(text)}</span></div>', unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_pipeline(path: str) -> TrustRAGPipeline:
    return TrustRAGPipeline(path)


# ----------------------------------------------------------------------------- sidebar

with st.sidebar:
    st.markdown(
        f'<div class="tr-brand"><div class="tr-logo">{ICON_PLANE}</div><div><b>TrustRAG</b><small>TRAVEL ADVISORY</small></div></div>',
        unsafe_allow_html=True,
    )
    system = st.radio(
        "System",
        list(SYSTEM_LABELS),
        index=2,
        format_func=lambda value: f"{value}  {SYSTEM_LABELS[value]}",
    )
    if system == "S0":
        st.markdown('<p class="tr-side-note plain">Baseline: no trust ranking or verification.</p>', unsafe_allow_html=True)
    config_path = st.selectbox(
        "Knowledge base",
        list(KNOWLEDGE_BASES),
        format_func=KNOWLEDGE_BASES.get,
        help="The benchmark corpus covers visa, vaccination and customs rules for 15 destinations.",
    )
    APP_CONFIG = yaml.safe_load(open(config_path, encoding="utf-8")) or {}
    LIVE_AVAILABLE = bool(APP_CONFIG.get("live", {}).get("enabled", False))
    backend = st.selectbox("LLM", LLM_BACKENDS, index=0)
    model_name = (APP_CONFIG.get("llm", {}).get(backend, {}) or {}).get("model")
    st.markdown(
        f'<p class="tr-side-note">\u25cf {esc(backend)}{" · " + esc(model_name) if model_name else " · template writer"}</p>',
        unsafe_allow_html=True,
    )
    show_trace = st.toggle("Show trace", value=False)
    live_check = st.toggle(
        "Live check",
        value=False,
        disabled=not LIVE_AVAILABLE,
        help="Fetch configured official sources for this question (demo only). Benchmark runs never go online.",
    )
    st.caption(f"Freshness rule: {APP_CONFIG.get('trust', {}).get('freshness_mode', 'last_verified')}")
    st.button("Clear chat", on_click=lambda: st.session_state.update(messages=[], compare_results=None))

# ----------------------------------------------------------------------------- chat

if "messages" not in st.session_state:
    st.session_state.messages = []
if "compare_results" not in st.session_state:
    st.session_state.compare_results = None

st.markdown("# Travel advisory")
st.markdown(
    '<p class="tr-legend">Ask about visa, health-entry or customs rules. <strong>Verified</strong> means every '
    "sentence is backed by the cited evidence; <strong>confidence</strong> is how sure the system is about the value "
    "itself, and drops when sources disagree.</p>",
    unsafe_allow_html=True,
)

for message in st.session_state.messages:
    if message["role"] == "user":
        render_user(message["content"])
    else:
        render_answer(message["response"], message.get("backend", backend), show_trace)

question = st.chat_input("Ask about visa, health-entry, or customs rules")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    render_user(question)
    spinner = "Checking sources..." if backend == "mock" else f"Checking sources and writing the answer with {backend}..."
    with st.spinner(spinner):
        try:
            response = get_pipeline(config_path).answer(question, system=system, llm=backend, live_check=live_check)
        except EmbeddingModelUnavailable as error:
            st.error(str(error))
            st.stop()
    render_answer(response, backend, show_trace)
    st.session_state.messages.append({"role": "assistant", "response": response, "backend": backend})

if st.session_state.messages:
    if backend != "mock":
        st.caption(f"Compare runs the question through all four systems, so it takes about four times as long with {backend}.")
    if st.button("Compare RAG systems"):
        last_query = next(m["content"] for m in reversed(st.session_state.messages) if m["role"] == "user")
        results = []
        with st.spinner("Running the same question through each system..."):
            for selected in SYSTEM_LABELS:
                try:
                    result = get_pipeline(config_path).answer(last_query, system=selected, llm=backend, live_check=live_check)
                except Exception as error:  # show the failure in that column instead of breaking the page
                    result = {"answer": f"System could not run: {error}", "verification_status": "SYSTEM_UNAVAILABLE",
                              "confidence": 0.0, "sources": []}
                results.append((selected, result))
        st.session_state.compare_results = results

if st.session_state.compare_results:
    st.markdown("#### System comparison")
    for column, (selected, result) in zip(st.columns(4), st.session_state.compare_results):
        with column:
            confidence = (f'<span class="tr-chip tr-blue">Confidence {float(result.get("confidence", 0)):.0%}</span>'
                          if result.get("verification_status") not in {"CLARIFICATION", "OUT_OF_SCOPE", "SYSTEM_UNAVAILABLE"} else "")
            st.markdown(
                f'<div class="tr-compare"><h4>{esc(selected)} · {esc(SYSTEM_LABELS[selected])}</h4>'
                f'<p>{answer_html(result.get("answer", ""))}</p>'
                f'<div class="tr-chips">{status_chip(result)}{confidence}</div></div>',
                unsafe_allow_html=True,
            )
