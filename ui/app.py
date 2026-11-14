"""Streamlit chat client for the shared TrustRAG answer pipeline."""

from __future__ import annotations

import html
import re
from datetime import date
from typing import Any

import streamlit as st
import yaml

from trustrag.offline.embed_index import EmbeddingModelUnavailable
from trustrag.pipeline import TrustRAGPipeline

st.set_page_config(page_title="TrustRAG Travel Advisory", page_icon="🛡", layout="wide")

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
ICON_SHIELD = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3 4.5 6v5.5c0 4.6 3.1 8.2 7.5 9.5 4.4-1.3 7.5-4.9 7.5-9.5V6z"/></svg>'

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    :root {
      --tr-ink: #1d2330; --tr-ink-2: #5b6474; --tr-rule: #e4e7ec; --tr-bg: #f6f7f9; --tr-card: #ffffff;
      --tr-accent: #2f6fdb; --tr-accent-soft: #dfe9fb;
      --tr-good: #2c7a3f; --tr-good-soft: #d9f0de; --tr-warn: #8a5a00; --tr-warn-soft: #f6e3b4;
      --tr-bad: #a33a2b; --tr-bad-soft: #f8dcd6; --tr-muted-soft: #f0f2f5;
    }
    html, body, .stApp, [data-testid="stSidebar"], .stMarkdown, button, input, textarea {
      font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
    }
    .stApp { background: var(--tr-bg); }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { max-width: 980px; padding-top: 2.2rem; }

    /* Sidebar */
    [data-testid="stSidebar"] { background: var(--tr-card); border-right: 1px solid var(--tr-rule); }
    [data-testid="stSidebar"] .tr-brand { display: flex; align-items: center; gap: .5rem; font-size: 1.35rem; font-weight: 700; color: var(--tr-ink); margin: .2rem 0 .9rem; }
    .tr-brand svg { width: 22px; height: 22px; fill: none; stroke: var(--tr-ink); stroke-width: 2; }
    [data-testid="stSidebar"] label p, [data-testid="stSidebar"] .stWidgetLabel p { color: var(--tr-ink-2); font-size: .95rem; font-weight: 500; }
    [data-testid="stSidebar"] div[role="radiogroup"] { gap: .2rem; }
    [data-testid="stSidebar"] div[role="radiogroup"] label { width: 100%; padding: .45rem .7rem; border-radius: 8px; margin: 0; cursor: pointer; }
    [data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child { display: none; }
    [data-testid="stSidebar"] div[role="radiogroup"] label:hover { background: var(--tr-muted-soft); }
    [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) { background: var(--tr-accent-soft); }
    [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p { color: var(--tr-accent); font-weight: 600; }
    [data-testid="stSidebar"] div[role="radiogroup"] label p { color: var(--tr-ink); font-size: .98rem; }
    .tr-side-note { color: var(--tr-ink-2); font-size: .82rem; margin-top: -.4rem; }

    /* Chat */
    .tr-user { display: flex; justify-content: flex-end; margin: .4rem 0 .8rem; }
    .tr-user span { background: var(--tr-card); border: 1px solid var(--tr-rule); border-radius: 18px; padding: .6rem 1.1rem; color: var(--tr-ink); font-size: 1.02rem; max-width: 80%; }
    .tr-card { background: var(--tr-card); border: 1px solid var(--tr-rule); border-radius: 16px; padding: 1.3rem 1.4rem 1.1rem; margin-bottom: 1.4rem; color: var(--tr-ink); }
    .tr-answer { font-size: 1.12rem; line-height: 1.6; margin: 0 0 .9rem; }
    .tr-answer strong { font-weight: 700; }
    .tr-cite { color: var(--tr-ink-2); font-size: .9em; white-space: nowrap; }
    .tr-chips { display: flex; flex-wrap: wrap; gap: .45rem; margin-bottom: .55rem; }
    .tr-chip { display: inline-flex; align-items: center; gap: .35rem; border-radius: 9px; padding: .28rem .7rem; font-size: .9rem; font-weight: 500; line-height: 1.3; }
    .tr-chip svg { width: 16px; height: 16px; fill: none; stroke: currentColor; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }
    .tr-good { background: var(--tr-good-soft); color: var(--tr-good); }
    .tr-warn { background: var(--tr-warn-soft); color: var(--tr-warn); }
    .tr-bad { background: var(--tr-bad-soft); color: var(--tr-bad); }
    .tr-blue { background: var(--tr-accent-soft); color: var(--tr-accent); }
    .tr-plain { background: var(--tr-muted-soft); color: var(--tr-ink-2); }
    .tr-section { color: var(--tr-ink-2); font-size: .95rem; font-weight: 500; margin: 1rem 0 .3rem; }
    table.tr-sources { width: 100%; border-collapse: collapse; font-size: .95rem; }
    table.tr-sources th { text-align: left; color: var(--tr-ink-2); font-weight: 500; padding: .45rem .3rem; border-bottom: 1px solid var(--tr-rule); }
    table.tr-sources td { padding: .55rem .3rem; border-bottom: 1px solid var(--tr-rule); vertical-align: middle; }
    table.tr-sources tr:last-child td { border-bottom: 0; }
    table.tr-sources td.num { font-variant-numeric: tabular-nums; width: 4.5rem; }
    table.tr-sources a { color: var(--tr-ink); text-decoration: none; }
    table.tr-sources a:hover { text-decoration: underline; }
    table.tr-sources tr.uncited td { color: var(--tr-ink-2); }
    .tr-tag { font-size: .72rem; font-weight: 600; letter-spacing: .03em; text-transform: uppercase; color: var(--tr-accent); margin-left: .35rem; }
    .tr-bar { width: 8rem; height: 8px; background: transparent; border-radius: 99px; }
    .tr-bar span { display: block; height: 8px; border-radius: 99px; background: var(--tr-accent); min-width: 12px; }
    .tr-bar span.low { background: #8d96a5; }
    .tr-callout { display: flex; gap: .55rem; align-items: flex-start; border-radius: 10px; padding: .7rem .9rem; margin-top: .9rem; font-size: .95rem; line-height: 1.45; }
    .tr-callout svg { width: 18px; height: 18px; flex: none; margin-top: .12rem; fill: none; stroke: currentColor; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }
    details.tr-breakdown { margin-top: .9rem; color: var(--tr-ink-2); font-size: .93rem; }
    details.tr-breakdown summary { cursor: pointer; line-height: 1.5; }
    details.tr-breakdown table { width: 100%; border-collapse: collapse; margin-top: .5rem; font-size: .85rem; }
    details.tr-breakdown th, details.tr-breakdown td { text-align: left; padding: .3rem .35rem; border-bottom: 1px solid var(--tr-rule); font-variant-numeric: tabular-nums; }
    details.tr-breakdown th { font-weight: 500; }
    .tr-compare { background: var(--tr-card); border: 1px solid var(--tr-rule); border-radius: 12px; padding: .9rem 1rem; height: 100%; color: var(--tr-ink); }
    .tr-compare h4 { font-size: .95rem; margin: 0 0 .5rem; font-weight: 600; }
    .tr-compare p { font-size: .9rem; line-height: 1.5; margin: 0 0 .6rem; }
    .tr-legend { color: var(--tr-ink-2); font-size: .85rem; margin: .2rem 0 1rem; }
    [data-testid="stChatInput"] { border-radius: 14px; }
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
        second.append(f'<span class="tr-chip tr-plain">{esc(category + suffix)}</span>')
    latency = response.get("latency_ms")
    if isinstance(latency, (int, float)):
        second.append(f'<span class="tr-chip tr-plain">Answered in {latency / 1000:.1f} s</span>')
    rows = f'<div class="tr-chips">{"".join(chips)}</div>'
    if second:
        rows += f'<div class="tr-chips">{"".join(second)}</div>'
    return rows


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
        bar_class = "" if trust >= 0.5 else "low"
        rows.append(
            f'<tr class="{"" if source.get("cited", True) else "uncited"}"><td>{link}{origin}</td>'
            f'<td>{esc(SOURCE_TYPES.get(source.get("source_type", "unknown"), source.get("source_type", "")))}</td>'
            f'<td class="num">{trust:.2f}</td>'
            f'<td><div class="tr-bar"><span class="{bar_class}" style="width:{width}%"></span></div></td></tr>'
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


def render_answer(response: dict[str, Any], backend: str, show_trace: bool) -> None:
    card = (
        f'<div class="tr-card"><p class="tr-answer">{answer_html(response.get("answer", ""))}</p>'
        f"{chips_html(response)}{sources_html(response)}{conflicts_html(response)}"
        f"{notices_html(response, backend)}{breakdown_html(response)}</div>"
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
    st.markdown(f'<div class="tr-brand">{ICON_SHIELD}TrustRAG</div>', unsafe_allow_html=True)
    system = st.radio(
        "System",
        list(SYSTEM_LABELS),
        index=2,
        format_func=lambda value: f"{value}  {SYSTEM_LABELS[value]}",
    )
    if system == "S0":
        st.markdown('<p class="tr-side-note">Baseline: no trust ranking or verification.</p>', unsafe_allow_html=True)
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
        f'<p class="tr-side-note">{esc(backend)}{" · " + esc(model_name) if model_name else " · template writer"}</p>',
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

if not st.session_state.messages:
    st.markdown("### Travel advisory")
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
