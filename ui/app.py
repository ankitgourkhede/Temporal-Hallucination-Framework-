"""Streamlit chatbot for TrustRAG (build prompt "Interfaces" section, Phase 4 spec §7).

Layout: a sidebar (system, LLM, live check, trace toggle, clear) and a chat. Every assistant answer is rendered as
the trust answer card: answer with [E#] citations, coloured status/confidence/freshness/category badges, sources
ranked by trust with trust bars (weak sources greyed), a conflict panel, an expandable per-source trust breakdown
and an optional pipeline trace. "Compare systems" runs the last question through S0-S3 side by side.
"""

from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from components import (  # noqa: E402
    CARD_CSS,
    answer_html,
    badges_html,
    conflicts_html,
    sources_html,
    trust_breakdown_rows,
)
from trustrag.pipeline import TrustRAGPipeline  # noqa: E402

CONFIG_PATH = "trustrag/config.yaml"
SYSTEM_LABELS = {
    "S0": "S0 Baseline RAG",
    "S1": "S1 Static trust",
    "S2": "S2 Policy trust",
    "S3": "S3 Learned trust",
}

st.set_page_config(page_title="TrustRAG Travel Advisory", page_icon="🛡", layout="wide")
st.markdown(CARD_CSS, unsafe_allow_html=True)
APP_CONFIG = yaml.safe_load(Path(CONFIG_PATH).read_text(encoding="utf-8")) or {}
LIVE_AVAILABLE = bool(APP_CONFIG.get("live", {}).get("enabled", False))

with st.sidebar:
    st.markdown("### 🛡 TrustRAG")
    system = st.radio(
        "System",
        list(SYSTEM_LABELS),
        index=2,
        format_func=SYSTEM_LABELS.get,
        help="S2 is the proposed policy-driven trust system. S0 is plain RAG with no trust layer.",
    )
    if system == "S0":
        st.caption("Baseline: no trust ranking and no verification. Use it to compare against S2.")
    backend = st.selectbox("LLM", ["mock", "ollama", "openai", "anthropic"], index=0)
    live_check = st.toggle(
        "Live check (demo)",
        value=False,
        disabled=not LIVE_AVAILABLE,
        help="Fetch configured official sources live. Benchmark runs always keep this off.",
    )
    show_trace = st.toggle("Show trace", value=False)
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.compare_results = None

st.session_state.setdefault("messages", [])
st.session_state.setdefault("compare_results", None)


def render_response(response: dict, *, show_pipeline_trace: bool, key: str) -> None:
    """Render one assistant answer as the trust answer card."""
    st.markdown(answer_html(response), unsafe_allow_html=True)
    st.markdown(badges_html(response), unsafe_allow_html=True)

    flags = response.get("flags") or []
    if "nationality_defaulted" in flags:
        st.caption("No nationality given, so an Indian passport was assumed.")
    if "index_stale" in flags:
        st.info("The official page has changed since our last snapshot. The live version was used.")
    if any(str(flag).startswith("live_check_failed") for flag in flags):
        st.caption("Couldn't check the official site live; the answer uses the saved snapshot.")

    trust_scored = response.get("system") != "S0"
    st.markdown(sources_html(response.get("sources") or [], trust_scored=trust_scored), unsafe_allow_html=True)
    st.markdown(conflicts_html(response.get("conflicts") or []), unsafe_allow_html=True)

    breakdown = trust_breakdown_rows(response)
    if breakdown:
        with st.expander("Trust breakdown"):
            st.dataframe(breakdown, hide_index=True, use_container_width=True)
            features = ["A authority", "F freshness", "G agreement", "QE quality", "SR reputation", "CP conflict"]
            chart = {row["Source"]: [row[name] for name in features] for row in breakdown[:4]}
            if chart:
                st.bar_chart({"feature": features, **chart}, x="feature", stack=False, height=220)
            weights = next(
                (item.get("weights") for item in ((response.get("trace") or {}).get("trust") or {}).get("evidence", [])
                 if item.get("weights")),
                None,
            )
            if weights:
                st.caption(
                    "Weights used ("
                    + str((response.get("trace") or {}).get("weights_source", ""))
                    + "): "
                    + " · ".join(f"{name} {value:.2f}" for name, value in weights.items())
                )
    if show_pipeline_trace:
        with st.expander("Pipeline trace"):
            st.json(response.get("trace") or {}, expanded=False)


for index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message.get("response"):
            render_response(message["response"], show_pipeline_trace=show_trace, key=f"history-{index}")
        else:
            st.markdown(message["content"])

question = st.chat_input("Ask about visa, health-entry, or customs rules")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Checking sources…"):
            response = TrustRAGPipeline(CONFIG_PATH).answer(
                question, system=system, llm=backend, live_check=live_check
            )
        render_response(response, show_pipeline_trace=show_trace, key="latest")
    st.session_state.messages.append({"role": "assistant", "content": response["answer"], "response": response})
    st.session_state.compare_results = None

if st.session_state.messages and st.button("Compare systems (S0–S3)"):
    last_query = next(m["content"] for m in reversed(st.session_state.messages) if m["role"] == "user")
    results = []
    with st.spinner("Running the same question through each system…"):
        for selected in SYSTEM_LABELS:
            try:
                result = TrustRAGPipeline(CONFIG_PATH).answer(
                    last_query, system=selected, llm=backend, live_check=live_check
                )
            except Exception as error:  # show the failure in its column instead of breaking the page
                result = {"answer": f"System could not run: {error}", "verification_status": "SYSTEM_UNAVAILABLE",
                          "confidence": 0.0, "sources": [], "system": selected}
            results.append((selected, result))
    st.session_state.compare_results = results

if st.session_state.compare_results:
    st.subheader("System comparison")
    for column, (selected, result) in zip(st.columns(len(st.session_state.compare_results)), st.session_state.compare_results):
        with column:
            st.markdown(f"**{SYSTEM_LABELS[selected]}**")
            st.markdown(answer_html(result), unsafe_allow_html=True)
            st.markdown(badges_html(result), unsafe_allow_html=True)
            st.markdown(
                sources_html(result.get("sources") or [], trust_scored=selected != "S0"),
                unsafe_allow_html=True,
            )
