"""Streamlit chat client for the shared TrustRAG answer pipeline."""

from __future__ import annotations

import streamlit as st
import yaml

from trustrag.pipeline import TrustRAGPipeline


st.set_page_config(page_title="TrustRAG Travel Advisory", page_icon="✈", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background: #f3f5f7; }
    [data-testid="stSidebar"] { background: #eef1f4; }
    [data-testid="stChatMessage"] { background: #ffffff; border: 1px solid #dfe4e8; border-radius: 10px; padding: 1rem; }
    [data-testid="stChatInput"] { background: #ffffff; }
    h1 { font-size: 1.8rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Travel Advisory")
APP_CONFIG = yaml.safe_load(open("trustrag/config.yaml", encoding="utf-8")) or {}
LIVE_AVAILABLE = bool(APP_CONFIG.get("live", {}).get("enabled", False))
SYSTEM_LABELS = {
    "S0": "Vanilla RAG",
    "S1": "Standard Trust RAG (static)",
    "S2": "TrustRAG (policy weights)",
    "S3": "TrustRAG (learned weights)",
}

with st.sidebar:
    system = st.selectbox(
        "System",
        list(SYSTEM_LABELS),
        index=2,
        format_func=lambda value: f"{SYSTEM_LABELS[value]} ({value})",
    )
    if system == "S0":
        st.caption("S0 is the unverified baseline; it may surface conflicting sources. Choose S2 for trust ranking and verification.")
    backend = st.selectbox("LLM", ["mock", "ollama", "openai", "anthropic"], index=0)
    show_trace = st.toggle("Show trace", value=False)
    st.button("Clear chat", on_click=lambda: st.session_state.update(messages=[]), icon="🗑")
    live_check = st.toggle(
        "Live check (demo)",
        value=LIVE_AVAILABLE,
        disabled=not LIVE_AVAILABLE,
        help="Fetch configured official sources for this demo query. Benchmark runs always disable live access.",
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("response"):
            response = message["response"]
            st.caption(
                f"{response['verification_status']} · {response['confidence']:.0%} confidence · "
                f"{response.get('category') or 'out of scope'} · {response.get('volatility') or 'n/a'}"
            )
            if response["sources"]:
                st.dataframe(
                    [
                        {"Source": source["title"], "Type": source["source_type"], "Origin": source.get("origin", "Offline index"), "Trust": source["trust"], "Cited": source["cited"], "URL": source["url"]}
                        for source in response["sources"]
                    ],
                    hide_index=True,
                    use_container_width=True,
                )
            if response["conflicts"]:
                with st.expander("Conflicting evidence"):
                    st.json(response["conflicts"])
            trust_trace = response.get("trace", {}).get("trust", {}).get("evidence", [])
            if trust_trace:
                with st.expander("Trust breakdown"):
                    st.dataframe(
                        [
                            {"Evidence": item.get("evidence_id"), "Authority A": item.get("A"), "Freshness F": item.get("F"),
                             "Agreement G": item.get("G"), "Quality QE": item.get("QE"), "Reputation SR": item.get("SR"),
                             "Conflict CP": item.get("CP"), "Trust": item.get("Trust")}
                            for item in trust_trace
                        ],
                        hide_index=True,
                        use_container_width=True,
                    )
            if show_trace:
                with st.expander("Pipeline trace"):
                    st.json(response["trace"])

question = st.chat_input("Ask about visa, health-entry, or customs rules")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        response = TrustRAGPipeline("trustrag/config.yaml").answer(
            question, system=system, llm=backend, live_check=live_check
        )
        st.markdown(response["answer"])
        st.caption(
            f"{response['verification_status']} · {response['confidence']:.0%} confidence · "
            f"{response.get('category') or 'out of scope'} · {response.get('volatility') or 'n/a'}"
        )
        if "index_stale" in response["flags"]:
            st.info("An official live page differs from the saved snapshot; the live evidence was used and queued for reindexing.")
        if any(flag.startswith("live_check_failed") for flag in response["flags"]):
            st.caption("Couldn't check one configured official site live; the answer uses available saved evidence.")
        if response["trace"].get("live_documents"):
            st.caption(f"Live-checked {response['trace'].get('live_documents')} official document(s).")
        if response["sources"]:
            st.dataframe(
                [
                    {"Source": source["title"], "Type": source["source_type"], "Origin": source.get("origin", "Offline index"), "Trust": source["trust"], "Cited": source["cited"], "URL": source["url"]}
                    for source in response["sources"]
                ],
                hide_index=True,
                use_container_width=True,
            )
        if response["unsupported_claims"]:
            st.warning("Some claims could not be verified.")
            st.json(response["unsupported_claims"])
        if response["conflicts"]:
            with st.expander("Conflicting evidence"):
                st.json(response["conflicts"])
        trust_trace = response.get("trace", {}).get("trust", {}).get("evidence", [])
        if trust_trace:
            with st.expander("Trust breakdown"):
                st.dataframe(
                    [
                        {"Evidence": item.get("evidence_id"), "Authority A": item.get("A"), "Freshness F": item.get("F"),
                         "Agreement G": item.get("G"), "Quality QE": item.get("QE"), "Reputation SR": item.get("SR"),
                         "Conflict CP": item.get("CP"), "Trust": item.get("Trust")}
                        for item in trust_trace
                    ],
                    hide_index=True,
                    use_container_width=True,
                )
        if show_trace:
            with st.expander("Pipeline trace"):
                st.json(response["trace"])
    st.session_state.messages.append({"role": "assistant", "content": response["answer"], "response": response})

if "compare_results" not in st.session_state:
    st.session_state.compare_results = None

if st.button("Compare RAG systems", disabled=not st.session_state.messages):
    last_query = next(message["content"] for message in reversed(st.session_state.messages) if message["role"] == "user")
    results = []
    with st.spinner("Running the same question through each available system..."):
        for selected_system in SYSTEM_LABELS:
            try:
                result = TrustRAGPipeline("trustrag/config.yaml").answer(
                    last_query, system=selected_system, llm=backend, live_check=live_check
                )
            except Exception as error:
                result = {
                    "answer": f"System could not run: {error}",
                    "verification_status": "SYSTEM_UNAVAILABLE",
                    "confidence": 0.0,
                    "sources": [],
                }
            results.append({"system": selected_system, "label": SYSTEM_LABELS[selected_system], "response": result})
    st.session_state.compare_results = results

if st.session_state.compare_results:
    st.subheader("System comparison")
    columns = st.columns(4)
    for column, result_item in zip(columns, st.session_state.compare_results):
        with column:
            result = result_item["response"]
            st.markdown(f"**{result_item['label']} ({result_item['system']})**")
            st.write(result["answer"])
            st.caption(f"{result['verification_status']} · {result['confidence']:.0%} confidence")
            if result.get("sources"):
                st.dataframe(
                    [{"Source": source["title"], "Type": source["source_type"], "Trust": source["trust"]} for source in result["sources"]],
                    hide_index=True,
                    use_container_width=True,
                )