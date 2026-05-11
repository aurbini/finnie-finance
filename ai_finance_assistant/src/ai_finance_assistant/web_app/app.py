from __future__ import annotations

import json
import uuid

import pandas as pd
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from ai_finance_assistant.core.exceptions import ConfigurationError
from ai_finance_assistant.core.llm import chat_model_llm
from ai_finance_assistant.core.logging_config import configure_logging
from ai_finance_assistant.core.settings import load_env, load_settings
from ai_finance_assistant.rag.embeddings import get_embeddings
from ai_finance_assistant.rag.vectorstore import load_faiss_index
from ai_finance_assistant.utils.market_client import MarketClient
from ai_finance_assistant.utils.paths import faiss_index_directory
from ai_finance_assistant.workflow.graph_builder import GraphDependencies, build_compiled_graph, make_retrieve_fn


def _bootstrap():
    settings = load_settings()
    env = load_env()
    embedder = get_embeddings(settings)
    idx = load_faiss_index(embedder, faiss_index_directory(settings))
    llm = chat_model_llm(settings.llm, env)
    market = MarketClient(market=settings.market, env=env)
    deps = GraphDependencies(
        llm=llm,
        settings=settings,
        retrieve=make_retrieve_fn(idx, settings),
        market=market,
    )
    graph = build_compiled_graph(deps)
    return settings, env, graph, market


@st.cache_resource(show_spinner=False)
def cached_bootstrap():
    return _bootstrap()


def run() -> None:
    configure_logging("INFO")
    st.set_page_config(page_title="AI Finance Assistant", layout="wide")

    try:
        settings, env, graph, market = cached_bootstrap()
    except ConfigurationError as exc:
        st.error(str(exc))
        st.info(
            "Copy `.env.example` to `.env`, add at least one chat model key, then restart "
            "Streamlit (`streamlit run ...`)."
        )
        st.stop()

    st.title("AI Finance Assistant")
    st.markdown(f"_{settings.app.educational_disclaimer}_")

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "portfolio_rows" not in st.session_state:
        st.session_state.portfolio_rows = [
            {"symbol": "VOO", "weight_pct": 55.0},
            {"symbol": "VXUS", "weight_pct": 20.0},
            {"symbol": "BND", "weight_pct": 25.0},
        ]
    if "goal_context" not in st.session_state:
        st.session_state.goal_context = ""
    if "chat_snapshot" not in st.session_state:
        st.session_state.chat_snapshot = []
    if "last_meta" not in st.session_state:
        st.session_state.last_meta = {}

    tabs = st.tabs(["Assistant", "Portfolio", "Markets", "Goals", "About"])

    with tabs[0]:
        _assistant(graph)

    with tabs[1]:
        _portfolio()

    with tabs[2]:
        _markets(market)

    with tabs[3]:
        _goals(graph)

    with tabs[4]:
        _about()


def _assistant(graph) -> None:
    st.subheader("Conversation")
    _, col_b = st.columns([2, 1])
    with col_b:
        if st.button("New chat thread"):
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.chat_snapshot = []
            st.session_state.last_meta = {}
            st.rerun()
        st.caption(f"Thread `{st.session_state.thread_id[:8]}…`")

    user = st.chat_input("Ask a question (try “Roth vs traditional”, portfolio text, or ticker symbol like MSFT)...")
    invoke_cfg = {
        "configurable": {
            "thread_id": st.session_state.thread_id,
            "portfolio_json": json.dumps(st.session_state.portfolio_rows),
            "goal_context": st.session_state.goal_context,
        }
    }

    if user:
        with st.spinner("Thinking with LangGraph + retrieval…"):
            output = graph.invoke({"messages": [HumanMessage(content=user)]}, config=invoke_cfg)
        st.session_state.chat_snapshot = list(output.get("messages", []))
        st.session_state.last_meta = {
            "intent": output.get("intent"),
            "sources": output.get("retrieved_sources_md", ""),
        }

    meta = st.session_state.get("last_meta", {})
    if meta.get("intent"):
        st.info(f"Last routed intent: **{meta.get('intent')}**")
        with st.expander("Retrieved KB sources (titles)"):
            st.markdown(meta.get("sources") or "_No sources surfaced._")

    for m in st.session_state.get("chat_snapshot", []):
        if isinstance(m, HumanMessage):
            st.chat_message("user").markdown(str(m.content))
        elif isinstance(m, AIMessage):
            st.chat_message("assistant").markdown(str(m.content))

    if not st.session_state.get("chat_snapshot"):
        st.caption("Start by asking a question—the graph will route to the right educational agent.")


def _portfolio() -> None:
    st.subheader("Holdings (weights are illustrative)")
    st.caption(
        "Provide `symbol` plus `weight_pct` (0–100). The portfolio agent merges this table with knowledge-base context."
    )

    df = pd.DataFrame(st.session_state.portfolio_rows)
    edited = st.data_editor(df, num_rows="dynamic", use_container_width=True)
    rows = edited.to_dict("records")
    # drop empty rows
    clean = [r for r in rows if str(r.get("symbol", "")).strip()]
    st.session_state.portfolio_rows = clean

    if clean:
        weights = [float(r.get("weight_pct") or 0) for r in clean]
        symbols = [str(r.get("symbol")) for r in clean]
        st.bar_chart(pd.Series(weights, index=symbols))

    st.markdown("---")
    if st.button("Preview JSON sent to agents"):
        st.json(clean)


def _markets(market_client: MarketClient) -> None:
    st.subheader("Market snapshot (yfinance-backed)")
    ticker = st.text_input("Ticker", value="SPY").upper().strip()
    if st.button("Fetch quote & history"):
        q = market_client.quote(ticker)
        if q.error:
            st.error(q.error)
        if q.warning:
            st.warning(q.warning)
        cols = st.columns(3)
        cols[0].metric("Symbol", q.symbol)
        cols[1].metric("Last price (best effort)", f"{q.price:.2f}" if q.price is not None else "n/a")
        cols[2].metric("Currency", q.currency or "n/a")
        if q.name:
            st.caption(q.name)
        hist = market_client.history(ticker)
        if hist.empty:
            st.info("No historical rows returned—APIs can be flaky; try again shortly.")
        else:
            st.line_chart(hist["Close"])


def _goals(graph) -> None:
    st.subheader("Goal planning helper")
    goal = st.text_area("Describe the goal", value="Retire in ~25 years while keeping moderate volatility.")
    horizon = st.slider("Years to goal (illustrative)", 1, 50, 25)
    risk = st.slider("Self-assessed risk tolerance (1 conservative – 5 adventurous)", 1, 5, 3)
    if st.button("Save goal context"):
        st.session_state.goal_context = (
            f"Goal summary: {goal}\nHorizon years: {horizon}\nRisk score: {risk}/5"
        )
        st.success("Saved. Open the Assistant tab and ask for goal planning guidance.")

    if st.button("Draft quick plan (invokes graph)") and st.session_state.goal_context:
        invoke_cfg = {
            "configurable": {
                "thread_id": st.session_state.thread_id,
                "portfolio_json": json.dumps(st.session_state.portfolio_rows),
                "goal_context": st.session_state.goal_context,
            }
        }
        with st.spinner("Contacting goal-planning agent…"):
            out = graph.invoke(
                {
                    "messages": [
                        HumanMessage(
                            content=(
                                "Goal planning: review my saved goal context, horizon, and risk score. "
                                "Offer educational tradeoffs, not promises."
                            )
                        )
                    ]
                },
                config=invoke_cfg,
            )
        st.session_state.chat_snapshot = list(out.get("messages", []))
        st.session_state.last_meta = {
            "intent": out.get("intent"),
            "sources": out.get("retrieved_sources_md", ""),
        }
        ai = [m for m in out.get("messages", []) if isinstance(m, AIMessage)]
        if ai:
            st.markdown(ai[-1].content)


def _about() -> None:
    st.subheader("Architecture & safety notes")
    st.markdown(
        """
- **Router** uses lightweight keyword rules inspired by the assignment rubric; it is not infallible—
  rephrase if the wrong specialist answers.
- **RAG** snippets are generated educational articles stored under `data/articles` and embedded locally.
- **Markets** integrate `yfinance` with optional Alpha Vantage when `ALPHA_VANTAGE_API_KEY` is set.
- **Multi-turn memory** uses LangGraph's in-process `MemorySaver` keyed by the thread id shown in the Assistant tab.
- **Disclaimers:** This project is for classroom-style literacy, not personalized investment, tax, or legal advice.
"""
    )


if __name__ == "__main__":
    run()
