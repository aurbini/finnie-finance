from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from ai_finance_assistant.agents.portfolio_metrics import summarize_holdings_json
from ai_finance_assistant.agents.prompts import system_for
from ai_finance_assistant.core.settings import Settings
from ai_finance_assistant.rag.retrieve import format_context_for_prompt, format_sources_markdown, retrieve_similar
from ai_finance_assistant.utils.market_client import MarketClient
from ai_finance_assistant.workflow.router import classify_intent, last_user_text
from ai_finance_assistant.workflow.state import AgentGraphState, IntentName

if TYPE_CHECKING:
    from langchain_community.vectorstores import FAISS


class RetrieveFn(Protocol):
    def __call__(self, query: str, categories: list[str] | None) -> tuple[str, str]: ...


@dataclass
class GraphDependencies:
    llm: BaseChatModel
    settings: Settings
    retrieve: RetrieveFn
    market: MarketClient


def make_retrieve_fn(store: "FAISS | None", settings: Settings) -> RetrieveFn:
    def _inner(query: str, categories: list[str] | None) -> tuple[str, str]:
        if store is None:
            return (
                "(Vector store not loaded — run `python -m ai_finance_assistant.rag.build_index`.)",
                "",
            )
        chunks, _ = retrieve_similar(store, settings, query, categories=categories)
        if not chunks:
            return ("(No matching knowledge base excerpts.)", "")
        return format_context_for_prompt(chunks), format_sources_markdown(chunks)

    return _inner


def _append_disclaimer(settings: Settings, text: str) -> str:
    d = settings.app.educational_disclaimer.strip()
    if not d:
        return text
    return f"{text.strip()}\n\n*{d}*"


def _invoke_agent(
    deps: GraphDependencies,
    messages: list,
    system_extra: str,
    context_block: str,
) -> str:
    sys = SystemMessage(content=system_extra + "\n\nKnowledge excerpts:\n" + context_block)
    prompt_messages = [sys, *messages[-8:]]
    out = deps.llm.invoke(prompt_messages)
    return str(getattr(out, "content", out))


def route_node(state: AgentGraphState) -> dict[str, Any]:
    text = last_user_text(state.get("messages", []))
    intent: IntentName = classify_intent(text)
    return {"intent": intent, "routed_agent": intent}


def _mapping_to_dict(mapping: Mapping | None) -> dict:
    if not mapping:
        return {}
    if isinstance(mapping, dict):
        return mapping
    return dict(mapping)


def build_compiled_graph(deps: GraphDependencies):
    def finance_qa(state: AgentGraphState, config: Mapping | None = None) -> dict[str, Any]:
        text = last_user_text(state["messages"])
        ctx, src = deps.retrieve(text, None)
        body = _invoke_agent(
            deps,
            state["messages"],
            system_for("finance_qa"),
            ctx,
        )
        return {
            "messages": [AIMessage(content=_append_disclaimer(deps.settings, body))],
            "retrieved_sources_md": src,
        }

    def portfolio_analysis(state: AgentGraphState, config: Mapping | None = None) -> dict[str, Any]:
        text = last_user_text(state["messages"])
        cfg = _mapping_to_dict(_mapping_to_dict(config).get("configurable"))
        portfolio_json = cfg.get("portfolio_json", "[]")
        metrics = summarize_holdings_json(str(portfolio_json))
        ctx, src = deps.retrieve(text, ["portfolio", "investing_basics"])
        body = _invoke_agent(
            deps,
            state["messages"],
            system_for("portfolio_analysis") + f"\n\nComputed metrics (illustrative):\n{metrics}",
            ctx,
        )
        return {
            "messages": [AIMessage(content=_append_disclaimer(deps.settings, body))],
            "retrieved_sources_md": src,
        }

    def market_analysis(state: AgentGraphState, config: Mapping | None = None) -> dict[str, Any]:
        text = last_user_text(state["messages"])
        sym = text.strip().split()[0] if text.strip() else "SPY"
        q = deps.market.quote(sym)
        hist = deps.market.history(sym)
        last_close = None
        if not hist.empty:
            last_close = float(hist["Close"].iloc[-1])
        snap = (
            f"Symbol: {q.symbol}\n"
            f"Name: {q.name}\n"
            f"Quote price (best effort): {q.price}\n"
            f"History last close: {last_close}\n"
            f"Warnings/errors: {q.warning or ''} {q.error or ''}"
        )
        ctx, src = deps.retrieve(text, ["markets", "investing_basics"])
        body = _invoke_agent(
            deps,
            state["messages"],
            system_for("market_analysis") + f"\n\nMarket snapshot (verify externally):\n{snap}",
            ctx,
        )
        return {
            "messages": [AIMessage(content=_append_disclaimer(deps.settings, body))],
            "retrieved_sources_md": src,
        }

    def goal_planning(state: AgentGraphState, config: Mapping | None = None) -> dict[str, Any]:
        text = last_user_text(state["messages"])
        cfg = _mapping_to_dict(_mapping_to_dict(config).get("configurable"))
        goal_ctx = cfg.get("goal_context", "")
        ctx, src = deps.retrieve(text, ["retirement", "investing_basics", "behavioral"])
        body = _invoke_agent(
            deps,
            state["messages"],
            system_for("goal_planning") + f"\n\nUser goal context from UI:\n{goal_ctx}",
            ctx,
        )
        return {
            "messages": [AIMessage(content=_append_disclaimer(deps.settings, body))],
            "retrieved_sources_md": src,
        }

    def news_synthesis(state: AgentGraphState, config: Mapping | None = None) -> dict[str, Any]:
        text = last_user_text(state["messages"])
        ctx, src = deps.retrieve(text, ["markets", "behavioral"])
        body = _invoke_agent(
            deps,
            state["messages"],
            system_for("news_synthesis"),
            ctx,
        )
        return {
            "messages": [AIMessage(content=_append_disclaimer(deps.settings, body))],
            "retrieved_sources_md": src,
        }

    def tax_education(state: AgentGraphState, config: Mapping | None = None) -> dict[str, Any]:
        text = last_user_text(state["messages"])
        ctx, src = deps.retrieve(text, ["taxes"])
        body = _invoke_agent(
            deps,
            state["messages"],
            system_for("tax_education"),
            ctx,
        )
        return {
            "messages": [AIMessage(content=_append_disclaimer(deps.settings, body))],
            "retrieved_sources_md": src,
        }

    g = StateGraph(AgentGraphState)
    g.add_node("router", route_node)
    g.add_node("finance_qa", finance_qa)
    g.add_node("portfolio_analysis", portfolio_analysis)
    g.add_node("market_analysis", market_analysis)
    g.add_node("goal_planning", goal_planning)
    g.add_node("news_synthesis", news_synthesis)
    g.add_node("tax_education", tax_education)

    g.add_edge(START, "router")

    def _branch(state: AgentGraphState) -> IntentName:
        intent = state.get("intent")
        if intent in {
            "finance_qa",
            "portfolio_analysis",
            "market_analysis",
            "goal_planning",
            "news_synthesis",
            "tax_education",
        }:
            return intent  # type: ignore[return-value]
        return "finance_qa"

    g.add_conditional_edges(
        "router",
        _branch,
        {
            "finance_qa": "finance_qa",
            "portfolio_analysis": "portfolio_analysis",
            "market_analysis": "market_analysis",
            "goal_planning": "goal_planning",
            "news_synthesis": "news_synthesis",
            "tax_education": "tax_education",
        },
    )
    for name in (
        "finance_qa",
        "portfolio_analysis",
        "market_analysis",
        "goal_planning",
        "news_synthesis",
        "tax_education",
    ):
        g.add_edge(name, END)

    return g.compile(checkpointer=MemorySaver())


def user_message(text: str) -> HumanMessage:
    return HumanMessage(content=text)
