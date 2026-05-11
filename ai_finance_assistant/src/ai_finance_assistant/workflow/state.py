from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


IntentName = Literal[
    "finance_qa",
    "portfolio_analysis",
    "market_analysis",
    "goal_planning",
    "news_synthesis",
    "tax_education",
]


class AgentGraphState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    intent: IntentName | str
    routed_agent: str
    retrieved_sources_md: str
