from __future__ import annotations

import re

from ai_finance_assistant.workflow.state import IntentName


def classify_intent(text: str) -> IntentName:
    raw = text.strip()
    # Tickers typed in ALL CAPS (e.g., MSFT) — avoids misrouting common words like "risk".
    slug = raw.replace(".", "")
    if raw.isupper() and 1 <= len(slug) <= 5 and slug.isalpha():
        return "market_analysis"

    t = text.lower()

    portfolio_pat = (
        r"\b(portfolio|holdings|allocation|diversif|etf basket|positions|weights|rebalance)\b"
    )
    market_pat = (
        r"\b(stock price|share price|ticker|nasdaq|s&p|spy|market cap|yahoo finance|intraday|today's\b price)\b"
    )
    goal_pat = r"\b(goal|save for|buy a house|retire early|financial plan|risk appetite|risk tolerance|time horizon|college fund)\b"
    tax_pat = r"\b(401k|403b|ira|roth|tax bracket|withhold|capital gains|amt|hsa|step.up in basis|estate tax)\b"
    news_pat = r"\b(news|headlines|what happened|earnings call|macro|fed rate|inflation print)\b"

    if re.search(tax_pat, t):
        return "tax_education"
    if re.search(portfolio_pat, t):
        return "portfolio_analysis"
    if re.search(goal_pat, t):
        return "goal_planning"
    if re.search(news_pat, t):
        return "news_synthesis"
    if re.search(market_pat, t):
        return "market_analysis"
    return "finance_qa"


def last_user_text(messages: list) -> str:
    for m in reversed(messages):
        if getattr(m, "type", None) == "human" or m.__class__.__name__ == "HumanMessage":
            return str(m.content)
    return ""
