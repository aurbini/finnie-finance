from ai_finance_assistant.workflow.router import classify_intent


def test_classifier_tax_intent_first():
    assert classify_intent("Should I prioritize Roth IRA or 401k match?") == "tax_education"


def test_classifier_portfolio_keyword():
    assert classify_intent("How concentrated is my tech portfolio?") == "portfolio_analysis"


def test_classifier_goal_keywords():
    assert classify_intent("I want to retire early — how should I think about risk appetite?") == "goal_planning"


def test_classifier_news_keywords():
    assert classify_intent("Summarize headlines about inflation prints") == "news_synthesis"


def test_classifier_market_all_caps_ticker():
    assert classify_intent("MSFT") == "market_analysis"


def test_default_finance_qa():
    assert classify_intent("What is diversification in one paragraph?") == "finance_qa"
