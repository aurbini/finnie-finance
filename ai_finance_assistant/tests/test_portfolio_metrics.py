import json

from ai_finance_assistant.agents.portfolio_metrics import summarize_holdings_json


def test_summarize_basic_weights():
    rows = [
        {"symbol": "AAA", "weight_pct": "60"},
        {"symbol": "BBB", "weight_pct": "40"},
    ]
    blob = summarize_holdings_json(json.dumps(rows))
    assert "60" in blob and "BBB" in blob


def test_invalid_payload():
    blob = summarize_holdings_json("{")
    assert "invalid" in blob.lower()


def test_empty_rows():
    assert "empty" in summarize_holdings_json("[]").lower()
