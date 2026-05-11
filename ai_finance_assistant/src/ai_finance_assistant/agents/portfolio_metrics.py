from __future__ import annotations

import json
from typing import Any

import pandas as pd


def summarize_holdings_json(portfolio_json: str | None) -> str:
    if not portfolio_json:
        return "No portfolio JSON supplied."
    try:
        rows: list[Any] = json.loads(portfolio_json)
    except json.JSONDecodeError:
        return "Holdings payload is invalid JSON."
    if not isinstance(rows, list) or not rows:
        return "Holdings table is empty — add instruments and allocations for analysis."

    df = pd.DataFrame(rows)
    if "symbol" not in df.columns or "weight_pct" not in df.columns:
        return "Each row needs `symbol` and `weight_pct`."
    weights = pd.to_numeric(df["weight_pct"], errors="coerce").fillna(0.0)
    total = weights.sum()
    parts = weights / 100.0
    hhi = float((parts ** 2).sum()) * 100  # approximate concentration index scaled to 0-100
    top_symbols = df.assign(w=weights).sort_values("w", ascending=False).head(5)

    diag = []
    diag.append(f"Total reported weight (%): **{total:.2f}** — treat as illustrative if incomplete.")
    diag.append(f"Concentration (HHI-style, higher = more concentrated): **{hhi:.2f}**")
    diag.append("\nTop weights:\n")
    for _, row in top_symbols.iterrows():
        diag.append(f"- `{row['symbol']}` → {float(row['w']):.2f}%")
    if len(df["symbol"]) < 10 and hhi > 35:
        diag.append(
            "\nEducational note: concentrated portfolios swing more; diversification can "
            "reduce single-name shock—this is informational, not a recommendation."
        )
    return "\n".join(diag)
