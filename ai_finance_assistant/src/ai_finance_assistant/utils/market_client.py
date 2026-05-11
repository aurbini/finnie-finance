from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd
import requests
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from ai_finance_assistant.core.logging_config import get_logger
from ai_finance_assistant.core.settings import EnvSettings, MarketConf

_log = get_logger(__name__)


@dataclass
class QuoteResult:
    symbol: str
    price: float | None
    currency: str | None
    name: str | None
    error: str | None = None
    warning: str | None = None


class MarketClient:
    """yfinance-first quotes with optional Alpha Vantage fallback."""

    def __init__(self, *, market: MarketConf, env: EnvSettings) -> None:
        self.market = market
        self.env = env
        self._cache: dict[str, tuple[float, QuoteResult]] = {}
        self._ttl = float(market.quote_cache_ttl_seconds)

    def quote(self, symbol: str) -> QuoteResult:
        sym = symbol.upper().strip()
        now = time.time()
        cached = self._cache.get(sym)
        if cached and now - cached[0] < self._ttl:
            return cached[1]

        res = self._quote_yfinance(sym)
        if res.price is None and self.env.alpha_vantage_api_key:
            av = self._quote_alpha_vantage(sym)
            if av.price is not None:
                self._cache[sym] = (now, av)
                return av
            if av.warning and not res.error:
                res.warning = av.warning
            if av.error and not res.error:
                res.error = av.error

        if res.price is None and not res.error:
            res.error = "Could not resolve a live price. Check the symbol or try again later."
        self._cache[sym] = (now, res)
        return res

    def _quote_yfinance(self, symbol: str) -> QuoteResult:
        try:
            t = yf.Ticker(symbol)
            info = t.info or {}
            hist = t.history(period="7d")
            price: float | None = None
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])
            if price is None:
                raw = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
                price = float(raw) if raw is not None else None
            return QuoteResult(
                symbol=symbol,
                price=price,
                currency=info.get("currency"),
                name=info.get("shortName") or info.get("longName"),
            )
        except Exception as exc:  # pragma: no cover - network variability
            _log.warning("yfinance quote failed for %s: %s", symbol, exc)
            return QuoteResult(
                symbol=symbol,
                price=None,
                currency=None,
                name=None,
                error="Live quote temporarily unavailable (yfinance).",
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=0.5, max=8),
        reraise=False,
    )
    def _quote_alpha_vantage(self, symbol: str) -> QuoteResult:
        key = self.env.alpha_vantage_api_key
        if not key:
            return QuoteResult(symbol=symbol, price=None, currency=None, name=None)
        params = {"function": "GLOBAL_QUOTE", "symbol": symbol, "apikey": key}
        try:
            r = requests.get(self.market.alpha_vantage_base_url, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
            if "Note" in data or "Information" in data:
                return QuoteResult(
                    symbol=symbol,
                    price=None,
                    currency=None,
                    name=None,
                    warning="Alpha Vantage rate limit hit; try again soon or rely on yfinance.",
                )
            q = data.get("Global Quote", {})
            price_raw = q.get("05. price")
            return QuoteResult(
                symbol=symbol,
                price=float(price_raw) if price_raw else None,
                currency="USD",
                name=q.get("01. symbol"),
                warning="Quote via Alpha Vantage—cross-check with another source before acting.",
            )
        except Exception as exc:  # pragma: no cover
            _log.warning("Alpha Vantage failed: %s", exc)
            return QuoteResult(
                symbol=symbol,
                price=None,
                currency=None,
                name=None,
                error="Alpha Vantage request failed.",
            )

    def history(self, symbol: str, days: int | None = None) -> pd.DataFrame:
        days = days or self.market.history_days_default
        try:
            t = yf.Ticker(symbol.upper())
            return t.history(period=f"{max(days, 5)}d")
        except Exception as exc:  # pragma: no cover
            _log.warning("yfinance history failed for %s: %s", symbol, exc)
            return pd.DataFrame()
