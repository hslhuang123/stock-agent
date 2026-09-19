"""Tools the analyst may call.

These are plain functions over a ``MarketContext``. The same functions back both
the deterministic rule engine and the LLM function-calling loop, so numbers are
identical regardless of which analyst is active.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from core.features.indicators import add_indicators, swing_levels
from core.features.trending import compute_feature_table


@dataclass
class MarketContext:
    """Everything the analyst needs, resolved once per run."""

    prices: dict[str, pd.DataFrame]
    features: pd.DataFrame
    benchmark: str = "SPY"
    extra: dict[str, Any] = field(default_factory=dict)

    def has(self, ticker: str) -> bool:
        return ticker in self.features.index and ticker in self.prices

    def row(self, ticker: str) -> pd.Series:
        return self.features.loc[ticker]

    def frame(self, ticker: str) -> pd.DataFrame:
        raw = self.prices[ticker]
        return raw if "rsi14" in raw.columns else add_indicators(raw)


# --------------------------------------------------------------------------- #
# Tool implementations
# --------------------------------------------------------------------------- #

def get_trend_score(ctx: MarketContext, ticker: str) -> dict[str, Any]:
    row = ctx.row(ticker)
    return {
        "ticker": ticker,
        "trend_score": round(float(row["trend_score"]), 1),
        "factor_scores": {
            k.replace("score_", ""): float(row[k])
            for k in row.index
            if k.startswith("score_")
        },
    }


def get_technicals(ctx: MarketContext, ticker: str) -> dict[str, Any]:
    row = ctx.row(ticker)
    keys = [
        "close", "rsi14", "macd_hist", "atr14", "atr_pct", "sma20", "sma50",
        "sma200", "sma50_slope", "ann_vol", "vol_ratio", "ext_from_sma20",
    ]
    out = {k: _clean(row.get(k)) for k in keys}
    out["above_sma50"] = bool(row["close"] > row["sma50"]) if pd.notna(row["sma50"]) else None
    out["above_sma200"] = bool(row["close"] > row["sma200"]) if pd.notna(row["sma200"]) else None
    return out


def get_levels(ctx: MarketContext, ticker: str) -> dict[str, Any]:
    df = ctx.frame(ticker)
    support, resistance = swing_levels(df, 20)
    return {
        "ticker": ticker,
        "support_20d": _clean(support),
        "resistance_20d": _clean(resistance),
        "high_52w": _clean(df["Close"].tail(252).max()),
        "low_52w": _clean(df["Close"].tail(252).min()),
    }


def get_price_history(ctx: MarketContext, ticker: str, days: int = 60) -> dict[str, Any]:
    df = ctx.frame(ticker).tail(days)
    return {
        "ticker": ticker,
        "days": len(df),
        "start": df.index[0].date().isoformat() if len(df) else None,
        "end": df.index[-1].date().isoformat() if len(df) else None,
        "close_series": [round(float(x), 2) for x in df["Close"].tolist()],
        "volume_mean": _clean(float(df["Volume"].mean()) if len(df) else None),
    }


def get_relative_strength(ctx: MarketContext, ticker: str) -> dict[str, Any]:
    row = ctx.row(ticker)
    return {
        "ticker": ticker,
        "benchmark": ctx.benchmark,
        "ret_1m": _clean(row.get("ret_1m")),
        "ret_3m": _clean(row.get("ret_3m")),
        "ret_6m": _clean(row.get("ret_6m")),
        "rel_strength_3m": _clean(row.get("rel_strength_3m")),
    }


def get_fundamentals(ctx: MarketContext, ticker: str) -> dict[str, Any]:
    """Best-effort fundamentals via yfinance. Returns {} when unavailable."""
    try:
        import yfinance as yf

        info = yf.Ticker(ticker).info or {}
    except Exception:
        return {"ticker": ticker, "available": False}
    keys = ["sector", "industry", "marketCap", "trailingPE", "forwardPE", "priceToSalesTrailing12Months",
            "profitMargins", "revenueGrowth", "earningsGrowth", "debtToEquity", "returnOnEquity"]
    out = {k: info.get(k) for k in keys if info.get(k) is not None}
    out["ticker"] = ticker
    out["available"] = bool(out)
    return out


def get_news(ctx: MarketContext, ticker: str, limit: int = 6) -> dict[str, Any]:
    """Best-effort recent headlines via yfinance."""
    try:
        import yfinance as yf

        items = yf.Ticker(ticker).news or []
    except Exception:
        return {"ticker": ticker, "available": False, "headlines": []}
    headlines = []
    for item in items[:limit]:
        content = item.get("content", item) if isinstance(item, dict) else {}
        title = content.get("title") or item.get("title")
        publisher = (content.get("provider") or {}).get("displayName") if isinstance(content.get("provider"), dict) else item.get("publisher")
        if title:
            headlines.append({"title": title, "publisher": publisher})
    return {"ticker": ticker, "available": bool(headlines), "headlines": headlines}


def get_market_regime(ctx: MarketContext) -> dict[str, Any]:
    """Broad market context from the benchmark."""
    if ctx.benchmark not in ctx.features.index:
        return {"benchmark": ctx.benchmark, "available": False}
    row = ctx.row(ctx.benchmark)
    return {
        "benchmark": ctx.benchmark,
        "available": True,
        "close": _clean(row.get("close")),
        "ret_1m": _clean(row.get("ret_1m")),
        "ret_3m": _clean(row.get("ret_3m")),
        "above_sma50": bool(row["close"] > row["sma50"]) if pd.notna(row["sma50"]) else None,
        "above_sma200": bool(row["close"] > row["sma200"]) if pd.notna(row["sma200"]) else None,
        "ann_vol": _clean(row.get("ann_vol")),
    }


def _clean(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        f = float(value)
        if np.isnan(f) or np.isinf(f):
            return None
        return round(f, 4)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


# --------------------------------------------------------------------------- #
# Registry + LLM tool schemas
# --------------------------------------------------------------------------- #

TOOL_REGISTRY: dict[str, Callable[..., dict[str, Any]]] = {
    "get_trend_score": get_trend_score,
    "get_technicals": get_technicals,
    "get_levels": get_levels,
    "get_price_history": get_price_history,
    "get_relative_strength": get_relative_strength,
    "get_fundamentals": get_fundamentals,
    "get_news": get_news,
    "get_market_regime": get_market_regime,
}

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_trend_score",
            "description": "Composite 0-100 trend score and the per-factor sub-scores.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_technicals",
            "description": "RSI, MACD, ATR, moving averages, volatility, volume ratio.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_levels",
            "description": "Support, resistance, and 52-week high/low levels.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_relative_strength",
            "description": "Returns over 1/3/6 months and excess return vs benchmark.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_fundamentals",
            "description": "Headline fundamentals: sector, valuation, growth, margins.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_news",
            "description": "Recent news headlines for the ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "limit": {"type": "integer", "default": 6},
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_market_regime",
            "description": "Broad market trend/volatility context from the benchmark index.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


def dispatch(ctx: MarketContext, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute a tool by name. Unknown tools and bad args return an error dict."""
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        return {"error": f"unknown tool: {name}"}
    args = dict(arguments or {})
    try:
        if name == "get_market_regime":
            return fn(ctx)
        return fn(ctx, **args)
    except Exception as exc:  # pragma: no cover - defensive
        return {"error": f"{name} failed: {exc}"}
