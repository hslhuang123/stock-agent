"""Trend detection: turn raw price frames into a ranked feature table.

All numbers here are deterministic. Nothing in this module calls an LLM.

Metric computation lives in :mod:`core.features.metrics` so the exact same code
can evaluate the universe today and on any historical date (see the backtester).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from core.features.indicators import add_indicators, pct_return
from core.features.metrics import TREND_QUALITY_MAX, metrics_at, trend_quality

log = logging.getLogger(__name__)
__all__ = ["compute_feature_table", "rank_trending", "top_candidates", "WEIGHT_TO_COLUMN", "TREND_QUALITY_MAX"]

# Column used for each weighted factor.
WEIGHT_TO_COLUMN = {
    "momentum_3m": "ret_3m",
    "momentum_6m": "ret_6m",
    "relative_strength_3m": "rel_strength_3m",
    "trend_quality": "trend_quality_raw",
    "volume_surge": "vol_ratio",
    "proximity_52w_high": "proximity_52w_high",
    "risk_adjusted_momentum": "risk_adj_mom",
}


def compute_feature_table(
    prices: dict[str, pd.DataFrame],
    benchmark: pd.DataFrame | None = None,
    min_history_days: int = 130,
) -> pd.DataFrame:
    """Build one row of metrics per ticker (latest observation)."""
    bench_ret_3m = np.nan
    if benchmark is not None and len(benchmark) > 63:
        bench_ret_3m = pct_return(benchmark["Close"], 63)

    rows: list[dict[str, Any]] = []
    for ticker, raw in prices.items():
        df = raw if "sma20" in raw.columns else add_indicators(raw)
        if len(df) < min_history_days:
            log.debug("skip %s: only %d rows", ticker, len(df))
            continue
        metrics = metrics_at(df, -1, bench_ret_3m=bench_ret_3m)
        if metrics is None:
            continue
        metrics["ticker"] = ticker
        metrics["trend_quality_raw"] = trend_quality(metrics)
        rows.append(metrics)

    if not rows:
        return pd.DataFrame()

    table = pd.DataFrame(rows).set_index("ticker")
    return table


def _percentile(series: pd.Series) -> pd.Series:
    if series.dropna().empty:
        return pd.Series(0.5, index=series.index)
    return series.rank(pct=True).fillna(0.5)


def rank_trending(
    table: pd.DataFrame,
    weights: dict[str, float],
    screening: dict[str, Any],
) -> pd.DataFrame:
    """Apply liquidity filters and compute the weighted 0-100 trend score."""
    if table.empty:
        return table

    df = table.copy()

    # --- Hard filters (tradability) ---------------------------------------
    df = df[df["close"] >= float(screening.get("min_price", 5.0))]
    df = df[df["avg_dollar_volume"] >= float(screening.get("min_avg_dollar_volume", 0))]
    df = df[df["ret_3m"].notna() & df["ret_6m"].notna()]

    if df.empty:
        return df

    # --- Percentile-rank each factor across the surviving universe --------
    total_weight = sum(weights.values()) or 1.0
    score = pd.Series(0.0, index=df.index)
    for factor, weight in weights.items():
        col = WEIGHT_TO_COLUMN.get(factor)
        if col is None or col not in df.columns:
            continue
        ranked = _percentile(df[col])
        df[f"score_{factor}"] = (ranked * 100).round(1)
        score += ranked * (weight / total_weight)

    df["trend_score"] = (score * 100).round(1)
    df = df.sort_values("trend_score", ascending=False)
    return df


def top_candidates(ranked: pd.DataFrame, n: int) -> pd.DataFrame:
    return ranked.head(n)
