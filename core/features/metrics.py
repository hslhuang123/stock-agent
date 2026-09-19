"""Point-in-time metric computation.

``metrics_at(df, idx)`` computes every screening metric using **only** data up to
and including ``idx``. The backtest leans on this to avoid look-ahead bias: the
same code that ranks today's universe also ranked it on any past date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.features.indicators import add_indicators

TREND_QUALITY_MAX = 5.0


def _ret_at(close: pd.Series, idx: int, periods: int) -> float:
    if idx - periods < 0:
        return np.nan
    base = close.iloc[idx - periods]
    if not base:
        return np.nan
    return float(close.iloc[idx] / base - 1.0)


def _ann_vol_at(close: pd.Series, idx: int, n: int = 63) -> float:
    start = max(0, idx - n)
    window = close.iloc[start : idx + 1].pct_change().dropna()
    if len(window) < 5:
        return np.nan
    return float(window.std() * np.sqrt(252))


def _levels_at(df: pd.DataFrame, idx: int, lookback: int = 20) -> tuple[float, float]:
    start = max(0, idx - lookback + 1)
    window = df.iloc[start : idx + 1]
    if window.empty:
        return np.nan, np.nan
    return float(window["Low"].min()), float(window["High"].max())


def trend_quality(metrics: dict) -> float:
    """Count how many structural trend conditions hold (0..5)."""
    score = 0.0
    close = metrics.get("close")
    sma20, sma50, sma200 = metrics.get("sma20"), metrics.get("sma50"), metrics.get("sma200")
    if _ok(close) and _ok(sma20) and close > sma20:
        score += 1
    if _ok(sma20) and _ok(sma50) and sma20 > sma50:
        score += 1
    if _ok(sma50) and _ok(sma200) and sma50 > sma200:
        score += 1
    if _ok(metrics.get("sma50_slope")) and metrics["sma50_slope"] > 0:
        score += 1
    if _ok(metrics.get("macd_hist")) and metrics["macd_hist"] > 0:
        score += 1
    return score


def _ok(value) -> bool:
    return value is not None and not (isinstance(value, float) and np.isnan(value))


def metrics_at(
    df: pd.DataFrame,
    idx: int,
    bench_ret_3m: float | None = None,
    min_history: int = 0,
) -> dict | None:
    """Return the metric row for ``df`` at position ``idx`` (negative = from end).

    ``df`` must already have indicators attached (``add_indicators``).
    """
    if idx < 0:
        idx += len(df)
    if idx < min_history or idx < 0 or idx >= len(df):
        return None

    if "sma20" not in df.columns:
        df = add_indicators(df)

    close = df["Close"]
    last = float(close.iloc[idx])
    if not last or np.isnan(last):
        return None

    ret_3m = _ret_at(close, idx, 63)
    ret_6m = _ret_at(close, idx, 126)
    vol_ann = _ann_vol_at(close, idx, 63)
    support, resistance = _levels_at(df, idx, 20)

    window_252 = close.iloc[max(0, idx - 251) : idx + 1]
    high_52w = float(window_252.max())
    low_52w = float(window_252.min())

    sma50 = df["sma50"]
    sma50_slope = np.nan
    if idx - 21 >= 0 and _ok(sma50.iloc[idx]) and _ok(sma50.iloc[idx - 21]) and sma50.iloc[idx - 21]:
        sma50_slope = float(sma50.iloc[idx] / sma50.iloc[idx - 21] - 1.0)

    vol = df["Volume"]
    v20_start = max(0, idx - 19)
    v60_start = max(0, idx - 59)
    vol20 = float(vol.iloc[v20_start : idx + 1].mean())
    vol60 = float(vol.iloc[v60_start : idx + 1].mean())
    vol_ratio = vol20 / vol60 if vol60 else np.nan

    dollar_vol = float((close * vol).iloc[v20_start : idx + 1].mean())

    sma20_val = float(df["sma20"].iloc[idx]) if _ok(df["sma20"].iloc[idx]) else np.nan

    return {
        "date": df.index[idx].date().isoformat(),
        "close": last,
        "ret_1m": _ret_at(close, idx, 21),
        "ret_3m": ret_3m,
        "ret_6m": ret_6m,
        "rel_strength_3m": (ret_3m - bench_ret_3m)
        if _ok(ret_3m) and _ok(bench_ret_3m)
        else np.nan,
        "rsi14": float(df["rsi14"].iloc[idx]) if _ok(df["rsi14"].iloc[idx]) else np.nan,
        "macd_hist": float(df["macd_hist"].iloc[idx]) if _ok(df["macd_hist"].iloc[idx]) else np.nan,
        "atr14": float(df["atr14"].iloc[idx]) if _ok(df["atr14"].iloc[idx]) else np.nan,
        "atr_pct": float(df["atr14"].iloc[idx] / last) if _ok(df["atr14"].iloc[idx]) and last else np.nan,
        "sma20": sma20_val,
        "sma50": float(df["sma50"].iloc[idx]) if _ok(df["sma50"].iloc[idx]) else np.nan,
        "sma200": float(df["sma200"].iloc[idx]) if _ok(df["sma200"].iloc[idx]) else np.nan,
        "sma50_slope": sma50_slope,
        "ann_vol": vol_ann,
        "vol_ratio": vol_ratio,
        "avg_dollar_volume": dollar_vol,
        "support_20d": support,
        "resistance_20d": resistance,
        "high_52w": high_52w,
        "low_52w": low_52w,
        "pct_from_52w_high": (last / high_52w - 1) if high_52w else np.nan,
        "proximity_52w_high": (last / high_52w) if high_52w else np.nan,
        "risk_adj_mom": (ret_3m / vol_ann)
        if _ok(ret_3m) and _ok(vol_ann) and vol_ann
        else np.nan,
        "ext_from_sma20": (last / sma20_val - 1) if _ok(sma20_val) and sma20_val else np.nan,
    }


def price_at(df: pd.DataFrame, date: pd.Timestamp) -> tuple[int, float] | tuple[None, None]:
    """Last available close on or before ``date``. Returns (position, price)."""
    pos = int(df.index.searchsorted(date, side="right")) - 1
    if pos < 0 or pos >= len(df):
        return None, None
    price = float(df["Close"].iloc[pos])
    if np.isnan(price):
        return None, None
    return pos, price
