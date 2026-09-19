"""Pure pandas/numpy technical indicators (no TA-Lib dependency)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=max(2, n // 2)).mean()


def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, min_periods=max(2, n // 2), adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.fillna(100.0).where(avg_gain.notna(), np.nan)


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


def true_range(df: pd.DataFrame) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    components = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    )
    return components.max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / n, min_periods=n, adjust=False).mean()


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=max(2, n // 2)).std()
    return mid - k * sd, mid, mid + k * sd


def annualized_vol(close: pd.Series, n: int = 63) -> float:
    rets = close.pct_change().dropna().tail(n)
    if len(rets) < 5:
        return float("nan")
    return float(rets.std() * np.sqrt(252))


def pct_return(close: pd.Series, periods: int) -> float:
    if len(close) <= periods:
        return float("nan")
    return float(close.iloc[-1] / close.iloc[-1 - periods] - 1.0)


def swing_levels(df: pd.DataFrame, lookback: int = 20) -> tuple[float, float]:
    """Simple support/resistance from recent rolling extremes."""
    window = df.tail(lookback)
    if window.empty:
        return float("nan"), float("nan")
    return float(window["Low"].min()), float(window["High"].max())


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the standard indicator set to an OHLCV frame."""
    out = df.copy()
    close = out["Close"]
    out["sma20"] = sma(close, 20)
    out["sma50"] = sma(close, 50)
    out["sma200"] = sma(close, 200)
    out["ema21"] = ema(close, 21)
    out["rsi14"] = rsi(close, 14)
    out["macd"], out["macd_signal"], out["macd_hist"] = macd(close)
    out["atr14"] = atr(out, 14)
    out["bb_low"], out["bb_mid"], out["bb_high"] = bollinger(close, 20, 2.0)
    out["ret_1d"] = close.pct_change()
    return out
