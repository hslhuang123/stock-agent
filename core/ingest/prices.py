"""Price data ingestion with on-disk caching and an offline synthetic fallback."""
from __future__ import annotations

import hashlib
import logging
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from core.config import DATA_DIR

log = logging.getLogger(__name__)

PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
_CACHE_TTL_HOURS = 6


def _cache_path(key: str) -> Path:
    digest = hashlib.sha1(key.encode()).hexdigest()[:12]
    return DATA_DIR / "prices" / f"{digest}.parquet"


def _normalize(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Clean a raw yfinance frame into Open/High/Low/Close/Volume."""
    if df is None or len(df) == 0:
        return None
    df = df.rename(columns={c: str(c).title() for c in df.columns})
    cols = [c for c in PRICE_COLUMNS if c in df.columns]
    if "Close" not in cols:
        return None
    df = df[cols].copy()
    df = df.dropna(subset=["Close"])
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def _load_cache(key: str) -> pd.DataFrame | None:
    path = _cache_path(key)
    if not path.exists():
        return None
    age_hours = (pd.Timestamp.now() - pd.Timestamp(path.stat().st_mtime, unit="s")).total_seconds() / 3600
    if age_hours > _CACHE_TTL_HOURS:
        return None
    try:
        return pd.read_parquet(path)
    except Exception:  # pragma: no cover - corrupt cache
        return None


def _save_cache(key: str, df: pd.DataFrame) -> None:
    path = _cache_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path)
    except Exception as exc:  # pragma: no cover
        log.debug("cache write failed: %s", exc)


def synthetic_prices(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
    trend_bias: float | None = None,
) -> pd.DataFrame:
    """Deterministic geometric-brownian-motion prices for offline use/tests."""
    days = _period_to_days(period)
    seed = int(hashlib.sha1(ticker.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    if trend_bias is None:
        trend_bias = float(rng.normal(0.08, 0.35))  # annualized drift

    annual_vol = float(rng.uniform(0.18, 0.65))
    dt = 1 / 252
    shocks = rng.normal(
        (trend_bias - 0.5 * annual_vol**2) * dt,
        annual_vol * np.sqrt(dt),
        size=days,
    )
    # Add a mild regime shift so trends look realistic.
    half = days // 2
    shocks[half:] += rng.normal(0, 0.0015, size=days - half)
    close = 40.0 * np.exp(np.cumsum(shocks))

    intraday = np.abs(rng.normal(0, annual_vol * np.sqrt(dt), size=days)) + 0.004
    high = close * (1 + intraday)
    low = close * (1 - intraday * rng.uniform(0.6, 1.4, size=days))
    open_ = close * (1 + rng.normal(0, 0.002, size=days))
    base_vol = float(rng.uniform(3e6, 4e7))
    volume = base_vol * np.exp(rng.normal(0, 0.35, size=days))

    index = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=index,
    )


def _period_to_days(period: str) -> int:
    mapping = {
        "1mo": 22, "3mo": 66, "6mo": 130, "1y": 252, "2y": 504,
        "5y": 1260, "10y": 2520, "max": 2520,
    }
    if period in mapping:
        return mapping[period]
    if period.endswith("y"):
        return int(period[:-1]) * 252
    if period.endswith("mo"):
        return int(period[:-2]) * 22
    return 252


def _trim_to_period(df: pd.DataFrame, period: str, interval: str) -> pd.DataFrame:
    """Keep only the most recent ``period`` worth of bars.

    Used after a wider retry so a fallback to ``5y`` still returns a frame with
    the row count the caller asked for. Indicators warm up over the same number
    of bars either way, which keeps results deterministic.
    """
    if not period or str(period) == "max":
        return df
    if str(interval) not in ("1d", "1wk", "1mo"):
        return df
    n = _period_to_days(str(period))
    if interval == "1wk":
        n = max(1, n // 5)
    elif interval == "1mo":
        n = max(1, n // 21)
    if len(df) <= n:
        return df
    return df.iloc[-n:]


def _fetch_chunk(tickers: list[str], period: str, interval: str) -> dict[str, pd.DataFrame]:
    """Batch-download one small chunk, falling back to per-ticker calls."""
    import yfinance as yf

    out: dict[str, pd.DataFrame] = {}
    try:
        batch = yf.download(
            tickers,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )
        if isinstance(batch.columns, pd.MultiIndex):
            level = batch.columns.get_level_values(0)
            for ticker in tickers:
                if ticker in level:
                    frame = _normalize(batch[ticker])
                    if frame is not None and len(frame):
                        out[ticker] = frame
        elif len(tickers) == 1:
            frame = _normalize(batch)
            if frame is not None and len(frame):
                out[tickers[0]] = frame
    except Exception as exc:  # pragma: no cover - network dependent
        log.warning("batch download failed (%s); falling back to per-ticker", exc)

    for ticker in [t for t in tickers if t not in out]:
        try:
            frame = _normalize(
                yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
            )
            if frame is not None and len(frame):
                out[ticker] = frame
        except Exception as exc:  # pragma: no cover
            log.debug("per-ticker download failed for %s: %s", ticker, exc)
    return out


def _fetch_yfinance(
    tickers: list[str],
    period: str,
    interval: str,
    batch_size: int = 40,
    pause: float = 0.5,
) -> dict[str, pd.DataFrame]:
    """Chunked batch download.

    A single ``yf.download`` covering a whole discovered universe is both slow
    and prone to partial failure, so requests are split into chunks with a
    short pause between them.
    """
    out: dict[str, pd.DataFrame] = {}
    size = max(1, int(batch_size))
    chunks = [tickers[i : i + size] for i in range(0, len(tickers), size)]
    for i, chunk in enumerate(chunks):
        out.update(_fetch_chunk(chunk, period, interval))
        if pause and i < len(chunks) - 1:
            time.sleep(pause)
    return out


def _fetch_with_retry(
    tickers: list[str],
    period: str,
    interval: str,
    retry_periods: list[str],
    batch_size: int,
    pause: float,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Fetch, then retry any empty results with progressively wider windows.

    Widening only helps when the ticker *has* history outside the requested
    window (recent listings, transient gaps). It cannot help a delisted or
    invalid symbol, which is why the caller must still handle a hard failure.
    """
    out = _fetch_yfinance(tickers, period, interval, batch_size, pause)
    missing = [t for t in tickers if t not in out]

    for retry_period in retry_periods or []:
        if not missing:
            break
        log.info("retrying %d ticker(s) with period=%s", len(missing), retry_period)
        wider = _fetch_yfinance(missing, str(retry_period), interval, batch_size, pause)
        for ticker, frame in wider.items():
            out[ticker] = _trim_to_period(frame, period, interval)
        missing = [t for t in tickers if t not in out]

    return out, missing


def fetch_prices(
    tickers: list[str],
    period: str = "1y",
    interval: str = "1d",
    use_cache: bool = True,
    synthetic: bool = False,
    retry_periods: list[str] | None = None,
    on_missing: str = "drop",
    batch_size: int = 40,
    batch_pause_seconds: float = 0.5,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Return ({ticker: OHLCV df}, errors).

    ``synthetic=True`` forces generated data for every ticker (offline tests and
    the UI's offline toggle).

    ``on_missing`` controls what happens when a live ticker yields nothing even
    after the retry ladder:

    * ``"drop"`` (default) — omit the ticker and record an error. The returned
      mapping then contains only real data, so a screen can never silently mix
      fabricated prices into real results.
    * ``"synthetic"`` — fabricate prices as a last resort. Smoke runs only.
    """
    errors: list[str] = []
    result: dict[str, pd.DataFrame] = {}

    if synthetic:
        for ticker in tickers:
            result[ticker] = synthetic_prices(ticker, period, interval)
        return result, errors

    for ticker in tickers:
        cached = _load_cache(f"{ticker}|{period}|{interval}") if use_cache else None
        if cached is not None and len(cached):
            result[ticker] = cached

    remaining = [t for t in tickers if t not in result]

    if remaining:
        retries = list(retry_periods) if retry_periods is not None else ["2y", "5y", "max"]
        fetched, missing = _fetch_with_retry(
            remaining, period, interval, retries, batch_size, batch_pause_seconds
        )
        for ticker, frame in fetched.items():
            result[ticker] = frame
            if use_cache:
                _save_cache(f"{ticker}|{period}|{interval}", frame)

        use_synthetic = str(on_missing).strip().lower() == "synthetic"
        for ticker in missing:
            if use_synthetic:
                result[ticker] = synthetic_prices(ticker, period, interval)
                errors.append(f"{ticker}: no live data after retries — using synthetic data")
            else:
                errors.append(f"{ticker}: no live data after retries — dropped")

    return result, errors


def last_trading_day() -> date:
    today = date.today()
    if today.weekday() == 5:
        return today - timedelta(days=1)
    if today.weekday() == 6:
        return today - timedelta(days=2)
    return today
