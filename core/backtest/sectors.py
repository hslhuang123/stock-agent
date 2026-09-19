"""Sector lookup for attribution.

Sectors come from a static map for the default universe (fast and reliable),
with a best-effort yfinance fallback for anything else. Results are cached to
``data/sectors.json``.

Sector labels follow GICS-style buckets.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from core.config import DATA_DIR

log = logging.getLogger(__name__)

# Static map for the default universe + common names. Keeps attribution fast
# and deterministic (no network dependency for the core use case).
STATIC_SECTORS: dict[str, str] = {
    # Information Technology
    "AAPL": "Information Technology", "MSFT": "Information Technology",
    "NVDA": "Information Technology", "AMD": "Information Technology",
    "AVGO": "Information Technology", "CRM": "Information Technology",
    "ADBE": "Information Technology", "ORCL": "Information Technology",
    "INTC": "Information Technology", "QCOM": "Information Technology",
    "MU": "Information Technology", "AMAT": "Information Technology",
    "LRCX": "Information Technology", "PANW": "Information Technology",
    "NOW": "Information Technology", "PLTR": "Information Technology",
    "SNOW": "Information Technology", "DDOG": "Information Technology",
    "CRWD": "Information Technology", "NET": "Information Technology",
    "ACN": "Information Technology", "IBM": "Information Technology",
    "CSCO": "Information Technology", "TXN": "Information Technology",
    "INTU": "Information Technology",
    # Communication Services
    "GOOGL": "Communication Services", "GOOG": "Communication Services",
    "META": "Communication Services", "NFLX": "Communication Services",
    "DIS": "Communication Services", "T": "Communication Services",
    "VZ": "Communication Services", "TMUS": "Communication Services",
    "T.TO": "Communication Services",
    # Consumer Discretionary
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
    "SHOP": "Consumer Discretionary", "HD": "Consumer Discretionary",
    "MCD": "Consumer Discretionary", "NKE": "Consumer Discretionary",
    "SBUX": "Consumer Discretionary", "LOW": "Consumer Discretionary",
    "BKNG": "Consumer Discretionary",
    # Consumer Staples
    "WMT": "Consumer Staples", "COST": "Consumer Staples",
    "KO": "Consumer Staples", "PEP": "Consumer Staples",
    "PG": "Consumer Staples", "PM": "Consumer Staples",
    # Financials
    "JPM": "Financials", "BAC": "Financials", "GS": "Financials",
    "MS": "Financials", "V": "Financials", "MA": "Financials",
    "AXP": "Financials", "PYPL": "Financials", "COIN": "Financials",
    "XYZ": "Financials", "SQ": "Financials", "BLK": "Financials",
    "SCHW": "Financials", "C": "Financials", "WFC": "Financials",
    # Health Care
    "UNH": "Health Care", "LLY": "Health Care", "JNJ": "Health Care",
    "PFE": "Health Care", "MRK": "Health Care", "ABBV": "Health Care",
    "TMO": "Health Care", "ABT": "Health Care", "DHR": "Health Care",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy",
    "CVE.TO": "Energy",
    # Industrials
    "CAT": "Industrials", "DE": "Industrials", "BA": "Industrials",
    "GE": "Industrials", "HON": "Industrials", "UPS": "Industrials",
    "RTX": "Industrials", "LMT": "Industrials", "UBER": "Industrials",
    # Materials / Utilities / Real Estate
    "LIN": "Materials", "SHW": "Materials", "NEE": "Utilities",
    "DUK": "Utilities", "AMT": "Real Estate", "PLD": "Real Estate",
}

CACHE_PATH = DATA_DIR / "sectors.json"


def _load_cache() -> dict[str, str]:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:  # pragma: no cover
            return {}
    return {}


def _save_cache(data: dict[str, str]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        CACHE_PATH.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    except Exception:  # pragma: no cover
        log.debug("could not write sector cache")


def _fetch_sector(ticker: str) -> str | None:
    try:
        import yfinance as yf

        info = yf.Ticker(ticker).info or {}
        sector = info.get("sector")
        return str(sector) if sector else None
    except Exception:
        return None


def get_sectors(tickers: list[str], allow_network: bool = True) -> dict[str, str]:
    """Return {ticker: sector}, falling back to 'Unknown'."""
    cache = _load_cache()
    out: dict[str, str] = {}
    missing: list[str] = []

    for ticker in tickers:
        if ticker in STATIC_SECTORS:
            out[ticker] = STATIC_SECTORS[ticker]
        elif ticker in cache:
            out[ticker] = cache[ticker]
        else:
            missing.append(ticker)

    if missing and allow_network:
        fetched: dict[str, str] = {}
        for ticker in missing:
            sector = _fetch_sector(ticker)
            if sector:
                fetched[ticker] = sector
        if fetched:
            cache.update(fetched)
            _save_cache(cache)
        for ticker in missing:
            out[ticker] = fetched.get(ticker, "Unknown")

    for ticker in missing:
        out.setdefault(ticker, "Unknown")

    return out


def sector_weights(picks: list[str], sectors: dict[str, str]) -> dict[str, float]:
    """Equal-weight sector exposure of a pick list."""
    if not picks:
        return {}
    counts: dict[str, int] = {}
    for t in picks:
        counts[sectors.get(t, "Unknown")] = counts.get(sectors.get(t, "Unknown"), 0) + 1
    n = len(picks)
    return {s: c / n for s, c in counts.items()}
