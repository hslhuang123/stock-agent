"""Company name lookup.

Names come from a static seed map for the default universe (fast, offline-safe),
with a best-effort yfinance fallback for anything else. Results cache to
``data/company_names.json`` so repeated runs never re-hit the network.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from core.config import DATA_DIR

log = logging.getLogger(__name__)

# Seed map for the default universe + common names. Treated as a warm cache;
# yfinance fills in anything missing.
STATIC_NAMES: dict[str, str] = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "NVDA": "NVIDIA Corporation",
    "AMZN": "Amazon.com, Inc.",
    "GOOGL": "Alphabet Inc.",
    "GOOG": "Alphabet Inc.",
    "META": "Meta Platforms, Inc.",
    "TSLA": "Tesla, Inc.",
    "AMD": "Advanced Micro Devices, Inc.",
    "AVGO": "Broadcom Inc.",
    "NFLX": "Netflix, Inc.",
    "CRM": "Salesforce, Inc.",
    "ADBE": "Adobe Inc.",
    "ORCL": "Oracle Corporation",
    "INTC": "Intel Corporation",
    "QCOM": "QUALCOMM Incorporated",
    "MU": "Micron Technology, Inc.",
    "AMAT": "Applied Materials, Inc.",
    "LRCX": "Lam Research Corporation",
    "PANW": "Palo Alto Networks, Inc.",
    "NOW": "ServiceNow, Inc.",
    "UBER": "Uber Technologies, Inc.",
    "SHOP": "Shopify Inc.",
    "XYZ": "Block, Inc.",
    "SQ": "Block, Inc.",
    "PYPL": "PayPal Holdings, Inc.",
    "COIN": "Coinbase Global, Inc.",
    "PLTR": "Palantir Technologies Inc.",
    "SNOW": "Snowflake Inc.",
    "DDOG": "Datadog, Inc.",
    "CRWD": "CrowdStrike Holdings, Inc.",
    "NET": "Cloudflare, Inc.",
    "JPM": "JPMorgan Chase & Co.",
    "BAC": "Bank of America Corporation",
    "GS": "The Goldman Sachs Group, Inc.",
    "MS": "Morgan Stanley",
    "V": "Visa Inc.",
    "MA": "Mastercard Incorporated",
    "AXP": "American Express Company",
    "UNH": "UnitedHealth Group Incorporated",
    "LLY": "Eli Lilly and Company",
    "JNJ": "Johnson & Johnson",
    "PFE": "Pfizer Inc.",
    "MRK": "Merck & Co., Inc.",
    "ABBV": "AbbVie Inc.",
    "XOM": "Exxon Mobil Corporation",
    "CVX": "Chevron Corporation",
    "CAT": "Caterpillar Inc.",
    "DE": "Deere & Company",
    "BA": "The Boeing Company",
    "GE": "GE Aerospace",
    "HON": "Honeywell International Inc.",
    "WMT": "Walmart Inc.",
    "COST": "Costco Wholesale Corporation",
    "HD": "The Home Depot, Inc.",
    "MCD": "McDonald's Corporation",
    "NKE": "NIKE, Inc.",
    "DIS": "The Walt Disney Company",
    "KO": "The Coca-Cola Company",
    "PEP": "PepsiCo, Inc.",
    # common extras
    "ACN": "Accenture plc", "IBM": "International Business Machines Corporation",
    "CSCO": "Cisco Systems, Inc.", "TXN": "Texas Instruments Incorporated",
    "INTU": "Intuit Inc.", "T": "AT&T Inc.", "VZ": "Verizon Communications Inc.",
    "TMUS": "T-Mobile US, Inc.", "SBUX": "Starbucks Corporation",
    "LOW": "Lowe's Companies, Inc.", "BKNG": "Booking Holdings Inc.",
    "PG": "The Procter & Gamble Company", "PM": "Philip Morris International Inc.",
    "BLK": "BlackRock, Inc.", "SCHW": "The Charles Schwab Corporation",
    "C": "Citigroup Inc.", "WFC": "Wells Fargo & Company",
    "TMO": "Thermo Fisher Scientific Inc.", "ABT": "Abbott Laboratories",
    "DHR": "Danaher Corporation", "COP": "ConocoPhillips",
    "SLB": "Schlumberger Limited", "UPS": "United Parcel Service, Inc.",
    "RTX": "RTX Corporation", "LMT": "Lockheed Martin Corporation",
    "LIN": "Linde plc", "SHW": "The Sherwin-Williams Company",
    "NEE": "NextEra Energy, Inc.", "DUK": "Duke Energy Corporation",
    "AMT": "American Tower Corporation", "PLD": "Prologis, Inc.",
    "T.TO": "TELUS Corporation", "CVE.TO": "Cenovus Energy Inc.",
}

CACHE_PATH = DATA_DIR / "company_names.json"


def _load_cache() -> dict[str, str]:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:  # pragma: no cover - corrupt cache
            return {}
    return {}


def _save_cache(data: dict[str, str]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        CACHE_PATH.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    except Exception:  # pragma: no cover
        log.debug("could not write company-name cache")


def _fetch_name(ticker: str) -> str | None:
    """Best-effort name via yfinance. Returns None on any failure."""
    try:
        import yfinance as yf

        info = yf.Ticker(ticker).info or {}
    except Exception:
        return None
    for key in ("shortName", "longName", "displayName", "name"):
        value = info.get(key)
        if value:
            return str(value)
    return None


def get_company_names(
    tickers: list[str],
    allow_network: bool = True,
    persist: bool = True,
) -> dict[str, str]:
    """Return ``{ticker: company name}``, falling back to the ticker itself."""
    cache = {**STATIC_NAMES, **_load_cache()}
    out: dict[str, str] = {}
    missing: list[str] = []

    for ticker in tickers:
        if ticker in cache:
            out[ticker] = cache[ticker]
        else:
            missing.append(ticker)

    fetched: dict[str, str] = {}
    if missing and allow_network:
        for ticker in missing:
            name = _fetch_name(ticker)
            if name:
                fetched[ticker] = name
        if fetched and persist:
            merged = _load_cache()
            merged.update(fetched)
            _save_cache(merged)

    for ticker in missing:
        out[ticker] = fetched.get(ticker, ticker)

    return out


def get_company_name(ticker: str, allow_network: bool = True) -> str:
    return get_company_names([ticker], allow_network=allow_network).get(ticker, ticker)
