"""Dynamic universe discovery via the Yahoo Finance equity screener.

The screener returns *today's* eligible names — it has no history. That makes it
correct for the live screen and unusable for a historical backtest, which must
rebuild its universe point-in-time (see :mod:`core.universe.point_in_time`).

Network access is confined to :func:`discover_universe`; everything else in this
module is pure and offline-testable.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from core.config import DATA_DIR

log = logging.getLogger(__name__)

CACHE_PATH = DATA_DIR / "universe" / "discovered.json"

# Yahoo rejects a screener query larger than this. Offset pagination is not
# implemented, so a larger pool cannot be discovered in one pass.
SCREENER_MAX_RESULTS = 250

# Suffixes that differentiate share classes of the same company. Stripped
# before name-based de-duplication so GOOG/GOOGL collapse to one entry.
_CLASS_SUFFIX = re.compile(
    r"\s*[-–,]?\s*(class\s+[a-z]|series\s+[a-z]|cl\s+[a-z]|"
    r"common\s+stock|ordinary\s+shares?|depositary\s+shares?)\s*$",
    re.IGNORECASE,
)


def _normalize_company(name: str) -> str:
    """Lowercase, strip punctuation and share-class suffixes for de-duplication."""
    cleaned = _CLASS_SUFFIX.sub("", str(name)).strip().lower()
    return re.sub(r"[^a-z0-9]+", "", cleaned)


@dataclass
class DiscoveryResult:
    """Outcome of a screener discovery pass."""

    tickers: list[str] = field(default_factory=list)
    names: dict[str, str] = field(default_factory=dict)
    total_available: int = 0
    gates: dict[str, Any] = field(default_factory=dict)
    screeners: list[str] = field(default_factory=list)
    discovered_at: str = ""
    source: str = "screener"
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "discovered_at": self.discovered_at,
            "source": self.source,
            "total_available": self.total_available,
            "gates": self.gates,
            "screeners": self.screeners,
            "tickers": self.tickers,
            "names": self.names,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DiscoveryResult":
        return cls(
            tickers=list(data.get("tickers", [])),
            names=dict(data.get("names", {})),
            total_available=int(data.get("total_available", 0)),
            gates=dict(data.get("gates", {})),
            screeners=list(data.get("screeners", [])),
            discovered_at=str(data.get("discovered_at", "")),
            source=str(data.get("source", "cache")),
        )


def _build_query(discovery: dict[str, Any]) -> Any:
    """Translate the configured gates into a yfinance ``EquityQuery``.

    Returns ``None`` when no gates are configured (caller should then rely on
    screeners alone, or fall back to a static universe).
    """
    from yfinance import EquityQuery as Q

    clauses: list[Any] = []

    region = discovery.get("region")
    if region:
        clauses.append(Q("is-in", ["region", str(region)]))

    exchanges = [str(e) for e in (discovery.get("exchanges") or [])]
    if exchanges:
        clauses.append(Q("is-in", ["exchange", *exchanges]))

    min_price = discovery.get("min_price")
    if min_price:
        clauses.append(Q("gt", ["intradayprice", float(min_price)]))

    min_market_cap = discovery.get("min_market_cap")
    if min_market_cap:
        clauses.append(Q("gt", ["intradaymarketcap", float(min_market_cap)]))

    min_avg_volume = discovery.get("min_avg_volume_3m")
    if min_avg_volume:
        clauses.append(Q("gt", ["avgdailyvol3m", float(min_avg_volume)]))

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return Q("and", clauses)


def _collect_rows(discovery: dict[str, Any], max_results: int, errors: list[str]) -> tuple[list[dict], int]:
    """Run the gate query plus any configured screeners; merge the rows."""
    import yfinance as yf

    rows: list[dict] = []
    seen: set[str] = set()
    total_available = 0

    query = None
    try:
        query = _build_query(discovery)
    except Exception as exc:  # pragma: no cover - defensive
        errors.append(f"discovery query could not be built: {exc}")

    if query is not None:
        try:
            payload = yf.screen(
                query,
                size=max_results,
                sortField=discovery.get("sort_field", "intradaymarketcap"),
                sortAsc=bool(discovery.get("sort_asc", False)),
            )
            total_available = int(payload.get("total", 0) or 0)
            for row in payload.get("quotes") or []:
                symbol = row.get("symbol")
                if symbol and symbol not in seen:
                    seen.add(symbol)
                    rows.append(row)
        except Exception as exc:
            errors.append(f"screener query failed: {exc}")

    for screener in discovery.get("screeners") or []:
        try:
            payload = yf.screen(str(screener), size=max_results)
            if not total_available:
                total_available = int(payload.get("total", 0) or 0)
            added = 0
            for row in payload.get("quotes") or []:
                symbol = row.get("symbol")
                if symbol and symbol not in seen:
                    seen.add(symbol)
                    rows.append(row)
                    added += 1
            log.info("screener %s contributed %d new names", screener, added)
        except Exception as exc:
            errors.append(f"screener '{screener}' failed: {exc}")

    return rows, total_available


def _rows_to_result(
    rows: list[dict],
    discovery: dict[str, Any],
    total_available: int,
    errors: list[str],
) -> DiscoveryResult:
    """Filter, order and de-duplicate screener rows into a result."""
    max_results = int(discovery.get("max_results", len(rows)) or len(rows))
    dedupe = bool(discovery.get("dedupe_companies", True))

    # The screener is equity-only, but rows can still carry non-equity types.
    quotes = [
        r
        for r in rows
        if str(r.get("quoteType", "EQUITY")).upper() in {"EQUITY", ""}
        and r.get("symbol")
    ]

    tickers: list[str] = []
    names: dict[str, str] = {}
    seen_names: set[str] = set()

    for row in quotes:
        if len(tickers) >= max_results:
            break
        symbol = str(row["symbol"]).strip().upper()
        name = str(row.get("shortName") or row.get("longName") or symbol).strip()
        if dedupe:
            key = _normalize_company(name)
            if key and key in seen_names:
                continue
            seen_names.add(key)
        tickers.append(symbol)
        names[symbol] = name

    return DiscoveryResult(
        tickers=tickers,
        names=names,
        total_available=total_available,
        gates=dict(discovery),
        screeners=list(discovery.get("screeners") or []),
        discovered_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        source="screener",
        errors=errors,
    )


def _apply_max_results(result: DiscoveryResult, max_results: int) -> DiscoveryResult:
    """Trim a cached result to the requested size (cache may be larger)."""
    if max_results and len(result.tickers) > max_results:
        result.tickers = result.tickers[:max_results]
    return result


def discover_universe(
    settings: dict[str, Any],
    max_results: int | None = None,
    allow_network: bool = True,
    use_cache: bool = True,
) -> DiscoveryResult:
    """Discover the current eligible universe from the Yahoo screener.

    ``use_cache=False`` bypasses the discovery cache entirely (no read, no
    write) so a caller can force a real query.

    Never raises for network problems: on failure returns an empty result with
    ``errors`` populated, so callers can fall back to the static list or cache.
    """
    cfg = settings.get("universe_config", {}) or {}
    discovery = dict(cfg.get("discovery", {}) or {})
    if max_results is None:
        max_results = int(cfg.get("max_results", SCREENER_MAX_RESULTS))
    max_results = int(max_results)

    errors: list[str] = []
    # Yahoo rejects a query whose size exceeds SCREENER_MAX_RESULTS outright
    # rather than truncating, so clamp it here and say so rather than failing
    # the whole discovery pass.
    if max_results > SCREENER_MAX_RESULTS:
        errors.append(
            f"max_results {max_results} exceeds the screener limit of "
            f"{SCREENER_MAX_RESULTS} — clamped (offset pagination is not implemented)"
        )
        max_results = SCREENER_MAX_RESULTS
    discovery["max_results"] = max_results

    if not allow_network:
        if use_cache:
            cached = load_discovered()
            if cached is not None:
                cached.errors = errors + ["network disabled — using cached discovery"]
                return _apply_max_results(cached, max_results)
        return DiscoveryResult(
            gates=discovery,
            discovered_at="",
            source="unavailable",
            errors=errors + ["network disabled and no usable discovery cache"],
        )

    # Reuse a recently written cache so a UI refresh followed by a screen run
    # does not hit the screener twice.
    prefer_minutes = float(cfg.get("cache_minutes", 0) or 0)
    if use_cache and prefer_minutes > 0:
        age = cache_age_minutes()
        if age is not None and age <= prefer_minutes:
            cached = load_discovered()
            if cached is not None and cached.tickers:
                cached.source = "cache-fresh"
                cached.errors = errors + cached.errors
                return _apply_max_results(cached, max_results)

    rows, total_available = _collect_rows(discovery, max_results, errors)
    if not rows and errors:
        return DiscoveryResult(
            gates=discovery,
            discovered_at="",
            source="failed",
            errors=errors,
        )

    result = _rows_to_result(rows, discovery, total_available, errors)
    if use_cache:
        save_discovered(result)
    return result


def save_discovered(result: DiscoveryResult) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        CACHE_PATH.write_text(
            json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
    except Exception as exc:  # pragma: no cover - disk issues
        log.debug("could not write discovery cache: %s", exc)


def load_discovered() -> DiscoveryResult | None:
    if not CACHE_PATH.exists():
        return None
    try:
        return DiscoveryResult.from_dict(json.loads(CACHE_PATH.read_text(encoding="utf-8")))
    except Exception:  # pragma: no cover - corrupt cache
        return None


def cache_age_minutes() -> float | None:
    """Age of the discovery cache in minutes, or None when absent."""
    if not CACHE_PATH.exists():
        return None
    return (time.time() - CACHE_PATH.stat().st_mtime) / 60.0
