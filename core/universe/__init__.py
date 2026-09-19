"""Universe resolution: static, dynamically discovered, or hybrid.

Single entry point — :func:`resolve_universe` — so the live pipeline and the
backtest can never drift apart on what "the universe" means.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from core.ingest.company_names import get_company_names
from core.universe.discover import (
    CACHE_PATH,
    DiscoveryResult,
    discover_universe,
    load_discovered,
    save_discovered,
)
from core.universe.point_in_time import (
    dollar_volume_frame,
    liquidity_ranking,
    liquidity_universe_at,
    ranking_from_frame,
    universe_at_from_frame,
)

log = logging.getLogger(__name__)

VALID_MODES = ("static", "dynamic", "hybrid")

__all__ = [
    "DiscoveryResult",
    "UniverseResolution",
    "apply_universe_overrides",
    "discover_universe",
    "dollar_volume_frame",
    "load_discovered",
    "liquidity_ranking",
    "liquidity_universe_at",
    "ranking_from_frame",
    "resolve_universe",
    "save_discovered",
    "static_universe",
    "universe_at_from_frame",
    "CACHE_PATH",
]


@dataclass
class UniverseResolution:
    """The resolved tradeable universe plus provenance for reporting."""

    tickers: list[str] = field(default_factory=list)
    names: dict[str, str] = field(default_factory=dict)
    mode: str = "static"
    source: str = "static"
    benchmark: str = "SPY"
    benchmark_excluded: bool = False
    discovered_at: str | None = None
    total_available: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "source": self.source,
            "benchmark": self.benchmark,
            "benchmark_excluded": self.benchmark_excluded,
            "discovered_at": self.discovered_at,
            "total_available": self.total_available,
            "size": len(self.tickers),
            "tickers": self.tickers,
        }


def static_universe(settings: dict[str, Any]) -> list[str]:
    """The configured seed list, de-duplicated and upper-cased."""
    raw = settings.get("universe") or []
    return list(dict.fromkeys(str(t).strip().upper() for t in raw if str(t).strip()))


def apply_universe_overrides(
    settings: dict[str, Any],
    mode: str | None = None,
    max_results: int | None = None,
    backtest_mode: str | None = None,
) -> dict[str, Any]:
    """Return a copy of ``settings`` with universe overrides applied.

    Used by the CLI scripts so ``--universe-mode dynamic`` does not require
    editing ``config/settings.yaml``.
    """
    if mode is None and max_results is None and backtest_mode is None:
        return settings

    config = dict(settings.get("universe_config", {}) or {})
    if mode:
        resolved = str(mode).strip().lower()
        if resolved not in VALID_MODES:
            raise ValueError(f"unknown universe mode {mode!r}; expected one of {VALID_MODES}")
        config["mode"] = resolved
    if max_results is not None:
        config["max_results"] = int(max_results)
    if backtest_mode:
        backtest = dict(config.get("backtest", {}) or {})
        backtest["mode"] = str(backtest_mode).strip().lower()
        config["backtest"] = backtest
    return {**settings, "universe_config": config}


def resolve_universe(
    settings: dict[str, Any],
    mode: str | None = None,
    allow_network: bool = True,
    use_cache: bool = True,
) -> UniverseResolution:
    """Resolve the tradeable universe according to ``universe_config.mode``.

    The benchmark is always removed from the returned tickers (it is fetched
    separately for relative strength and regime) and reported via
    ``benchmark_excluded``.

    Fails soft: if discovery is unavailable the previously cached result is
    used, and failing that the static list, with the reason recorded in
    ``errors`` rather than raising.
    """
    cfg = settings.get("universe_config", {}) or {}
    resolved_mode = str(mode or cfg.get("mode") or "static").strip().lower()
    if resolved_mode not in VALID_MODES:
        resolved_mode = "static"

    benchmark = str(settings.get("benchmark", "SPY")).strip().upper()
    seed = static_universe(settings)

    errors: list[str] = []
    names: dict[str, str] = {}
    tickers: list[str] = []
    source = resolved_mode
    discovered_at: str | None = None
    total_available = 0

    if resolved_mode in ("dynamic", "hybrid"):
        result = discover_universe(
            settings, allow_network=allow_network, use_cache=use_cache
        )
        errors.extend(result.errors)

        if result.tickers:
            tickers = list(result.tickers)
            names.update(result.names)
            source = result.source
            discovered_at = result.discovered_at
            total_available = result.total_available
            # Persistence is handled inside discover_universe, which only writes
            # genuine screener results (a cache-derived result may be trimmed by
            # `max_results`, and writing that back would shrink the cache).
        else:
            cached = load_discovered() if use_cache else None
            if cached and cached.tickers:
                tickers = list(cached.tickers)
                names.update(cached.names)
                source = "cache"
                discovered_at = cached.discovered_at
                total_available = cached.total_available
                errors.append("discovery unavailable — using cached universe")
            else:
                tickers = list(seed)
                source = "static-fallback"
                errors.append(
                    "discovery unavailable and no cache — falling back to the static universe"
                )

        if resolved_mode == "hybrid":
            tickers = list(dict.fromkeys(seed + tickers))
            source = f"hybrid:{source}"
    else:
        tickers = list(seed)

    # Names for seed/static entries resolve offline; discovered names were
    # already returned by the screener at no extra cost.
    offline_names = get_company_names(tickers, allow_network=False)
    for ticker in tickers:
        names.setdefault(ticker, offline_names.get(ticker, ticker))

    benchmark_excluded = benchmark in tickers
    tickers = [t for t in tickers if t != benchmark]

    if not tickers:
        errors.append("universe resolved empty — check universe_config and the seed list")

    return UniverseResolution(
        tickers=tickers,
        names=names,
        mode=resolved_mode,
        source=source,
        benchmark=benchmark,
        benchmark_excluded=benchmark_excluded,
        discovered_at=discovered_at,
        total_available=total_available,
        errors=errors,
    )
