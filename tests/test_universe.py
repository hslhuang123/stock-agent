"""Dynamic universe discovery, point-in-time reconstruction and ingest policy."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import core.ingest.prices as prices_mod
from core.backtest.engine import BacktestConfig, run_backtest
from core.backtest.robustness import active_by_period_for, build_panel, evaluate_panel
from core.config import load_settings
from core.features.indicators import add_indicators
from core.ingest.prices import fetch_prices, synthetic_prices
from core.universe import resolve_universe
from core.universe.discover import _normalize_company, _rows_to_result
from core.universe.point_in_time import (
    dollar_volume_frame,
    liquidity_ranking,
    liquidity_universe_at,
)


@pytest.fixture(scope="module")
def settings():
    return load_settings()


# --------------------------------------------------------------------------- #
# Point-in-time reconstruction — no look-ahead
# --------------------------------------------------------------------------- #

def test_point_in_time_universe_ignores_future_volume():
    """Tampering with bars after `as_of` must not change that date's universe."""
    tickers = [f"T{i}" for i in range(8)]
    prices = {t: synthetic_prices(t, "2y") for t in tickers}
    as_of = prices["T0"].index[300]

    before = liquidity_universe_at(prices, as_of, top_n=3)

    # Make one ticker wildly liquid, but only *after* the decision date.
    tampered = {}
    for t in tickers:
        df = prices[t].copy()
        df.loc[df.index > as_of, "Volume"] = 1e12 if t == "T7" else 1.0
        tampered[t] = df

    after = liquidity_universe_at(tampered, as_of, top_n=3)
    assert before == after, "universe at a past date changed when future data changed"


def test_future_volume_does_affect_later_dates():
    """Sanity check that the previous test isn't passing for the wrong reason."""
    tickers = [f"T{i}" for i in range(8)]
    prices = {t: synthetic_prices(t, "2y") for t in tickers}
    later = prices["T0"].index[-1]

    baseline = liquidity_universe_at(prices, later, top_n=3)
    boosted = {t: df.copy() for t, df in prices.items()}
    boosted["T7"]["Volume"] = boosted["T7"]["Volume"] * 1e9

    assert liquidity_universe_at(boosted, later, top_n=3) != baseline


def test_liquidity_ranking_excludes_illiquid_and_short_history():
    prices = {t: synthetic_prices(t, "1y") for t in ["AAA", "BBB", "CCC"]}
    prices["CCC"]["Volume"] = 1.0
    as_of = prices["AAA"].index[-1]

    ranking = liquidity_ranking(prices, as_of, lookback_days=20)
    assert "CCC" not in ranking or ranking["CCC"] < ranking["AAA"]

    top = liquidity_universe_at(prices, as_of, top_n=2)
    assert "CCC" not in top
    assert len(top) == 2

    # A ticker with fewer bars than the window is not eligible.
    short = {"NEW": prices["AAA"].tail(3)}
    assert liquidity_universe_at(short, short["NEW"].index[-1], top_n=5) == []


def test_panel_point_in_time_matches_engine(settings):
    """The fast evaluator must agree with the engine under a dynamic universe."""
    settings = dict(settings)
    tickers = [f"P{i:02d}" for i in range(40)]
    settings["universe"] = tickers + ["SPY"]
    settings["screening"] = dict(settings["screening"], min_avg_dollar_volume=0)
    settings["universe_config"] = {
        **settings["universe_config"],
        "backtest": {"mode": "point_in_time_liquidity", "pool_size": 15, "lookback_days": 20},
    }
    config = BacktestConfig(lookback="3y", rebalance_days=21, holding_days=21, top_n=5, cost_bps=10)

    reference = run_backtest(settings=settings, config=config, synthetic=True, use_cache=False)

    prices, _ = fetch_prices(["SPY"] + tickers, period="3y", synthetic=True, use_cache=False)
    frames = {t: (df if "sma20" in df.columns else add_indicators(df)) for t, df in prices.items()}
    panel = build_panel(frames, "SPY", config, tickers)
    active = active_by_period_for(
        panel, dollar_volume_frame(frames), pool_size=15, lookback_days=20, benchmark="SPY"
    )

    # The point-in-time universe must actually be a subset (i.e. it did something).
    assert any(len(s) < len(tickers) for s in active)

    fast = evaluate_panel(
        panel, settings["scoring_weights"], settings["screening"],
        top_n=5, cost_bps=10.0, weight_mode="equal", active_by_period=active,
    )

    ref_net = [p["net_return"] for p in reference["periods"]]
    assert len(fast["net_returns"]) == len(ref_net)
    np.testing.assert_allclose(fast["net_returns"], ref_net, rtol=0, atol=1e-4)


# --------------------------------------------------------------------------- #
# Discovery (offline: row filtering and de-duplication)
# --------------------------------------------------------------------------- #

def test_discovery_dedupes_share_classes_and_drops_non_equities():
    rows = [
        {"symbol": "GOOGL", "shortName": "Alphabet Inc.", "quoteType": "EQUITY"},
        {"symbol": "GOOG", "shortName": "Alphabet Inc.", "quoteType": "EQUITY"},
        {"symbol": "AAPL", "shortName": "Apple Inc.", "quoteType": "EQUITY"},
        {"symbol": "SPY", "shortName": "SPDR S&P 500 ETF", "quoteType": "ETF"},
        {"symbol": "MSFT", "shortName": "Microsoft Corporation", "quoteType": "EQUITY"},
    ]
    result = _rows_to_result(rows, {"max_results": 10, "dedupe_companies": True}, 4, [])
    assert result.tickers == ["GOOGL", "AAPL", "MSFT"]
    assert result.names["GOOGL"] == "Alphabet Inc."


def test_discovery_respects_max_results():
    rows = [
        {"symbol": f"S{i}", "shortName": f"Company {i}", "quoteType": "EQUITY"}
        for i in range(10)
    ]
    result = _rows_to_result(rows, {"max_results": 3, "dedupe_companies": True}, 10, [])
    assert len(result.tickers) == 3


def test_normalize_company_strips_class_suffixes():
    assert _normalize_company("Alphabet Inc. Class A") == _normalize_company("Alphabet Inc. Class C")
    assert _normalize_company("Apple Inc.") != _normalize_company("Microsoft Corporation")


def test_cached_discovery_respects_max_results(monkeypatch, tmp_path):
    """A smaller request must trim the *result*, not the cache."""
    import core.universe.discover as disc

    monkeypatch.setattr(disc, "CACHE_PATH", tmp_path / "discovered.json")
    disc.save_discovered(
        disc.DiscoveryResult(
            tickers=[f"S{i:02d}" for i in range(50)],
            source="screener",
            discovered_at="2020-01-01T00:00:00Z",
        )
    )

    settings = {"universe_config": {"discovery": {}, "max_results": 10}}
    result = disc.discover_universe(settings, max_results=10, allow_network=False)

    assert len(result.tickers) == 10
    assert len(disc.load_discovered().tickers) == 50, "cache was shrunk by a trimmed result"


def test_max_results_above_screener_limit_is_clamped(monkeypatch, tmp_path):
    """Yahoo rejects size > 250 outright; we clamp instead of failing the pass."""
    import core.universe.discover as disc

    monkeypatch.setattr(disc, "CACHE_PATH", tmp_path / "discovered.json")
    settings = {"universe_config": {"discovery": {}, "max_results": 1000}}

    result = disc.discover_universe(settings, max_results=1000, allow_network=False)

    assert result.gates["max_results"] == disc.SCREENER_MAX_RESULTS
    assert any("exceeds the screener limit" in e for e in result.errors)


def test_resolve_does_not_write_back_a_trimmed_cache(monkeypatch, tmp_path):
    import core.universe.discover as disc

    monkeypatch.setattr(disc, "CACHE_PATH", tmp_path / "discovered.json")
    disc.save_discovered(
        disc.DiscoveryResult(
            tickers=[f"S{i:02d}" for i in range(50)],
            source="screener",
            discovered_at="2020-01-01T00:00:00Z",
        )
    )

    settings = {
        "benchmark": "SPY",
        "universe": [],
        "universe_config": {
            "mode": "dynamic",
            "discovery": {},
            "max_results": 10,
            "cache_minutes": 10_000,
        },
    }
    resolution = resolve_universe(settings, mode="dynamic", allow_network=True, use_cache=True)

    assert len(resolution.tickers) == 10
    assert len(disc.load_discovered().tickers) == 50


# --------------------------------------------------------------------------- #
# resolve_universe
# --------------------------------------------------------------------------- #

def test_resolve_static_returns_configured_list(settings):
    resolution = resolve_universe(settings, mode="static")
    assert resolution.mode == "static"
    assert resolution.source == "static"
    assert set(resolution.tickers) == {t.upper() for t in settings["universe"]}
    assert "SPY" not in resolution.tickers


def test_resolve_removes_benchmark_and_flags_it(settings):
    settings = dict(settings)
    settings["universe"] = ["AAPL", "SPY"]
    resolution = resolve_universe(settings, mode="static")
    assert resolution.tickers == ["AAPL"]
    assert resolution.benchmark_excluded is True


def test_resolve_dynamic_without_network_falls_back(settings):
    resolution = resolve_universe(settings, mode="dynamic", allow_network=False, use_cache=False)
    assert resolution.source == "static-fallback"
    assert resolution.tickers, "should still return a usable universe"
    assert any("discovery" in e for e in resolution.errors)


def test_resolve_hybrid_unions_seed(settings):
    settings = dict(settings)
    settings["universe"] = ["ZZZZ"]
    resolution = resolve_universe(settings, mode="hybrid", allow_network=False, use_cache=False)
    # Discovery fails offline, so the seed must survive the union.
    assert "ZZZZ" in resolution.tickers


# --------------------------------------------------------------------------- #
# Ingest policy: retry ladder, drop vs synthetic
# --------------------------------------------------------------------------- #

def test_retry_ladder_widens_then_trims(monkeypatch):
    calls: list[str] = []

    def fake(tickers, period, interval, batch_size=40, pause=0.5):
        calls.append(period)
        if period == "5y":
            return {t: synthetic_prices(t, "5y") for t in tickers}
        return {}

    monkeypatch.setattr(prices_mod, "_fetch_yfinance", fake)
    result, errors = fetch_prices(
        ["ZZZ"], period="1y", use_cache=False, retry_periods=["2y", "5y"]
    )

    assert calls == ["1y", "2y", "5y"]
    assert "ZZZ" in result
    # The wider fetch is trimmed back to the requested window.
    assert len(result["ZZZ"]) <= prices_mod._period_to_days("1y")
    assert errors == []


def test_missing_drops_by_default(monkeypatch):
    monkeypatch.setattr(prices_mod, "_fetch_yfinance", lambda *a, **k: {})
    result, errors = fetch_prices(["ZZZ"], use_cache=False, retry_periods=[])
    assert result == {}
    assert any("dropped" in e for e in errors)


def test_missing_can_fabricate_when_explicitly_allowed(monkeypatch):
    monkeypatch.setattr(prices_mod, "_fetch_yfinance", lambda *a, **k: {})
    result, errors = fetch_prices(
        ["ZZZ"], use_cache=False, retry_periods=[], on_missing="synthetic"
    )
    assert "ZZZ" in result
    assert any("synthetic" in e for e in errors)


def test_synthetic_mode_never_touches_the_network(monkeypatch):
    def explode(*_a, **_k):  # pragma: no cover - must not run
        raise AssertionError("network access attempted in synthetic mode")

    monkeypatch.setattr(prices_mod, "_fetch_yfinance", explode)
    result, errors = fetch_prices(["AAA", "BBB"], synthetic=True)
    assert set(result) == {"AAA", "BBB"}
    assert errors == []
