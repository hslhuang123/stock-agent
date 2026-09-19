"""Attribution tests: OLS/HAC correctness, sector neutralization, hysteresis."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.backtest.attribution import (
    aggregate_sector_weights,
    excess_stats,
    market_attribution,
    ols_nw,
    run_attribution_suite,
    sector_neutral_returns,
    turnover_variants,
)
from core.backtest.engine import BacktestConfig
from core.backtest.robustness import build_panel, evaluate_panel
from core.backtest.sectors import get_sectors, sector_weights
from core.config import load_settings
from core.features.indicators import add_indicators
from core.ingest.prices import fetch_prices


@pytest.fixture(scope="module")
def settings():
    s = load_settings()
    s["universe"] = [f"A{i:02d}" for i in range(30)] + ["SPY"]
    s["screening"] = dict(s["screening"], min_avg_dollar_volume=0)
    return s


def _frames(tickers, period="3y"):
    prices, _ = fetch_prices(tickers, period=period, synthetic=True, use_cache=False)
    return {t: (df if "sma20" in df.columns else add_indicators(df)) for t, df in prices.items()}


# --------------------------------------------------------------------------- #
# OLS / Newey-West
# --------------------------------------------------------------------------- #

def test_ols_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.04, 500)
    y = 0.005 + 1.5 * x + rng.normal(0, 0.001, 500)
    reg = ols_nw(y, np.column_stack([np.ones(len(x)), x]))
    assert reg["params"][0] == pytest.approx(0.005, abs=0.001)
    assert reg["params"][1] == pytest.approx(1.5, abs=0.05)
    assert reg["r2"] > 0.99
    assert reg["t"][1] > 10


def test_ols_alpha_zero_when_pure_beta():
    """A pure multiple of the benchmark must have ~zero alpha and beta == multiple."""
    rng = np.random.default_rng(1)
    bench = rng.normal(0.01, 0.05, 400)
    strat = 2.0 * bench
    reg = ols_nw(strat, np.column_stack([np.ones(len(bench)), bench]))
    assert reg["params"][0] == pytest.approx(0.0, abs=1e-10)
    assert reg["params"][1] == pytest.approx(2.0, abs=1e-8)


def test_ols_handles_short_sample():
    reg = ols_nw([0.01], np.column_stack([np.ones(1), [0.02]]))
    assert reg["params"] == [None, None]


def test_market_attribution_flags_beta_only():
    """Levered index exposure plus idiosyncratic noise, but no alpha."""
    rng = np.random.default_rng(2)
    bench = rng.normal(0.01, 0.05, 400)
    strat = 1.4 * bench + rng.normal(0, 0.01, 400)
    out = market_attribution(strat, bench, ppy=12)
    assert out["beta"] == pytest.approx(1.4, abs=0.05)
    assert out["r_squared"] > 0.9
    assert out["significant_alpha"] is False


def test_market_attribution_detects_real_alpha():
    rng = np.random.default_rng(3)
    bench = rng.normal(0.005, 0.04, 400)
    strat = 0.008 + 1.0 * bench + rng.normal(0, 0.005, 400)
    out = market_attribution(strat, bench, ppy=12)
    assert out["alpha_annualized"] > 0
    assert out["significant_alpha"] is True


def test_excess_stats_significance():
    rng = np.random.default_rng(4)
    strong = rng.normal(0.02, 0.02, 200)
    out = excess_stats(strong, ppy=12)
    assert out["significant"] is True
    noise = rng.normal(0.0, 0.05, 200)
    assert excess_stats(noise, ppy=12)["significant"] is False


# --------------------------------------------------------------------------- #
# Sectors
# --------------------------------------------------------------------------- #

def test_static_sector_map_covers_default_universe():
    settings = load_settings()
    sectors = get_sectors(settings["universe"], allow_network=False)
    unknown = [t for t, s in sectors.items() if s == "Unknown"]
    assert not unknown, f"unmapped tickers: {unknown}"


def test_sector_weights_sum_to_one():
    sectors = {"A": "Tech", "B": "Tech", "C": "Energy"}
    w = sector_weights(["A", "B", "C"], sectors)
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["Tech"] == pytest.approx(2 / 3)


def test_sector_neutral_matches_allocation(settings):
    tickers = [t for t in settings["universe"] if t != "SPY"]
    frames = _frames(["SPY"] + tickers)
    config = BacktestConfig(lookback="3y", rebalance_days=21, holding_days=21, top_n=5)
    panel = build_panel(frames, "SPY", config, tickers)
    ev = evaluate_panel(panel, settings["scoring_weights"], settings["screening"], top_n=5)
    sectors = {t: ("Tech" if i % 2 else "Energy") for i, t in enumerate(tickers)}

    rets = sector_neutral_returns(panel, ev, sectors)
    assert len(rets) == panel.n_periods
    # Sector benchmark should be somewhere in the neighbourhood of the strategy.
    assert -0.5 < float(np.mean(rets)) < 0.5


# --------------------------------------------------------------------------- #
# Hysteresis
# --------------------------------------------------------------------------- #

def test_hysteresis_reduces_turnover(settings):
    tickers = [t for t in settings["universe"] if t != "SPY"]
    frames = _frames(["SPY"] + tickers)
    config = BacktestConfig(lookback="3y", rebalance_days=21, holding_days=21, top_n=5)
    panel = build_panel(frames, "SPY", config, tickers)

    plain = evaluate_panel(panel, settings["scoring_weights"], settings["screening"], top_n=5)
    hyst = evaluate_panel(
        panel, settings["scoring_weights"], settings["screening"],
        top_n=5, selection="hysteresis", exit_multiple=2.5,
    )
    to_plain = np.mean([p["turnover"] for p in plain["periods"] if p["picks"]])
    to_hyst = np.mean([p["turnover"] for p in hyst["periods"] if p["picks"]])
    assert to_hyst <= to_plain


def test_hysteresis_still_fills_positions(settings):
    tickers = [t for t in settings["universe"] if t != "SPY"]
    frames = _frames(["SPY"] + tickers)
    config = BacktestConfig(lookback="3y", rebalance_days=21, holding_days=21, top_n=5)
    panel = build_panel(frames, "SPY", config, tickers)
    hyst = evaluate_panel(
        panel, settings["scoring_weights"], settings["screening"],
        top_n=5, selection="hysteresis", exit_multiple=1.5,
    )
    for p in hyst["periods"]:
        if p["eligible"]:
            assert len(p["picks"]) == 5


def test_eligible_is_populated(settings):
    tickers = [t for t in settings["universe"] if t != "SPY"]
    frames = _frames(["SPY"] + tickers)
    config = BacktestConfig(lookback="3y", rebalance_days=21, holding_days=21, top_n=5)
    panel = build_panel(frames, "SPY", config, tickers)
    ev = evaluate_panel(panel, settings["scoring_weights"], settings["screening"], top_n=5)
    assert any(p["eligible"] for p in ev["periods"])
    # Picks must always be a subset of eligible.
    for p in ev["periods"]:
        assert set(p["picks"]).issubset(set(p["eligible"]))


# --------------------------------------------------------------------------- #
# Suite
# --------------------------------------------------------------------------- #

def test_turnover_variants_shape(settings):
    tickers = [t for t in settings["universe"] if t != "SPY"]
    frames = _frames(["SPY"] + tickers)
    config = BacktestConfig(lookback="3y", top_n=5)
    rows = turnover_variants(
        frames, "SPY", tickers, settings["scoring_weights"], settings["screening"], config,
        top_n=5, rebalance_list=(21,), cost_grid=(10.0, 50.0),
        variants=(("topn", 0.0), ("hysteresis", 2.0)),
    )
    assert len(rows) == 2
    for row in rows:
        assert set(row["edge_by_cost"]) == {"10", "50"}
        assert row["avg_turnover"] is not None


def test_attribution_suite_offline(settings):
    result = run_attribution_suite(
        settings=settings,
        config=BacktestConfig(lookback="3y", top_n=5),
        synthetic=True,
        use_cache=False,
        cost_grid=(10.0, 50.0),
        rebalance_list=(21,),
    )
    assert result["markdown"].startswith("# Factor Attribution")
    assert result["verdict"]["overall"] in {"PASS", "PARTIAL", "FAIL"}
    assert "beta" in result["market"]
    assert "t_stat" in result["sector_neutral"]
    assert result["sector_weights"]
    assert result["turnover_variants"]
    assert "Market alpha (Jensen)" in result["markdown"]


def test_attribution_resolves_universe_not_settings_universe(settings, monkeypatch):
    """Universe resolution has exactly one entry point (AGENT.md §5.5).

    Attribution must go through ``resolve_universe`` (so it honours
    static/dynamic/hybrid) rather than reading ``settings['universe']``
    directly — which silently ignored ``universe_config`` (the P0c bug).
    """
    import core.backtest.attribution as att
    from core.universe import UniverseResolution

    custom = [f"Z{i:02d}" for i in range(12)]

    def fake_resolve(_settings, **_kwargs):
        return UniverseResolution(tickers=custom + ["SPY"], mode="dynamic", source="test")

    monkeypatch.setattr(att, "resolve_universe", fake_resolve)

    s = dict(settings)
    s["universe"] = [f"A{i:02d}" for i in range(30)] + ["SPY"]

    result = run_attribution_suite(
        settings=s,
        config=BacktestConfig(lookback="2y", top_n=5),
        synthetic=True,
        use_cache=False,
        cost_grid=(10.0,),
        rebalance_list=(21,),
    )
    # The resolved list is used (benchmark excluded), not settings['universe'].
    assert result["universe_size"] == len(custom)
