"""Robustness suite tests.

The most important test here is ``test_panel_matches_engine``: it pins the fast
panel evaluator to the reference backtester so the two can never drift.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.backtest.engine import BacktestConfig, run_backtest
from core.backtest.robustness import (
    build_panel,
    cost_sensitivity,
    evaluate_panel,
    parameter_sweep,
    run_robustness_suite,
    universe_bootstrap,
)
from core.config import load_settings
from core.features.indicators import add_indicators
from core.ingest.prices import fetch_prices


@pytest.fixture(scope="module")
def settings():
    s = load_settings()
    s["universe"] = [f"R{i:02d}" for i in range(40)] + ["SPY"]
    s["screening"] = dict(s["screening"], min_avg_dollar_volume=0)
    return s


def _frames(settings, tickers, period="3y"):
    prices, _ = fetch_prices(tickers, period=period, synthetic=True, use_cache=False)
    return {t: (df if "sma20" in df.columns else add_indicators(df)) for t, df in prices.items()}


# --------------------------------------------------------------------------- #
# The critical guard: panel evaluator == reference engine
# --------------------------------------------------------------------------- #

def test_panel_matches_engine(settings):
    tickers = [t for t in settings["universe"] if t != "SPY"]
    config = BacktestConfig(lookback="3y", rebalance_days=21, holding_days=21, top_n=5, cost_bps=10)

    reference = run_backtest(settings=settings, config=config, synthetic=True, use_cache=False)

    frames = _frames(settings, ["SPY"] + tickers)
    panel = build_panel(frames, "SPY", config, tickers)
    fast = evaluate_panel(
        panel, settings["scoring_weights"], settings["screening"],
        top_n=5, cost_bps=10.0, weight_mode="equal",
    )

    assert len(fast["net_returns"]) == len(reference["periods"])
    # The reference engine rounds stored period returns to 4 dp; the panel keeps
    # full precision. Tolerance reflects that.
    ref_net = [p["net_return"] for p in reference["periods"]]
    np.testing.assert_allclose(fast["net_returns"], ref_net, rtol=0, atol=1e-4)
    np.testing.assert_allclose(
        fast["bench_returns"], [p["benchmark_return"] for p in reference["periods"]], rtol=0, atol=1e-4
    )


def test_panel_picks_match_engine(settings):
    tickers = [t for t in settings["universe"] if t != "SPY"]
    config = BacktestConfig(lookback="3y", rebalance_days=21, holding_days=21, top_n=5)
    reference = run_backtest(settings=settings, config=config, synthetic=True, use_cache=False)
    frames = _frames(settings, ["SPY"] + tickers)
    panel = build_panel(frames, "SPY", config, tickers)
    fast = evaluate_panel(panel, settings["scoring_weights"], settings["screening"], top_n=5, cost_bps=10.0)
    for got, exp in zip(fast["periods"], reference["periods"]):
        assert got["picks"] == exp["picks"]


# --------------------------------------------------------------------------- #
# Panel mechanics
# --------------------------------------------------------------------------- #

def test_panel_shapes(settings):
    tickers = [t for t in settings["universe"] if t != "SPY"]
    frames = _frames(settings, ["SPY"] + tickers)
    config = BacktestConfig(lookback="3y", rebalance_days=21, holding_days=21)
    panel = build_panel(frames, "SPY", config, tickers)
    assert panel.n_periods > 5
    for t in tickers:
        assert len(panel.metrics[t]) == panel.n_periods
        assert len(panel.fwd[t]) == panel.n_periods


def test_evaluate_subset_smaller_than_full(settings):
    tickers = [t for t in settings["universe"] if t != "SPY"]
    frames = _frames(settings, ["SPY"] + tickers)
    config = BacktestConfig(lookback="3y", rebalance_days=21, holding_days=21, top_n=5)
    panel = build_panel(frames, "SPY", config, tickers)
    full = evaluate_panel(panel, settings["scoring_weights"], settings["screening"], top_n=5)
    half = evaluate_panel(
        panel, settings["scoring_weights"], settings["screening"], top_n=5, tickers=tickers[:20]
    )
    assert set(half["periods"][0]["picks"]).issubset(set(tickers[:20]))
    assert len(full["net_returns"]) == len(half["net_returns"])


# --------------------------------------------------------------------------- #
# Individual tests
# --------------------------------------------------------------------------- #

def test_bootstrap_reports_distribution(settings):
    tickers = [t for t in settings["universe"] if t != "SPY"]
    frames = _frames(settings, ["SPY"] + tickers)
    config = BacktestConfig(lookback="3y", rebalance_days=21, holding_days=21, top_n=5)
    panel = build_panel(frames, "SPY", config, tickers)

    out = universe_bootstrap(
        panel, settings["scoring_weights"], settings["screening"],
        top_n=5, n_trials=6, drop_frac=0.3, seed=1,
    )
    assert out["n_trials"] == 6
    assert out["keep"] == int(round(len(tickers) * 0.7))
    assert 0.0 <= out["edge_positive_share"] <= 1.0
    assert out["edge_p5"] <= out["edge_median"] <= out["edge_p95"]
    assert len(out["trials"]) == 6


def test_cost_sensitivity_is_monotone(settings):
    tickers = [t for t in settings["universe"] if t != "SPY"]
    frames = _frames(settings, ["SPY"] + tickers)
    config = BacktestConfig(lookback="3y", rebalance_days=21, holding_days=21, top_n=5)
    panel = build_panel(frames, "SPY", config, tickers)

    out = cost_sensitivity(
        panel, settings["scoring_weights"], settings["screening"], top_n=5, cost_grid=(0.0, 50.0)
    )
    rows = out["rows"]
    assert rows[0]["net_total"] >= rows[1]["net_total"]
    assert out["max_cost_bps"] == 50.0


def test_parameter_sweep_produces_grid(settings):
    tickers = [t for t in settings["universe"] if t != "SPY"]
    frames = _frames(settings, ["SPY"] + tickers)
    config = BacktestConfig(lookback="3y", top_n=5)
    out = parameter_sweep(
        frames, "SPY", tickers, settings["scoring_weights"], settings["screening"], config,
        top_n_grid=(5, 10), rebalance_grid=(21, 42),
    )
    assert out["n_cells"] == 4
    combos = {(c["rebalance_days"], c["top_n"]) for c in out["cells"]}
    assert combos == {(21, 5), (21, 10), (42, 5), (42, 10)}


def test_verdict_criteria(settings):
    result = run_robustness_suite(
        settings=settings,
        config=BacktestConfig(lookback="3y", top_n=5),
        synthetic=True,
        use_cache=False,
        bootstrap_trials=4,
        top_n_grid=(5, 10),
        rebalance_grid=(21,),
        cost_grid=(0.0, 50.0),
    )
    v = result["verdict"]
    assert v["overall"] in {"PASS", "PARTIAL", "FAIL"}
    assert len(v["checks"]) == 3
    assert result["markdown"].startswith("# Robustness Gate")
    assert "Universe bootstrap" in result["markdown"]
    assert "edge_vs_universe" in result["baseline"]
