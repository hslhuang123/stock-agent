"""Backtest tests: look-ahead safety, metric consistency, engine mechanics."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.backtest.engine import BacktestConfig, run_backtest, _stats
from core.config import load_settings
from core.features.indicators import add_indicators
from core.features.metrics import metrics_at, price_at, trend_quality
from core.features.trending import compute_feature_table
from core.ingest.prices import synthetic_prices


@pytest.fixture(scope="module")
def settings():
    return load_settings()


# --------------------------------------------------------------------------- #
# Look-ahead safety — the whole point of the design
# --------------------------------------------------------------------------- #

def test_metrics_at_ignores_future_data():
    """A violent future move must not change a past metric row."""
    df = synthetic_prices("FUTURE", "1y")
    idx = len(df) - 40

    before = metrics_at(add_indicators(df), idx)
    assert before is not None

    tampered = df.copy()
    # Explode the last 30 bars — strictly after `idx`.
    tampered.iloc[idx + 1 :, tampered.columns.get_loc("Close")] *= 10.0
    after = metrics_at(add_indicators(tampered), idx)

    assert after is not None
    for key in ["close", "ret_3m", "ret_6m", "rsi14", "sma50", "high_52w", "proximity_52w_high"]:
        assert before[key] == pytest.approx(after[key]), f"{key} leaked future data"


def test_price_at_never_returns_future_price():
    df = synthetic_prices("LEVEL", "1y")
    cutoff = df.index[len(df) // 2]
    pos, price = price_at(df, cutoff)
    assert df.index[pos] <= cutoff
    assert price == pytest.approx(float(df["Close"].iloc[pos]))


def test_metrics_at_matches_feature_table(settings):
    """Point-in-time metrics at the last bar must equal the live screen."""
    tickers = ["A1", "A2", "A3"]
    prices = {t: add_indicators(synthetic_prices(t, "1y")) for t in tickers}
    bench = add_indicators(synthetic_prices("SPY", "1y"))

    table = compute_feature_table(prices, benchmark=bench)

    bench_ret_3m = float(bench["Close"].iloc[-1] / bench["Close"].iloc[-64] - 1.0)
    for t in tickers:
        m = metrics_at(prices[t], -1, bench_ret_3m=bench_ret_3m)
        assert m["close"] == pytest.approx(table.loc[t, "close"])
        assert m["ret_3m"] == pytest.approx(table.loc[t, "ret_3m"])
        assert m["rsi14"] == pytest.approx(table.loc[t, "rsi14"])
        assert trend_quality(m) == pytest.approx(table.loc[t, "trend_quality_raw"])


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #

def test_stats_basic_properties():
    returns = np.array([0.02, -0.01, 0.03, 0.01, -0.02] * 6)
    bench = np.zeros_like(returns)
    s = _stats(returns, bench, ppy=12)
    assert s["periods"] == 30
    assert s["hit_rate"] == pytest.approx(0.6)
    assert s["total_return"] > 0
    assert s["max_drawdown"] <= 0
    assert s["win_rate_vs_bench"] == pytest.approx(0.6)


def test_stats_all_negative_drawdown():
    returns = np.array([-0.05] * 10)
    s = _stats(returns, np.zeros(10), ppy=12)
    assert s["max_drawdown"] < 0
    assert s["total_return"] < 0


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #

def test_backtest_runs_offline(settings):
    settings = dict(settings)
    settings["universe"] = [f"B{i:02d}" for i in range(30)] + ["SPY"]
    settings["screening"] = dict(settings["screening"], min_avg_dollar_volume=0)
    config = BacktestConfig(
        lookback="3y", rebalance_days=21, holding_days=21, top_n=5, cost_bps=10, folds=3
    )

    result = run_backtest(settings=settings, config=config, synthetic=True, use_cache=False)

    assert result["summary"]["net"]["periods"] > 10
    assert len(result["equity_curve"]["net"]) == result["summary"]["net"]["periods"]
    assert len(result["periods"]) == result["summary"]["net"]["periods"]
    assert result["markdown"].startswith("# Walk-Forward Backtest")
    assert len(result["folds"]) == 3
    assert "ic_mean" in result["signal_quality"]
    # Equity curve must start near 1.0 after the first period.
    assert result["equity_curve"]["net"][0] > 0
    # Every period either has picks or is a documented cash period.
    for p in result["periods"]:
        assert "net_return" in p and p["net_return"] is not None


def test_backtest_costs_reduce_returns(settings):
    settings = dict(settings)
    settings["universe"] = [f"C{i:02d}" for i in range(25)] + ["SPY"]
    settings["screening"] = dict(settings["screening"], min_avg_dollar_volume=0)
    base = dict(lookback="3y", rebalance_days=21, holding_days=21, top_n=5, folds=2)

    cheap = run_backtest(settings=settings, config=BacktestConfig(**base, cost_bps=0.0), synthetic=True, use_cache=False)
    pricey = run_backtest(settings=settings, config=BacktestConfig(**base, cost_bps=50.0), synthetic=True, use_cache=False)

    assert pricey["summary"]["net"]["total_return"] <= cheap["summary"]["net"]["total_return"]
    # Gross is cost-independent.
    assert pricey["summary"]["gross"]["total_return"] == pytest.approx(
        cheap["summary"]["gross"]["total_return"]
    )


def test_backtest_start_end_filter(settings):
    settings = dict(settings)
    settings["universe"] = [f"D{i:02d}" for i in range(20)] + ["SPY"]
    settings["screening"] = dict(settings["screening"], min_avg_dollar_volume=0)
    config = BacktestConfig(lookback="4y", start="2024-01-01", end="2025-01-01", top_n=5, folds=2)
    result = run_backtest(settings=settings, config=config, synthetic=True, use_cache=False)
    assert result["periods"]
    assert all(p["date"] >= "2024-01-01" for p in result["periods"])
