"""Tests for indicators, scoring, risk and the offline pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.agent.analyst import analyze_rules
from core.agent.tools import MarketContext, dispatch
from core.config import load_settings
from core.features.indicators import add_indicators, atr, rsi, sma
from core.features.trending import compute_feature_table, rank_trending
from core.ingest.prices import synthetic_prices
from core.pipeline import run_pipeline
from core.risk.rules import position_size, validate_thesis


@pytest.fixture(scope="module")
def settings():
    return load_settings()


# --------------------------------------------------------------------------- #
# Indicators
# --------------------------------------------------------------------------- #

def test_sma_matches_manual():
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    assert sma(s, 3).iloc[-1] == pytest.approx(4.0)


def test_rsi_bounds_and_direction():
    up = pd.Series(np.linspace(10, 50, 60))
    down = pd.Series(np.linspace(50, 10, 60))
    assert 0 <= rsi(up).iloc[-1] <= 100
    assert rsi(up).iloc[-1] > 70
    assert rsi(down).iloc[-1] < 30


def test_atr_positive():
    df = synthetic_prices("TEST", "6mo")
    assert (atr(df, 14).dropna() > 0).all()


def test_add_indicators_columns():
    df = add_indicators(synthetic_prices("TEST", "1y"))
    for col in ["sma20", "sma50", "sma200", "rsi14", "atr14", "macd_hist"]:
        assert col in df.columns


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def test_rank_trending_is_ordered_and_bounded(settings):
    tickers = [f"T{i:02d}" for i in range(20)]
    prices = {t: synthetic_prices(t, "1y") for t in tickers}
    bench = synthetic_prices("SPY", "1y")

    table = compute_feature_table(prices, benchmark=bench)
    assert not table.empty

    ranked = rank_trending(table, settings["scoring_weights"], settings["screening"])
    assert ranked["trend_score"].between(0, 100).all()
    assert ranked["trend_score"].is_monotonic_decreasing


def test_liquidity_filter_removes_illiquid(settings):
    prices = {"AAA": synthetic_prices("AAA", "1y")}
    prices["AAA"]["Volume"] = 1.0  # penny-volume
    table = compute_feature_table(prices, benchmark=None)
    ranked = rank_trending(table, settings["scoring_weights"], settings["screening"])
    assert ranked.empty


# --------------------------------------------------------------------------- #
# Risk
# --------------------------------------------------------------------------- #

def test_position_size_respects_risk_budget():
    plan = position_size(
        account_size=100_000, risk_per_trade_pct=1.0, entry=100, stop=95, max_position_pct=10.0
    )
    # $1000 risk / $5 per share = 200 shares, but $20k > 10% cap -> 100 shares
    assert plan["shares"] == 100
    assert plan["position_value"] <= 100_000 * 0.10
    assert plan["capped_by_max_position"] is True


def test_position_size_invalid_stop():
    plan = position_size(100_000, 1.0, entry=100, stop=105)
    assert plan["shares"] == 0


def test_validator_vetoes_extended():
    thesis = {
        "conviction": 0.9,
        "technicals": {"rsi14": 88, "ext_from_sma20": 0.30},
        "position": {"shares": 10},
    }
    passed, reasons = validate_thesis(thesis, {"min_conviction": 0.45})
    assert not passed
    assert len(reasons) >= 2


# --------------------------------------------------------------------------- #
# Tools + end-to-end
# --------------------------------------------------------------------------- #

def test_dispatch_unknown_tool_errors():
    ctx = MarketContext(prices={}, features=pd.DataFrame())
    assert "error" in dispatch(ctx, "nope", {})


def test_pipeline_runs_offline(settings):
    settings = dict(settings)
    settings["universe"] = [f"T{i:02d}" for i in range(25)] + ["SPY"]
    settings["screening"] = dict(settings["screening"], top_n=8, min_avg_dollar_volume=0)
    settings["agent"] = dict(settings["agent"], top_candidates=3)

    result = run_pipeline(settings=settings, synthetic=True, provider="rules", use_cache=False)

    assert result["universe_size"] > 0
    assert 1 <= len(result["candidates"]) <= 3
    assert result["markdown"].startswith("# Stock Agent Report")
    for c in result["candidates"]:
        assert c["source"] == "rules"
        assert "validation" in c
        assert c["position"]["shares"] >= 0


def test_benchmark_is_never_a_candidate(settings):
    """Invariant: the benchmark is fetched for regime/relative strength but is
    never offered as a tradeable candidate (AGENT.md §5.4).

    Regression guard for the bug where `rank_trending` ranked the benchmark
    alongside the universe and it could reach the shortlist.
    """
    settings = dict(settings)
    settings["universe"] = [f"T{i:02d}" for i in range(25)] + ["SPY"]
    settings["screening"] = dict(
        settings["screening"], top_n=15, min_avg_dollar_volume=0, min_price=0
    )
    settings["agent"] = dict(settings["agent"], top_candidates=10)

    result = run_pipeline(settings=settings, synthetic=True, provider="rules", use_cache=False)

    assert result["shortlist"], "shortlist should not be empty"
    assert all(row["ticker"] != "SPY" for row in result["shortlist"])
    assert all(c["ticker"] != "SPY" for c in result["candidates"])
    # 26 configured tickers including SPY -> 25 tradeable names.
    assert result["universe_size"] == 25
    # The benchmark must remain visible to the regime read even though it is
    # not tradeable.
    assert result["regime"]["available"] is True
    assert result["regime"]["benchmark"] == "SPY"


def test_analyst_emits_full_thesis(settings):
    tickers = [f"T{i:02d}" for i in range(10)]
    prices = {t: synthetic_prices(t, "1y") for t in tickers}
    bench = synthetic_prices("SPY", "1y")
    table = compute_feature_table(prices, benchmark=bench)
    ranked = rank_trending(table, settings["scoring_weights"], {"min_avg_dollar_volume": 0, "min_price": 0})
    ctx = MarketContext(prices=prices, features=ranked, benchmark="SPY")

    thesis = analyze_rules(ctx, ranked.index[0], settings)
    d = thesis.to_dict()
    for key in ["ticker", "thesis", "conviction", "position", "technicals", "invalidation"]:
        assert key in d
    assert 0.0 <= d["conviction"] <= 0.95
    assert d["position"]["stop"] < d["price"]
