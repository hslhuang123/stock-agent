"""Sector-neutral scoring tests."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.backtest.attribution import run_attribution_suite, turnover_variants
from core.backtest.engine import BacktestConfig
from core.backtest.robustness import build_panel, evaluate_panel
from core.backtest.sector_neutral import (
    add_sector_scores,
    largest_remainder,
    sector_neutral_target,
)
from core.config import load_settings
from core.features.indicators import add_indicators
from core.ingest.prices import fetch_prices


@pytest.fixture(scope="module")
def settings():
    s = load_settings()
    s["universe"] = [f"S{i:02d}" for i in range(40)] + ["SPY"]
    s["screening"] = dict(s["screening"], min_avg_dollar_volume=0)
    return s


def _frames(tickers, period="3y"):
    prices, _ = fetch_prices(tickers, period=period, synthetic=True, use_cache=False)
    return {t: (df if "sma20" in df.columns else add_indicators(df)) for t, df in prices.items()}


# --------------------------------------------------------------------------- #
# largest remainder
# --------------------------------------------------------------------------- #

def test_largest_remainder_sums_to_total():
    quotas = largest_remainder({"a": 0.5, "b": 0.3, "c": 0.2}, 10)
    assert sum(quotas.values()) == 10
    assert quotas["a"] == 5
    assert quotas["b"] == 3
    assert quotas["c"] == 2


def test_largest_remainder_handles_rounding():
    quotas = largest_remainder({"a": 1, "b": 1, "c": 1}, 10)
    assert sum(quotas.values()) == 10
    assert set(quotas.values()) <= {3, 4}


def test_largest_remainder_edge_cases():
    assert largest_remainder({}, 5) == {}
    assert largest_remainder({"a": 1.0}, 0) == {}


# --------------------------------------------------------------------------- #
# within-sector scoring
# --------------------------------------------------------------------------- #

def test_within_sector_ranking_lifts_second_best_sector():
    """C is 3rd overall but best in Energy, so it must out-rank B on sector_score."""
    ranked = pd.DataFrame(
        {
            "trend_score": [90, 70, 80, 50],
            "ret_3m": [0.5, 0.2, 0.3, 0.1],
            "ret_6m": [0.6, 0.3, 0.4, 0.2],
            "rel_strength_3m": [0.4, 0.1, 0.2, 0.0],
            "trend_quality_raw": [5, 3, 4, 2],
            "vol_ratio": [1.5, 1.1, 1.3, 1.0],
            "proximity_52w_high": [1.0, 0.9, 0.95, 0.8],
            "risk_adj_mom": [0.5, 0.2, 0.3, 0.1],
        },
        index=["A", "B", "C", "D"],
    )
    sectors = {"A": "Tech", "B": "Tech", "C": "Energy", "D": "Energy"}
    out = add_sector_scores(ranked, load_settings()["scoring_weights"], sectors)

    assert out.loc["A", "sector_score"] > out.loc["B", "sector_score"]
    assert out.loc["C", "sector_score"] > out.loc["D", "sector_score"]
    # The pooled order was A > C > B > D; within-sector C jumps above B.
    assert out.loc["C", "sector_score"] > out.loc["B", "sector_score"]


def test_add_sector_scores_single_name_sector():
    ranked = pd.DataFrame(
        {"ret_3m": [0.5], "ret_6m": [0.5], "rel_strength_3m": [0.5],
         "trend_quality_raw": [5], "vol_ratio": [1.0], "proximity_52w_high": [1.0],
         "risk_adj_mom": [0.5]},
        index=["ONLY"],
    )
    out = add_sector_scores(ranked, load_settings()["scoring_weights"], {"ONLY": "Energy"})
    assert out.loc["ONLY", "sector_score"] == pytest.approx(50.0, abs=1.0)


# --------------------------------------------------------------------------- #
# quota targeting
# --------------------------------------------------------------------------- #

def test_sector_neutral_target_respects_quotas():
    ranked = pd.DataFrame(
        {"sector": ["Tech"] * 10 + ["Energy"] * 10 + ["Health"] * 10,
         "sector_score": list(np.linspace(100, 90, 10)) + list(np.linspace(80, 70, 10)) + list(np.linspace(60, 50, 10))},
        index=[f"T{i}" for i in range(10)] + [f"E{i}" for i in range(10)] + [f"H{i}" for i in range(10)],
    )
    ordered = list(ranked.sort_values("sector_score", ascending=False).index)
    picks = sector_neutral_target(ranked, ordered, top_n=9, sector_weighting="equal")
    assert len(picks) == 9
    counts = Counter(ranked.loc[t, "sector"] for t in picks)
    assert set(counts.values()) == {3}, counts


def test_sector_neutral_target_universe_weighting():
    ranked = pd.DataFrame(
        {"sector": ["Tech"] * 20 + ["Energy"] * 10,
         "sector_score": list(np.linspace(100, 80, 20)) + list(np.linspace(70, 60, 10))},
        index=[f"T{i}" for i in range(20)] + [f"E{i}" for i in range(10)],
    )
    ordered = list(ranked.sort_values("sector_score", ascending=False).index)
    picks = sector_neutral_target(ranked, ordered, top_n=6, sector_weighting="universe")
    counts = Counter(ranked.loc[t, "sector"] for t in picks)
    assert counts["Tech"] == 4 and counts["Energy"] == 2


# --------------------------------------------------------------------------- #
# evaluate_panel integration
# --------------------------------------------------------------------------- #

def test_sector_neutral_spreads_picks(settings):
    tickers = [t for t in settings["universe"] if t != "SPY"]
    frames = _frames(["SPY"] + tickers)
    config = BacktestConfig(lookback="3y", rebalance_days=21, holding_days=21, top_n=8)
    panel = build_panel(frames, "SPY", config, tickers)
    sectors = {t: ["Tech", "Energy", "Health", "Financials"][i % 4] for i, t in enumerate(tickers)}

    ev = evaluate_panel(
        panel, settings["scoring_weights"], settings["screening"],
        top_n=8, scoring="sector_neutral", sectors=sectors,
    )
    for p in ev["periods"]:
        if not p["picks"]:
            continue
        assert set(p["picks"]).issubset(set(p["eligible"]))
        counts = Counter(sectors[t] for t in p["picks"])
        assert len(counts) >= 3, counts
        assert max(counts.values()) <= 4, counts


def test_sector_neutral_without_sectors_falls_back(settings):
    tickers = [t for t in settings["universe"] if t != "SPY"]
    frames = _frames(["SPY"] + tickers)
    config = BacktestConfig(lookback="3y", rebalance_days=21, holding_days=21, top_n=5)
    panel = build_panel(frames, "SPY", config, tickers)

    with_sectors = evaluate_panel(
        panel, settings["scoring_weights"], settings["screening"],
        top_n=5, scoring="sector_neutral", sectors={t: "Tech" for t in tickers},
    )
    # Single-sector universe -> sector_score is uniform -> picks still produced.
    assert any(p["picks"] for p in with_sectors["periods"])


def test_hysteresis_works_with_sector_neutral(settings):
    tickers = [t for t in settings["universe"] if t != "SPY"]
    frames = _frames(["SPY"] + tickers)
    config = BacktestConfig(lookback="3y", rebalance_days=21, holding_days=21, top_n=6)
    panel = build_panel(frames, "SPY", config, tickers)
    sectors = {t: ["Tech", "Energy", "Health"][i % 3] for i, t in enumerate(tickers)}

    ev = evaluate_panel(
        panel, settings["scoring_weights"], settings["screening"],
        top_n=6, selection="hysteresis", exit_multiple=2.0,
        scoring="sector_neutral", sectors=sectors,
    )
    for p in ev["periods"]:
        if p["eligible"]:
            assert len(p["picks"]) == 6


# --------------------------------------------------------------------------- #
# suite wiring
# --------------------------------------------------------------------------- #

def test_turnover_variants_includes_scoring_modes(settings):
    tickers = [t for t in settings["universe"] if t != "SPY"]
    frames = _frames(["SPY"] + tickers)
    config = BacktestConfig(lookback="3y", top_n=5)
    sectors = {t: ["Tech", "Energy", "Health"][i % 3] for i, t in enumerate(tickers)}

    rows = turnover_variants(
        frames, "SPY", tickers, settings["scoring_weights"], settings["screening"], config,
        top_n=5, rebalance_list=(21,), cost_grid=(10.0,),
        variants=(("topn", 0.0),),
        sectors=sectors, scoring_list=("pooled", "sector_neutral"),
    )
    modes = {r["scoring"] for r in rows}
    assert modes == {"pooled", "sector_neutral"}
    assert len(rows) == 2


def test_attribution_suite_has_scoring_comparison(settings):
    result = run_attribution_suite(
        settings=settings,
        config=BacktestConfig(lookback="3y", top_n=5),
        synthetic=True,
        use_cache=False,
        cost_grid=(10.0,),
        rebalance_list=(21,),
    )
    comp = result["verdict"]["scoring_comparison"]
    assert "pooled" in comp
    assert "sector_neutral" in comp
    assert "Stock selection (sector-neutral)" in result["markdown"]
    assert "Pooled vs sector-neutral scoring" in result["markdown"]
    # Every variant row records its scoring mode.
    assert all("scoring" in v for v in result["turnover_variants"])
