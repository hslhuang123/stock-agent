"""Sector rotation diagnostic tests."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.backtest.engine import BacktestConfig
from core.backtest.robustness import build_panel, evaluate_panel
from core.backtest.rotation import rotation_skill, sector_weight_series
from core.config import load_settings
from core.features.indicators import add_indicators
from core.ingest.prices import fetch_prices


@pytest.fixture(scope="module")
def settings():
    s = load_settings()
    s["universe"] = [f"X{i:02d}" for i in range(40)] + ["SPY"]
    s["screening"] = dict(s["screening"], min_avg_dollar_volume=0)
    return s


@pytest.fixture(scope="module")
def built(settings):
    tickers = [t for t in settings["universe"] if t != "SPY"]
    prices, _ = fetch_prices(["SPY"] + tickers, period="3y", synthetic=True, use_cache=False)
    frames = {t: (d if "sma20" in d.columns else add_indicators(d)) for t, d in prices.items()}
    config = BacktestConfig(lookback="3y", rebalance_days=21, holding_days=21, top_n=6)
    panel = build_panel(frames, "SPY", config, tickers)
    sectors = {t: ["Tech", "Energy", "Health", "Financials"][i % 4] for i, t in enumerate(tickers)}
    return panel, sectors


def test_weight_series_shape(built):
    panel, sectors = built
    ev = evaluate_panel(panel, load_settings()["scoring_weights"], load_settings()["screening"], top_n=6)
    series = sector_weight_series(ev, sectors)
    assert len(series) == panel.n_periods
    assert set(series.columns) == {"Tech", "Energy", "Health", "Financials"}
    # Weights of a pick list sum to 1 (or 0 for empty periods).
    row_sums = series.sum(axis=1)
    assert ((np.isclose(row_sums, 1.0)) | (np.isclose(row_sums, 0.0))).all()


def test_rotation_skill_structure(built):
    panel, sectors = built
    settings = load_settings()
    ev = evaluate_panel(panel, settings["scoring_weights"], settings["screening"], top_n=6)
    out = rotation_skill(panel, ev, sectors)

    assert out["n_pairs"] > 0
    corr = out["corr_weight_vs_sector_return"]
    assert corr is None or -1.0 <= corr <= 1.0
    assert isinstance(out["is_dynamic"], bool)
    assert isinstance(out["is_predictive"], bool)
    assert out["weight_stats"]


def test_static_book_is_not_predictive(settings):
    """If every period holds the same sector mix, correlation must be ~0 and turbulence low."""
    tickers = [t for t in settings["universe"] if t != "SPY"]
    prices, _ = fetch_prices(["SPY"] + tickers, period="3y", synthetic=True, use_cache=False)
    frames = {t: (d if "sma20" in d.columns else add_indicators(d)) for t, d in prices.items()}
    config = BacktestConfig(lookback="3y", rebalance_days=21, holding_days=21, top_n=4)
    panel = build_panel(frames, "SPY", config, tickers)

    # One sector only -> weights are constant at 100% -> zero variance.
    sectors = {t: "Tech" for t in tickers}
    ev = evaluate_panel(panel, settings["scoring_weights"], settings["screening"], top_n=4)
    out = rotation_skill(panel, ev, sectors)
    assert out["is_dynamic"] is False
    assert out["is_predictive"] is False


def test_predictive_weights_detected():
    """Synthetic case: weight is high exactly when that sector's return is high."""
    from core.backtest.robustness import Panel
    import pandas as pd

    class FakePanel(Panel):
        pass

    n = 12
    schedule = [(i, i + 1, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01")) for i in range(n)]
    tickers = ["A", "B"]
    metrics = {t: [{} for _ in range(n)] for t in tickers}
    fwd = {"A": [0.10] * n, "B": [-0.05] * n}  # A always wins
    panel = FakePanel(
        tickers=tickers,
        schedule=schedule,
        bench_returns=[0.0] * n,
        metrics=metrics,
        fwd=fwd,
    )
    # Strategy always holds A (the strong sector) -> weight matches return sign.
    evaluation = {
        "periods": [{"picks": ["A"], "eligible": ["A", "B"]} for _ in range(n)],
        "net_returns": [0.10] * n,
    }
    sectors = {"A": "Tech", "B": "Energy"}
    out = rotation_skill(panel, evaluation, sectors)
    assert out["corr_weight_vs_sector_return"] is not None
    assert out["is_predictive"] is True
