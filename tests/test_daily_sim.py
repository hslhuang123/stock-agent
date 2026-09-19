"""Tests for the same-day round-trip simulator: no look-ahead, costs, structure."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.backtest.daily_sim import DailySimConfig, simulate_frames
from core.config import load_settings
from core.features.indicators import add_indicators
from core.ingest.prices import fetch_prices

TICKERS = ["SPY", "AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]


@pytest.fixture(scope="module")
def settings():
    return load_settings()


def _frames(tickers=TICKERS, period="1y") -> dict[str, pd.DataFrame]:
    prices, _ = fetch_prices(tickers, period=period, synthetic=True, use_cache=False)
    return {t: add_indicators(df) for t, df in prices.items()}


def _config(**kw) -> DailySimConfig:
    base = dict(lookback="1y", picks=2, shares=100, cost_bps=10.0, min_history_days=130)
    base.update(kw)
    return DailySimConfig(**base)


def test_simulation_is_deterministic(settings):
    frames = _frames()
    a = simulate_frames(frames, "SPY", settings, _config())
    b = simulate_frames(frames, "SPY", settings, _config())
    assert [d["net_pnl"] for d in a["days"]] == [d["net_pnl"] for d in b["days"]]
    assert a["summary"]["total_net"] == b["summary"]["total_net"]


def test_simulation_buys_and_sells_same_day(settings):
    result = simulate_frames(_frames(), "SPY", settings, _config())
    traded = [d for d in result["days"] if d["picks"]]
    assert traded, "expected at least one tradable day"
    for day in traded:
        assert day["date"] > day["signal_date"]
        for pick in day["picks"]:
            assert pick["shares"] == 100
            # net = gross - cost, and gross is (exit - entry) * shares
            expected_gross = pick["shares"] * (pick["exit"] - pick["entry"])
            # entry/exit are rounded for display, so allow up to 1 cent/share drift.
            assert pick["gross_pnl"] == pytest.approx(expected_gross, abs=1.01)
            assert pick["net_pnl"] == pytest.approx(pick["gross_pnl"] - pick["cost"], abs=0.05)
        assert day["net_pnl"] == pytest.approx(sum(p["net_pnl"] for p in day["picks"]), abs=0.05)


def test_simulation_ignores_future_data(settings):
    """Tampering bars after day P must not change any trade on or before P."""
    frames = _frames()
    before = simulate_frames(frames, "SPY", settings, _config())

    cutoff = max(2, len(frames["SPY"]) - 25)
    cutoff_date = frames["SPY"].index[cutoff - 1]
    tampered = {}
    for ticker, df in frames.items():
        df = df.copy()
        for col in ["Open", "High", "Low", "Close"]:
            df.iloc[cutoff:, df.columns.get_loc(col)] *= 5.0
        tampered[ticker] = df
    after = simulate_frames(tampered, "SPY", settings, _config())

    past_before = [d for d in before["days"] if d["date"] <= cutoff_date.date().isoformat()]
    past_after = [d for d in after["days"] if d["date"] <= cutoff_date.date().isoformat()]
    assert len(past_before) == len(past_after) and past_before

    for a, b in zip(past_before, past_after):
        assert a["date"] == b["date"]
        assert a["net_pnl"] == b["net_pnl"], f"future data leaked into {a['date']}"
        assert [p["ticker"] for p in a["picks"]] == [p["ticker"] for p in b["picks"]]


def test_costs_reduce_net_pnl(settings):
    frames = _frames()
    free = simulate_frames(frames, "SPY", settings, _config(cost_bps=0.0))
    costly = simulate_frames(frames, "SPY", settings, _config(cost_bps=50.0))
    assert costly["summary"]["total_cost"] > 0
    assert costly["summary"]["total_net"] < free["summary"]["total_net"]
    # Gross is cost-independent.
    assert costly["summary"]["total_gross"] == pytest.approx(free["summary"]["total_gross"], abs=1e-6)
