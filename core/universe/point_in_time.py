"""Point-in-time universe reconstruction.

A dynamic universe cannot be validated with a backtest unless the universe is
knowable *as of each rebalance date*. The Yahoo screener only ever returns
today's list, so using it historically is look-ahead bias.

Instead we rebuild the universe from data we already have: at each rebalance
date, rank a fixed pool by trailing average dollar volume and keep the top N.
Dollar volume as of date D is knowable on date D, so this is look-ahead free.

Caveat: this removes *hindsight ranking* bias, not *delisting* bias — companies
that went to zero are absent from the price pool entirely.

``dollar_volume_frame`` is split out so a backtest can build it once and then
slice it per rebalance instead of rebuilding it on every date.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import pandas as pd

__all__ = [
    "dollar_volume_frame",
    "liquidity_ranking",
    "liquidity_universe_at",
    "ranking_from_frame",
    "universe_at_from_frame",
    "universe_schedule",
]


def dollar_volume_frame(prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Wide frame of Close*Volume, indexed by date, one column per ticker."""
    series: dict[str, pd.Series] = {}
    for ticker, df in prices.items():
        if df is None or df.empty:
            continue
        if "Close" not in df.columns or "Volume" not in df.columns:
            continue
        frame = df[["Close", "Volume"]].copy()
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        series[str(ticker)] = frame["Close"] * frame["Volume"]
    if not series:
        return pd.DataFrame()
    return pd.DataFrame(series).sort_index()


def ranking_from_frame(
    dv: pd.DataFrame,
    as_of: Any,
    lookback_days: int = 20,
    min_bars: int | None = None,
) -> pd.Series:
    """Trailing average dollar volume per ticker using only bars up to ``as_of``.

    Returns a descending ``pd.Series``; empty when no ticker has enough history.
    """
    if dv is None or dv.empty:
        return pd.Series(dtype="float64")

    cutoff = pd.Timestamp(as_of)
    window = dv.loc[dv.index <= cutoff].tail(int(lookback_days))
    if window.empty:
        return pd.Series(dtype="float64")

    required = int(min_bars) if min_bars is not None else int(lookback_days)
    counts = window.notna().sum()
    eligible = counts[counts >= required].index

    ranking = window[eligible].mean().dropna().sort_values(ascending=False)
    ranking.index = ranking.index.astype(str)
    return ranking


def universe_at_from_frame(
    dv: pd.DataFrame,
    as_of: Any,
    top_n: int,
    lookback_days: int = 20,
    exclude: Iterable[str] = (),
    min_bars: int | None = None,
) -> list[str]:
    """Top ``top_n`` tickers by trailing dollar volume as of ``as_of``."""
    ranking = ranking_from_frame(dv, as_of, lookback_days, min_bars=min_bars)
    if ranking.empty:
        return []
    banned = {str(t) for t in exclude}
    picked = [str(t) for t in ranking.index if str(t) not in banned]
    return picked[: int(top_n)]


def liquidity_ranking(
    prices: dict[str, pd.DataFrame],
    as_of: Any,
    lookback_days: int = 20,
    min_bars: int | None = None,
) -> pd.Series:
    """Convenience wrapper: build the dollar-volume frame then rank it."""
    return ranking_from_frame(
        dollar_volume_frame(prices), as_of, lookback_days, min_bars=min_bars
    )


def liquidity_universe_at(
    prices: dict[str, pd.DataFrame],
    as_of: Any,
    top_n: int,
    lookback_days: int = 20,
    exclude: Iterable[str] = (),
    min_bars: int | None = None,
) -> list[str]:
    """Convenience wrapper: build the dollar-volume frame then take the top N."""
    return universe_at_from_frame(
        dollar_volume_frame(prices),
        as_of,
        top_n,
        lookback_days=lookback_days,
        exclude=exclude,
        min_bars=min_bars,
    )


def universe_schedule(
    prices: dict[str, pd.DataFrame],
    dates: Sequence[Any],
    top_n: int,
    lookback_days: int = 20,
    exclude: Iterable[str] = (),
) -> dict[pd.Timestamp, list[str]]:
    """Precompute the point-in-time universe for each rebalance date."""
    dv = dollar_volume_frame(prices)
    banned = {str(t) for t in exclude}
    return {
        pd.Timestamp(date): universe_at_from_frame(
            dv, date, top_n, lookback_days=lookback_days, exclude=banned
        )
        for date in dates
    }
