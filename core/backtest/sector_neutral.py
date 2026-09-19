"""Sector-neutral scoring and selection.

The pooled trend score ranks all names together, which in practice just surfaces
the hottest sector. Sector-neutral scoring instead:

1. **Ranks each factor within its sector**, so a name competes only against its
   sector peers.
2. **Fixes the sector allocation** (equal across sectors, or matching the
   universe's own distribution) and fills a quota within each sector.

If the signal has genuine stock-selection content, this variant should retain a
positive excess against a sector-matched benchmark. If it does not, the pooled
result was sector rotation and nothing more.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.features.trending import WEIGHT_TO_COLUMN


def largest_remainder(weights: dict[str, float], total: int) -> dict[str, int]:
    """Apportion ``total`` integer slots proportional to ``weights``."""
    if not weights or total <= 0:
        return {}
    total_w = sum(weights.values())
    if total_w <= 0:
        return {}
    raw = {k: (v / total_w) * total for k, v in weights.items()}
    floors = {k: int(np.floor(v)) for k, v in raw.items()}
    assigned = sum(floors.values())
    remainder = total - assigned
    if remainder > 0:
        order = sorted(raw, key=lambda k: (raw[k] - floors[k]), reverse=True)
        for k in order[:remainder]:
            floors[k] += 1
    return floors


def add_sector_scores(
    ranked: pd.DataFrame,
    weights: dict[str, float],
    sectors: dict[str, str],
) -> pd.DataFrame:
    """Add ``sector`` and ``sector_score`` (within-sector weighted percentile)."""
    df = ranked.copy()
    df["sector"] = [sectors.get(t, "Unknown") for t in df.index]

    total_weight = sum(weights.values()) or 1.0
    score = pd.Series(0.0, index=df.index)

    for _, group in df.groupby("sector"):
        for factor, weight in weights.items():
            col = WEIGHT_TO_COLUMN.get(factor)
            if col is None or col not in df.columns:
                continue
            # Percentile rank within the sector; single-name sectors get 0.5.
            ranks = group[col].rank(pct=True).fillna(0.5) if len(group) > 1 else pd.Series(0.5, index=group.index)
            score.loc[group.index] += ranks * (weight / total_weight)

    df["sector_score"] = (score * 100).round(1)
    return df


def sector_neutral_target(
    ranked: pd.DataFrame,
    ordered: list[str],
    top_n: int,
    sector_weighting: str = "universe",
) -> list[str]:
    """Pick ``top_n`` names respecting fixed sector quotas.

    ``sector_weighting="universe"`` matches the eligible universe's sector mix;
    ``"equal"`` gives every represented sector the same allocation.
    """
    if ranked.empty:
        return []

    counts = ranked["sector"].value_counts()
    if counts.empty:
        return ordered[:top_n]

    if sector_weighting == "equal":
        target = {s: 1.0 for s in counts.index}
    else:
        target = counts.to_dict()

    quotas = largest_remainder(target, top_n)
    sector_of = ranked["sector"].to_dict()

    picks: list[str] = []
    for sector, quota in quotas.items():
        if quota <= 0:
            continue
        taken = [t for t in ordered if sector_of.get(t) == sector][:quota]
        picks.extend(taken)

    if len(picks) < top_n:
        picks.extend(t for t in ordered if t not in picks)

    return picks[:top_n]


def sector_tilt(ranked: pd.DataFrame, sectors: dict[str, str]) -> dict[str, float]:
    """Average sector weight of a set of names (equal weighted)."""
    from core.backtest.sectors import sector_weights

    if ranked.empty:
        return {}
    totals: dict[str, float] = {}
    for ticker in ranked.index:
        for sector, w in sector_weights([ticker], sectors).items():
            totals[sector] = totals.get(sector, 0.0) + w
    n = len(ranked)
    return {s: round(v / n, 4) for s, v in sorted(totals.items(), key=lambda kv: -kv[1])}
