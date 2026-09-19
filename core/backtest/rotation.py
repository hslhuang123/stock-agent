"""Sector rotation diagnostic.

Distinguishes two very different explanations for a sector tilt:

* **Rotation** — the strategy dynamically moves into sectors that then outperform.
  That is skill, and it is tradable.
* **Static tilt** — the weights vary but carry no forecasting content. Any gain is
  a regime bet (e.g. tech beta during a tech bull market), not repeatable.

The test: correlate each period's sector weight with that same period's sector
return. Positive and meaningful means the tilt is predictive.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from core.backtest.robustness import Panel
from core.backtest.sectors import sector_weights


def sector_weight_series(evaluation: dict[str, Any], sectors: dict[str, str]) -> pd.DataFrame:
    """Per-period sector weights of the strategy's picks."""
    rows: list[dict[str, float]] = []
    for period in evaluation["periods"]:
        picks = period.get("picks") or []
        rows.append(sector_weights(picks, sectors) if picks else {})

    df = pd.DataFrame(rows).fillna(0.0)
    for sector in set(sectors.values()):
        if sector not in df.columns:
            df[sector] = 0.0
    return df[sorted(df.columns)]


def rotation_skill(
    panel: Panel,
    evaluation: dict[str, Any],
    sectors: dict[str, str],
) -> dict[str, Any]:
    """Correlate sector weights with same-period sector returns."""
    weights_flat: list[float] = []
    returns_flat: list[float] = []

    for step, period in enumerate(evaluation["periods"]):
        picks = period.get("picks") or []
        if not picks:
            continue
        weights = sector_weights(picks, sectors)

        by_sector: dict[str, list[float]] = defaultdict(list)
        for ticker in period.get("eligible") or []:
            value = panel.fwd.get(ticker, [None] * panel.n_periods)[step]
            if value is not None:
                by_sector[sectors.get(ticker, "Unknown")].append(float(value))

        for sector, values in by_sector.items():
            if values:
                weights_flat.append(weights.get(sector, 0.0))
                returns_flat.append(float(np.mean(values)))

    corr = None
    if len(weights_flat) > 2 and np.std(weights_flat) > 0 and np.std(returns_flat) > 0:
        corr = float(np.corrcoef(weights_flat, returns_flat)[0, 1])

    series = sector_weight_series(evaluation, sectors)
    stats: dict[str, dict[str, float]] = {}
    for sector in series.columns:
        col = series[sector].to_numpy(dtype=float)
        if col.mean() > 0.01 or col.std() > 0.01:
            stats[sector] = {
                "mean": round(float(col.mean()), 4),
                "std": round(float(col.std()), 4),
                "min": round(float(col.min()), 4),
                "max": round(float(col.max()), 4),
            }

    # Average dispersion across sectors = how much the book actually rotates.
    turbulence = float(np.mean([s["std"] for s in stats.values()])) if stats else 0.0

    return {
        "corr_weight_vs_sector_return": round(corr, 4) if corr is not None else None,
        "n_pairs": len(weights_flat),
        "rotation_turbulence": round(turbulence, 4),
        "weight_stats": stats,
        "is_dynamic": turbulence > 0.05,
        "is_predictive": bool(corr is not None and corr > 0.10),
    }
