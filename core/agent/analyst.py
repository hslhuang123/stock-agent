"""The analyst: builds a structured thesis for a candidate.

Two backends share one output schema:

* ``analyze_rules``  - deterministic, always available, fully explainable.
* ``analyze_llm``    - adds narrative synthesis via an OpenAI-compatible model.

Crucially the LLM never owns the numbers. Levels, position sizing and the
underlying technicals always come from the deterministic tools.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from core.agent.tools import (
    MarketContext,
    get_levels,
    get_market_regime,
    get_relative_strength,
    get_technicals,
    get_trend_score,
)
from core.risk.rules import build_position

log = logging.getLogger(__name__)


@dataclass
class Thesis:
    ticker: str
    as_of: str
    source: str
    price: float
    trend_score: float
    direction: str
    conviction: float
    horizon: str
    thesis: str
    name: str = ""
    catalysts: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    technicals: dict[str, Any] = field(default_factory=dict)
    levels: dict[str, Any] = field(default_factory=dict)
    position: dict[str, Any] = field(default_factory=dict)
    factor_scores: dict[str, Any] = field(default_factory=dict)
    relative_strength: dict[str, Any] = field(default_factory=dict)
    invalidation: str = ""
    citations: list[str] = field(default_factory=list)
    validation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _pct(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{value * 100:+.1f}%"


def _compute_conviction(row: pd.Series, trend_score: float) -> tuple[float, list[str]]:
    """Blend relative rank with trend-structure confirmation."""
    checks: list[tuple[str, bool]] = [
        ("price > SMA50", bool(row["close"] > row["sma50"]) if pd.notna(row["sma50"]) else False),
        ("SMA50 > SMA200", bool(row["sma50"] > row["sma200"]) if pd.notna(row["sma200"]) else False),
        ("SMA50 rising", bool(row.get("sma50_slope", 0) > 0)),
        ("beating benchmark (3m)", bool(row.get("rel_strength_3m", 0) > 0)),
        ("MACD positive", bool(row.get("macd_hist", 0) > 0)),
        ("volume expanding", bool(row.get("vol_ratio", 0) > 1.0)),
        ("near 52w high", bool(row.get("proximity_52w_high", 0) > 0.9)),
    ]
    confirmed = [name for name, ok in checks if ok]
    structure = len(confirmed) / len(checks)

    conviction = 0.55 * (trend_score / 100.0) + 0.45 * structure

    rsi = float(row.get("rsi14") or 50)
    if rsi > 75:
        conviction -= 0.08
    elif rsi < 45:
        conviction -= 0.05
    ext = row.get("ext_from_sma20")
    if ext is not None and not pd.isna(ext) and float(ext) > 0.15:
        conviction -= 0.08

    return float(min(max(conviction, 0.0), 0.95)), confirmed


def analyze_rules(ctx: MarketContext, ticker: str, settings: dict[str, Any]) -> Thesis:
    """Deterministic thesis built entirely from computed metrics."""
    row = ctx.row(ticker)
    risk = settings.get("risk", {})
    agent_cfg = settings.get("agent", {})

    technicals = get_technicals(ctx, ticker)
    levels = get_levels(ctx, ticker)
    trend = get_trend_score(ctx, ticker)
    rel = get_relative_strength(ctx, ticker)
    position = build_position(row, risk)

    trend_score = float(trend["trend_score"])
    conviction, confirmed = _compute_conviction(row, trend_score)

    # --- Bull case (catalysts) --------------------------------------------
    catalysts: list[str] = []
    if confirmed:
        catalysts.extend(confirmed)
    if row.get("ret_3m") is not None and pd.notna(row["ret_3m"]):
        catalysts.append(f"3-month momentum of {_pct(row['ret_3m'])}")
    if row.get("rel_strength_3m") is not None and pd.notna(row["rel_strength_3m"]):
        catalysts.append(f"{_pct(row['rel_strength_3m'])} vs {ctx.benchmark} over 3 months")

    # --- Bear case (risks) ------------------------------------------------
    risks: list[str] = []
    rsi = float(row.get("rsi14") or 50)
    if rsi > 70:
        risks.append(f"RSI {rsi:.0f} — overbought, pullback risk")
    if row.get("ann_vol") is not None and pd.notna(row["ann_vol"]):
        risks.append(f"annualized volatility {row['ann_vol'] * 100:.0f}%")
    ext = row.get("ext_from_sma20")
    if ext is not None and not pd.isna(ext) and float(ext) > 0.1:
        risks.append(f"extended {ext * 100:.0f}% above SMA20")
    if not (row.get("vol_ratio", 0) > 1.0):
        risks.append("volume not confirming the move")
    if row.get("pct_from_52w_high") is not None and pd.notna(row["pct_from_52w_high"]) and row["pct_from_52w_high"] < -0.15:
        risks.append(f"{row['pct_from_52w_high'] * 100:.0f}% below 52-week high")

    # --- Direction --------------------------------------------------------
    above50 = pd.notna(row["sma50"]) and row["close"] > row["sma50"]
    if trend_score >= 65 and above50:
        direction = "long"
    elif trend_score >= 50:
        direction = "watch"
    else:
        direction = "avoid"

    # --- Supporting levels / invalidation ---------------------------------
    support = levels.get("support_20d")
    sma50 = row.get("sma50")
    invalidation_parts = []
    if support:
        invalidation_parts.append(f"close below 20-day support {support:.2f}")
    if sma50 and pd.notna(sma50):
        invalidation_parts.append(f"close below SMA50 {float(sma50):.2f}")
    invalidation = " or ".join(invalidation_parts) or "trend score drops below 50"

    horizon = agent_cfg.get("horizon", "2-6 weeks")
    narrative = (
        f"{ticker} ranks in the top {max(1, round(100 - trend_score))}% of the screened universe "
        f"on trend quality (score {trend_score:.0f}/100). "
        f"Price is {_pct(row.get('ret_3m'))} over three months versus "
        f"{_pct(rel.get('benchmark_ret')) if rel.get('benchmark_ret') is not None else 'the benchmark'}, "
        f"with {len(confirmed)}/{7} structural confirmations. "
        f"{'Structure is bullish.' if above50 else 'Structure is not yet confirmed.'} "
        f"Suggested horizon: {horizon}."
    )

    regime = get_market_regime(ctx)
    if regime.get("available"):
        narrative += (
            f" Market regime: {ctx.benchmark} "
            f"{'above' if regime.get('above_sma200') else 'below'} its 200-day average."
        )

    return Thesis(
        ticker=ticker,
        as_of=str(row.get("date", "")),
        source="rules",
        price=round(float(row["close"]), 2),
        trend_score=trend_score,
        direction=direction,
        conviction=round(conviction, 2),
        horizon=horizon,
        thesis=narrative,
        catalysts=catalysts,
        risks=risks,
        technicals=technicals,
        levels=levels,
        position=position,
        factor_scores=trend.get("factor_scores", {}),
        relative_strength=rel,
        invalidation=invalidation,
        citations=["get_trend_score", "get_technicals", "get_levels", "get_relative_strength"],
    )


def analyze(
    ctx: MarketContext,
    ticker: str,
    settings: dict[str, Any],
    provider: str | None = None,
) -> Thesis:
    """Dispatch to the configured analyst, falling back to rules on failure."""
    provider = (provider or settings.get("agent", {}).get("provider", "rules")).lower()
    base = analyze_rules(ctx, ticker, settings)
    if provider != "llm":
        return base

    from core.config import env_llm_config

    llm_cfg = env_llm_config(settings)
    if not llm_cfg:
        log.warning("provider=llm but no OPENAI_API_KEY set; using rules")
        return base

    try:
        from core.agent.llm import refine_with_llm

        return refine_with_llm(ctx, base, settings, llm_cfg)
    except Exception as exc:  # pragma: no cover - network dependent
        log.warning("LLM analysis failed for %s (%s); using rules", ticker, exc)
        base.source = "rules (llm failed)"
        return base
