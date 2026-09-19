"""Factor attribution: is the edge skill, or just beta and sector exposure?

Three questions:

1. **Market alpha.** Regress strategy returns on the benchmark. If the intercept
   (Jensen's alpha) is ~0 once beta is accounted for, the strategy is a levered
   index fund with extra steps.
2. **Sector-neutral benchmark.** Build a benchmark that holds the *same sector
   weights* as the strategy but picks names at random within each sector. If the
   strategy does not beat it, the edge is sector allocation, not stock selection.
3. **Turnover reduction.** Hysteresis bands and longer holds cut trading costs.
   If no cost-viable variant exists, the signal is not tradable regardless of alpha.

Overlapping return windows make naive t-stats too generous, so all regressions use
Newey-West (HAC) standard errors.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from core.backtest.engine import BacktestConfig
from core.backtest.robustness import Panel, _summarize, build_panel, evaluate_panel
from core.backtest.sectors import get_sectors, sector_weights
from core.config import load_settings
from core.features.indicators import add_indicators
from core.ingest.prices import fetch_prices
from core.universe import resolve_universe

log = logging.getLogger(__name__)

ProgressFn = Callable[[str, float], None]


def _noop(_m: str, _p: float) -> None:  # pragma: no cover
    return None


# --------------------------------------------------------------------------- #
# OLS with Newey-West standard errors
# --------------------------------------------------------------------------- #

def ols_nw(y: Sequence[float], X: np.ndarray, lags: int | None = None) -> dict[str, Any]:
    """OLS with Newey-West (HAC) covariance. ``X`` should include a constant."""
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, k = X.shape
    if n <= k:
        return {"n": n, "params": [None] * k, "se": [None] * k, "t": [None] * k, "r2": None}

    xtx_inv = np.linalg.pinv(X.T @ X)
    params = xtx_inv @ (X.T @ y)
    resid = y - X @ params

    if lags is None:
        lags = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    lags = max(0, min(lags, n - 1))

    # Lag-0 term
    xr = X * resid[:, None]
    S = xr.T @ xr
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        G = (X[:-lag] * resid[lag:, None]).T @ (X[lag:] * resid[:-lag, None])
        S += w * (G + G.T)

    cov = xtx_inv @ S @ xtx_inv
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, params / se, np.nan)

    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None

    return {
        "n": n,
        "lags": lags,
        "params": [float(p) for p in params],
        "se": [float(s) for s in se],
        "t": [None if np.isnan(v) else float(v) for v in t],
        "r2": r2,
    }


def market_attribution(
    net_returns: Sequence[float], bench_returns: Sequence[float], ppy: float, lags: int | None = None
) -> dict[str, Any]:
    """Jensen's alpha and beta from regressing strategy returns on the benchmark."""
    reg = ols_nw(net_returns, np.column_stack([np.ones(len(net_returns)), bench_returns]), lags=lags)
    alpha = reg["params"][0]
    beta = reg["params"][1]
    return {
        "alpha_per_period": _r(alpha),
        "alpha_annualized": _r(alpha * ppy) if alpha is not None else None,
        "alpha_t": reg["t"][0],
        "beta": _r(beta),
        "beta_t": reg["t"][1],
        "r_squared": _r(reg["r2"]),
        "n": reg["n"],
        "hac_lags": reg["lags"],
        "significant_alpha": bool(
            alpha is not None and alpha > 0 and reg["t"][0] is not None and reg["t"][0] >= 2.0
        ),
    }


def excess_stats(excess: Sequence[float], ppy: float) -> dict[str, Any]:
    """Mean excess return with a t-stat."""
    arr = np.asarray(excess, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n == 0:
        return {"n": 0}
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if n > 1 else float("nan")
    t = mean / (sd / np.sqrt(n)) if n > 1 and sd > 0 else None
    return {
        "n": n,
        "mean_per_period": _r(mean),
        "mean_annualized": _r(mean * ppy),
        "std": _r(sd),
        "t_stat": _r(t),
        "positive_share": _r(float((arr > 0).mean())),
        "significant": bool(t is not None and t >= 2.0 and mean > 0),
    }


# --------------------------------------------------------------------------- #
# Sector-neutral benchmark
# --------------------------------------------------------------------------- #

def sector_neutral_returns(
    panel: Panel,
    evaluation: dict[str, Any],
    sectors: dict[str, str],
) -> list[float]:
    """Return series for a benchmark holding the strategy's sector weights.

    Within each sector we use the equal-weight average of all eligible names in
    that period, so the benchmark has the strategy's *allocation* but none of its
    *selection*.
    """
    out: list[float] = []
    for step in range(panel.n_periods):
        period = evaluation["periods"][step]
        picks = period.get("picks") or []
        eligible = period.get("eligible") or []
        if not picks or not eligible:
            out.append(0.0)
            continue

        weights = sector_weights(picks, sectors)
        by_sector: dict[str, list[float]] = defaultdict(list)
        for t in eligible:
            f = panel.fwd.get(t, [None] * panel.n_periods)[step]
            if f is not None:
                by_sector[sectors.get(t, "Unknown")].append(float(f))

        total = 0.0
        covered = 0.0
        for sector, w in weights.items():
            rets = by_sector.get(sector)
            if rets:
                total += w * float(np.mean(rets))
                covered += w
        out.append(total / covered if covered > 0 else 0.0)
    return out


def aggregate_sector_weights(evaluation: dict[str, Any], sectors: dict[str, str]) -> dict[str, float]:
    """Average sector exposure across all periods."""
    totals: dict[str, float] = defaultdict(float)
    n = 0
    for period in evaluation["periods"]:
        picks = period.get("picks") or []
        if not picks:
            continue
        for sector, w in sector_weights(picks, sectors).items():
            totals[sector] += w
        n += 1
    if n == 0:
        return {}
    return {s: round(v / n, 4) for s, v in sorted(totals.items(), key=lambda kv: -kv[1])}


# --------------------------------------------------------------------------- #
# Turnover reduction
# --------------------------------------------------------------------------- #

def turnover_variants(
    frames: dict[str, pd.DataFrame],
    benchmark: str,
    universe: Sequence[str],
    weights: dict[str, float],
    screening: dict[str, Any],
    base_config: BacktestConfig,
    top_n: int = 10,
    rebalance_list: Sequence[int] = (21, 63),
    cost_grid: Sequence[float] = (10.0, 30.0, 50.0),
    variants: Sequence[tuple[str, float]] = (("topn", 0.0), ("hysteresis", 1.5), ("hysteresis", 2.5)),
    sectors: dict[str, str] | None = None,
    scoring_list: Sequence[str] = ("pooled",),
    sector_weighting: str = "universe",
    progress: ProgressFn | None = None,
) -> list[dict[str, Any]]:
    """Compare selection rules on turnover, cost survival and sector-neutral alpha."""
    progress = progress or _noop
    out: list[dict[str, Any]] = []
    active_scoring = [s for s in scoring_list if s != "sector_neutral" or sectors]

    for idx, reb in enumerate(rebalance_list):
        progress(f"Turnover variants @ {reb}d…", idx / max(1, len(rebalance_list)))
        cfg = BacktestConfig(
            **{**base_config.to_dict(), "rebalance_days": int(reb), "holding_days": int(reb)}
        )
        try:
            panel = build_panel(frames, benchmark, cfg, universe)
        except RuntimeError:
            continue

        ppy = 252.0 / max(1, reb)
        for scoring in active_scoring:
            for selection, exit_multiple in variants:
                kwargs = {
                    "selection": selection,
                    "exit_multiple": exit_multiple,
                    "scoring": scoring,
                    "sectors": sectors,
                    "sector_weighting": sector_weighting,
                }
                ev_costs: dict[float, dict[str, Any]] = {}
                for cost in cost_grid:
                    ev = evaluate_panel(
                        panel, weights, screening, top_n=top_n, cost_bps=float(cost), **kwargs
                    )
                    ev_costs[float(cost)] = _summarize(ev, ppy)

                ev_free = evaluate_panel(
                    panel, weights, screening, top_n=top_n, cost_bps=0.0, **kwargs
                )
                turnovers = [p["turnover"] for p in ev_free["periods"] if p.get("picks")]
                avg_turnover = float(np.mean(turnovers)) if turnovers else 0.0

                base_ev = evaluate_panel(
                    panel, weights, screening, top_n=top_n, cost_bps=float(cost_grid[0]), **kwargs
                )
                attribution = market_attribution(
                    base_ev["net_returns"], base_ev["bench_returns"], ppy
                )
                free_summary = _summarize(ev_free, ppy)

                sector_attr: dict[str, Any] = {}
                if sectors:
                    sector_bench = sector_neutral_returns(panel, base_ev, sectors)
                    sector_attr = excess_stats(
                        np.array(base_ev["net_returns"]) - np.array(sector_bench), ppy
                    )

                out.append(
                    {
                        "rebalance_days": int(reb),
                        "scoring": scoring,
                        "selection": selection,
                        "exit_multiple": float(exit_multiple),
                        "avg_turnover": _r(avg_turnover),
                        "gross_total": free_summary["gross_total"],
                        "edge_by_cost": {str(int(c)): ev_costs[c]["edge_vs_universe"] for c in ev_costs},
                        "alpha_annualized": attribution["alpha_annualized"],
                        "alpha_t": attribution["alpha_t"],
                        "beta": attribution["beta"],
                        "sector_neutral_annualized": sector_attr.get("mean_annualized"),
                        "sector_neutral_t": sector_attr.get("t_stat"),
                        "sector_neutral_significant": sector_attr.get("significant"),
                        "sharpe": ev_costs[float(cost_grid[0])]["sharpe"],
                    }
                )

    return out


# --------------------------------------------------------------------------- #
# Suite
# --------------------------------------------------------------------------- #

def run_attribution_suite(
    settings: dict[str, Any] | None = None,
    config: BacktestConfig | None = None,
    synthetic: bool = False,
    use_cache: bool = True,
    cost_grid: Sequence[float] = (10.0, 30.0, 50.0),
    rebalance_list: Sequence[int] = (21, 63),
    sector_weighting: str = "universe",
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    settings = settings or load_settings()
    config = config or BacktestConfig()
    progress = progress or _noop

    weights = settings["scoring_weights"]
    screening = settings["screening"]
    benchmark = settings["benchmark"]
    ingest = settings.get("ingest", {}) or {}

    # Universe resolution has exactly one entry point (AGENT.md §5.5), so
    # attribution honours universe_config.mode (static/dynamic/hybrid) rather
    # than reading settings["universe"] directly.
    resolution = resolve_universe(settings, allow_network=not synthetic, use_cache=use_cache)
    universe = [t for t in resolution.tickers if t != benchmark]
    if not universe:
        raise ValueError("universe is empty")

    progress("Fetching history…", 0.02)
    prices, errors = fetch_prices(
        [benchmark] + universe, period=config.lookback, interval="1d",
        use_cache=use_cache, synthetic=synthetic,
        retry_periods=ingest.get("retry_periods"),
        on_missing=ingest.get("on_missing", "drop"),
        batch_size=int(ingest.get("batch_size", 40)),
        batch_pause_seconds=float(ingest.get("batch_pause_seconds", 0.5)),
    )
    errors = list(resolution.errors) + list(errors)
    frames = {t: (df if "sma20" in df.columns else add_indicators(df)) for t, df in prices.items()}
    if benchmark not in frames:
        raise RuntimeError(f"missing benchmark {benchmark}")

    progress("Sector lookup…", 0.10)
    sectors = get_sectors(universe, allow_network=not synthetic)

    progress("Building panel…", 0.15)
    panel = build_panel(frames, benchmark, config, universe)
    ppy = 252.0 / max(1, config.holding_days)

    progress("Market attribution…", 0.25)
    base_eval = evaluate_panel(
        panel, weights, screening, top_n=config.top_n,
        cost_bps=config.cost_bps, weight_mode=config.weight_mode,
    )
    base_summary = _summarize(base_eval, ppy)
    market = market_attribution(base_eval["net_returns"], base_eval["bench_returns"], ppy)

    progress("Sector-neutral benchmark…", 0.40)
    sector_rets = sector_neutral_returns(panel, base_eval, sectors)
    sector_excess = np.array(base_eval["net_returns"]) - np.array(sector_rets)
    sector_attr = excess_stats(sector_excess, ppy)
    sector_weights_avg = aggregate_sector_weights(base_eval, sectors)

    from core.backtest.rotation import rotation_skill

    rotation = rotation_skill(panel, base_eval, sectors)

    # How much of the universe's own return is explained by the market?
    universe_market = market_attribution(base_eval["universe_returns"], base_eval["bench_returns"], ppy)

    progress("Turnover reduction…", 0.55)
    variants = turnover_variants(
        frames, benchmark, universe, weights, screening, config,
        top_n=config.top_n, rebalance_list=rebalance_list, cost_grid=cost_grid,
        sectors=sectors,
        scoring_list=("pooled", "sector_neutral"),
        sector_weighting=sector_weighting,
        progress=lambda m, p: progress(m, 0.55 + 0.35 * p),
    )

    verdict = _build_attribution_verdict(market, sector_attr, variants, cost_grid)

    result: dict[str, Any] = {
        "run_id": pd.Timestamp.now("UTC").strftime("%Y%m%dT%H%M%SZ"),
        "config": config.to_dict(),
        "benchmark": benchmark,
        "universe": resolution.to_dict(),
        "universe_size": len(universe),
        "baseline": base_summary,
        "market": market,
        "universe_market": universe_market,
        "sector_neutral": sector_attr,
        "sector_weights": sector_weights_avg,
        "rotation": rotation,
        "sectors": sectors,
        "turnover_variants": variants,
        "verdict": verdict,
        "errors": errors,
    }

    from core.backtest.attribution_report import render_attribution_markdown

    result["markdown"] = render_attribution_markdown(result)
    progress("Done.", 1.0)
    return result


def _build_attribution_verdict(
    market: dict[str, Any],
    sector_attr: dict[str, Any],
    variants: list[dict[str, Any]],
    cost_grid: Sequence[float],
) -> dict[str, Any]:
    max_cost = str(int(max(cost_grid)))
    cost_viable = [
        v for v in variants
        if (v.get("edge_by_cost") or {}).get(max_cost) is not None
        and (v["edge_by_cost"][max_cost] or 0) > 0
    ]

    significant = [v for v in variants if v.get("sector_neutral_significant")]
    best_sig = (
        max(significant, key=lambda v: v.get("sector_neutral_annualized") or -9)
        if significant else None
    )
    best_any = max(
        (v for v in variants if v.get("sector_neutral_annualized") is not None),
        key=lambda v: v.get("sector_neutral_annualized") or -9,
        default=None,
    )

    if best_sig is not None:
        sel_detail = (
            f"{best_sig.get('scoring')}/{best_sig.get('rebalance_days')}d/{best_sig.get('selection')} "
            f"retains {_pct_str(best_sig.get('sector_neutral_annualized'))} ann. "
            f"(t={_fmt(best_sig.get('sector_neutral_t'))})"
        )
    else:
        sel_detail = (
            "no variant clears t=2 — best is "
            f"{_pct_str(best_any.get('sector_neutral_annualized'))} "
            f"(t={_fmt(best_any.get('sector_neutral_t'))})"
            if best_any else "no variants evaluated"
        )

    checks = [
        {
            "name": "Market alpha (Jensen)",
            "passed": bool(market.get("significant_alpha")),
            "detail": (
                f"alpha {_pct_str(market.get('alpha_annualized'))} ann., "
                f"t={_fmt(market.get('alpha_t'))}, beta={_fmt(market.get('beta'))}"
            ),
        },
        {
            "name": "Stock selection (sector-neutral)",
            "passed": bool(significant),
            "detail": sel_detail,
        },
        {
            "name": f"Cost-viable variant @ {max_cost}bps",
            "passed": len(cost_viable) > 0,
            "detail": (
                f"{len(cost_viable)}/{len(variants)} variants keep a positive edge at {max_cost} bps"
                if variants else "no variants evaluated"
            ),
        },
    ]
    n_passed = sum(1 for c in checks if c["passed"])
    overall = "PASS" if n_passed == 3 else ("PARTIAL" if n_passed == 2 else "FAIL")

    comparison = _scoring_comparison(variants)
    return {"checks": checks, "n_passed": n_passed, "overall": overall, "scoring_comparison": comparison}


def _scoring_comparison(variants: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate pooled vs sector-neutral across all variants."""
    out: dict[str, Any] = {}
    for mode in ("pooled", "sector_neutral"):
        rows = [v for v in variants if v.get("scoring", "pooled") == mode]
        if not rows:
            continue
        sec = [v["sector_neutral_annualized"] for v in rows if v.get("sector_neutral_annualized") is not None]
        ts = [v["sector_neutral_t"] for v in rows if v.get("sector_neutral_t") is not None]
        out[mode] = {
            "variants": len(rows),
            "mean_turnover": _r(float(np.mean([v["avg_turnover"] for v in rows if v.get("avg_turnover") is not None]))),
            "mean_sector_neutral_annualized": _r(float(np.mean(sec))) if sec else None,
            "best_sector_neutral_t": _r(max(ts)) if ts else None,
            "significant_variants": sum(1 for v in rows if v.get("sector_neutral_significant")),
        }
    return out


def _pct_str(v: Any) -> str:
    if v is None:
        return "n/a"
    try:
        return f"{float(v) * 100:+.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def _fmt(v: Any, digits: int = 2) -> str:
    if v is None:
        return "n/a"
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _r(value: Any, digits: int = 4) -> Any:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if (np.isnan(f) or np.isinf(f)) else round(f, digits)
