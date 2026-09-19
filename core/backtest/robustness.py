"""Robustness gate for the trend screen.

Before investing in a point-in-time universe rebuild, answer one question cheaply:
*is the edge broad and stable, or a fluke of a handful of names and one parameter
setting?*

Three tests:

1. **Universe bootstrap** — re-run on many random subsets of the universe. A real
   signal survives dropping 30% of names; a mirage does not.
2. **Parameter plateau** — sweep ``top_n`` and ``rebalance_days``. Real edges
   degrade gracefully; curve-fits spike at one setting and cliff elsewhere.
3. **Cost sensitivity** — push commission/slippage up. Momentum is turnover-heavy;
   an edge that dies at 30 bps is not tradable.

Everything runs off a precomputed :class:`Panel`, so a 30-trial bootstrap costs
one pass over the data rather than thirty.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from core.backtest.engine import BacktestConfig, _stats, build_schedule
from core.config import load_settings
from core.features.indicators import add_indicators
from core.features.metrics import metrics_at, price_at, trend_quality
from core.features.trending import rank_trending
from core.ingest.prices import fetch_prices
from core.universe import resolve_universe
from core.universe.point_in_time import dollar_volume_frame, universe_at_from_frame

log = logging.getLogger(__name__)

ProgressFn = Callable[[str, float], None]


@dataclass
class Panel:
    """Metrics + forward returns for every (ticker, rebalance) pair."""

    tickers: list[str]
    schedule: list[tuple[int, int, pd.Timestamp, pd.Timestamp]]
    bench_returns: list[float]
    metrics: dict[str, list[dict | None]] = field(default_factory=dict)
    fwd: dict[str, list[float | None]] = field(default_factory=dict)

    @property
    def n_periods(self) -> int:
        return len(self.schedule)


def _noop(_m: str, _p: float) -> None:  # pragma: no cover
    return None


def active_by_period_for(
    panel: Panel,
    dv: pd.DataFrame,
    pool_size: int,
    lookback_days: int,
    benchmark: str,
) -> list[set[str]]:
    """Point-in-time eligible-ticker sets, one per rebalance date.

    Mirrors the engine's per-date universe so ``evaluate_panel`` and
    ``run_backtest`` stay consistent when the backtest universe is dynamic.
    """
    return [
        set(
            universe_at_from_frame(
                dv, as_of, pool_size, lookback_days=lookback_days, exclude={benchmark}
            )
        )
        for _, _, as_of, _ in panel.schedule
    ]


def build_panel(
    frames: dict[str, pd.DataFrame],
    benchmark: str,
    config: BacktestConfig,
    universe: Sequence[str],
) -> Panel:
    """Precompute every metric and forward return once."""
    bench = frames[benchmark]
    bench_close = bench["Close"]
    universe = [t for t in universe if t != benchmark]
    schedule = build_schedule(bench.index, config)
    if not schedule:
        raise RuntimeError("no rebalance dates in range")

    bench_returns = [
        float(bench_close.iloc[j] / bench_close.iloc[i] - 1.0) for i, j, _, _ in schedule
    ]
    bench_ret_3m = [
        float(bench_close.iloc[i] / bench_close.iloc[i - 63] - 1.0) if i >= 63 else np.nan
        for i, _, _, _ in schedule
    ]

    metrics: dict[str, list[dict | None]] = {t: [] for t in universe}
    fwd: dict[str, list[float | None]] = {t: [] for t in universe}

    for step, (_, _, as_of, exit_date) in enumerate(schedule):
        bret = bench_ret_3m[step]
        for t in universe:
            df = frames.get(t)
            row: dict | None = None
            f: float | None = None
            if df is not None and len(df) >= config.min_history_days:
                entry_pos, entry_price = price_at(df, as_of)
                exit_pos, exit_price = price_at(df, exit_date)
                if (
                    entry_pos is not None
                    and exit_pos is not None
                    and exit_pos > entry_pos
                    and entry_pos >= config.min_history_days
                ):
                    row = metrics_at(df, entry_pos, bench_ret_3m=bret)
                    if row is not None:
                        row["trend_quality_raw"] = trend_quality(row)
                        f = float(exit_price / entry_price - 1.0)
            metrics[t].append(row)
            fwd[t].append(f)

    return Panel(
        tickers=list(universe),
        schedule=schedule,
        bench_returns=bench_returns,
        metrics=metrics,
        fwd=fwd,
    )


def evaluate_panel(
    panel: Panel,
    weights: dict[str, float],
    screening: dict[str, Any],
    top_n: int = 10,
    cost_bps: float = 10.0,
    weight_mode: str = "equal",
    random_draws: int = 25,
    tickers: Sequence[str] | None = None,
    selection: str = "topn",
    exit_multiple: float = 1.5,
    scoring: str = "pooled",
    sectors: dict[str, str] | None = None,
    sector_weighting: str = "universe",
    active_by_period: Sequence[set[str]] | None = None,
) -> dict[str, Any]:
    """Run the strategy over a panel. Mirrors ``engine.run_backtest`` exactly.

    ``selection="hysteresis"`` keeps existing holdings until their rank falls
    below ``top_n * exit_multiple``, which cuts turnover substantially.

    ``scoring="sector_neutral"`` ranks each factor *within* sector and fixes
    sector quotas, so selection skill is measured independently of sector tilts.
    Requires ``sectors``.
    """
    sector_mode = scoring == "sector_neutral" and bool(sectors)
    subset = list(tickers if tickers is not None else panel.tickers)
    net: list[float] = []
    gross: list[float] = []
    univ: list[float] = []
    rnd: list[float] = []
    ics: list[float] = []
    quintiles: dict[int, list[float]] = {q: [] for q in range(5)}
    periods: list[dict[str, Any]] = []
    prev_picks: set[str] = set()

    for step in range(panel.n_periods):
        if active_by_period is None:
            active = subset
        else:
            allowed = active_by_period[step]
            active = [t for t in subset if t in allowed]
        rows: list[dict] = []
        fmap: dict[str, float] = {}
        for t in active:
            m = panel.metrics.get(t, [None] * panel.n_periods)[step]
            f = panel.fwd.get(t, [None] * panel.n_periods)[step]
            if m is None or f is None:
                continue
            rows.append({"ticker": t, **m})
            fmap[t] = f

        period: dict[str, Any] = {"net_return": 0.0, "gross_return": 0.0, "picks": [], "eligible": []}

        if rows:
            table = pd.DataFrame(rows).set_index("ticker")
            ranked = rank_trending(table, weights, screening)
        else:
            ranked = pd.DataFrame()

        if not ranked.empty:
            ranked = ranked.copy()
            ranked["fwd_return"] = [fmap[t] for t in ranked.index]

            score_col = "trend_score"
            if sector_mode:
                from core.backtest.sector_neutral import add_sector_scores

                ranked = add_sector_scores(ranked, weights, sectors)
                score_col = "sector_score"
            ordered = list(ranked.sort_values(score_col, ascending=False).index)

            if len(ranked) >= 8 and ranked["fwd_return"].std() > 0:
                ic = ranked[score_col].rank().corr(ranked["fwd_return"].rank())
                if pd.notna(ic):
                    ics.append(float(ic))

            try:
                q = pd.qcut(ranked[score_col], 5, labels=False, duplicates="drop")
                for key, group in ranked.groupby(q):
                    quintiles[int(key)].append(float(group["fwd_return"].mean()))
            except ValueError:
                pass

            # Target pick set, then optional hysteresis on top of it.
            if sector_mode:
                from core.backtest.sector_neutral import sector_neutral_target

                target = sector_neutral_target(ranked, ordered, top_n, sector_weighting)
            else:
                target = ordered[:top_n]

            if selection == "hysteresis" and prev_picks:
                pos = {t: idx for idx, t in enumerate(ordered)}
                exit_cut = max(top_n, int(round(top_n * exit_multiple)))
                held = [t for t in ordered if t in prev_picks and pos[t] < exit_cut][:top_n]
                new = [t for t in target if t not in held]
                pick_list = (held + new)[:top_n]
                if len(pick_list) < top_n:
                    pick_list += [t for t in ordered if t not in pick_list][: top_n - len(pick_list)]
            else:
                pick_list = list(target)

            picks = ranked.loc[pick_list]
            returns = picks["fwd_return"].to_numpy(dtype=float)
            if weight_mode == "score" and returns.size:
                w = picks[score_col].to_numpy(dtype=float)
                g = float(np.sum(returns * w) / np.sum(w)) if w.sum() > 0 else float(returns.mean())
            else:
                g = float(returns.mean())

            current = set(pick_list)
            turnover = len(current - prev_picks) / max(1, len(current)) if prev_picks else 1.0
            prev_picks = current
            cost = turnover * 2.0 * (cost_bps / 10_000.0)

            gross.append(g)
            net.append(g - cost)
            period.update(
                {
                    "net_return": g - cost,
                    "gross_return": g,
                    "picks": list(pick_list),
                    "eligible": ordered,
                    "turnover": turnover,
                }
            )

            eligible = ranked["fwd_return"].to_numpy(dtype=float)
            univ.append(float(eligible.mean()))
            rng = np.random.default_rng(10_000 + step)
            k = min(top_n, len(eligible))
            draws = [
                float(rng.choice(eligible, size=k, replace=False).mean())
                for _ in range(random_draws)
            ]
            rnd.append(float(np.mean(draws)))
        else:
            gross.append(0.0)
            net.append(0.0)
            univ.append(0.0)
            rnd.append(0.0)

        periods.append(period)

    return {
        "net_returns": net,
        "gross_returns": gross,
        "universe_returns": univ,
        "random_returns": rnd,
        "ics": ics,
        "quintiles": quintiles,
        "periods": periods,
        "bench_returns": list(panel.bench_returns),
    }


def _summarize(evaluation: dict[str, Any], ppy: float) -> dict[str, Any]:
    net = np.array(evaluation["net_returns"])
    univ = np.array(evaluation["universe_returns"])
    bench = np.array(evaluation["bench_returns"])
    gross = np.array(evaluation["gross_returns"])
    net_stats = _stats(net, bench, ppy)
    univ_stats = _stats(univ, bench, ppy)
    return {
        "net_total": net_stats.get("total_return"),
        "gross_total": _stats(gross, bench, ppy).get("total_return"),
        "universe_total": univ_stats.get("total_return"),
        "benchmark_total": net_stats.get("benchmark_total_return"),
        "edge_vs_universe": (net_stats.get("total_return") or 0.0) - (univ_stats.get("total_return") or 0.0),
        "sharpe": net_stats.get("sharpe"),
        "max_drawdown": net_stats.get("max_drawdown"),
        "win_rate_vs_bench": net_stats.get("win_rate_vs_bench"),
        "ic_mean": float(np.mean(evaluation["ics"])) if evaluation["ics"] else None,
    }


# --------------------------------------------------------------------------- #
# Test 1: universe bootstrap
# --------------------------------------------------------------------------- #

def universe_bootstrap(
    panel: Panel,
    weights: dict[str, float],
    screening: dict[str, Any],
    top_n: int = 10,
    cost_bps: float = 10.0,
    weight_mode: str = "equal",
    n_trials: int = 30,
    drop_frac: float = 0.30,
    seed: int = 0,
    active_by_period: Sequence[set[str]] | None = None,
) -> dict[str, Any]:
    """Re-run on many random subsets of the universe."""
    ppy = 252.0 / max(1, panel.schedule[1][1] - panel.schedule[1][0]) if panel.n_periods else 12.0
    rng = np.random.default_rng(seed)
    keep_n = max(5, int(round(len(panel.tickers) * (1.0 - drop_frac))))

    trials: list[dict[str, Any]] = []
    for _ in range(n_trials):
        subset = rng.choice(panel.tickers, size=keep_n, replace=False)
        evaluation = evaluate_panel(
            panel, weights, screening, top_n=top_n, cost_bps=cost_bps,
            weight_mode=weight_mode, tickers=list(subset),
            active_by_period=active_by_period,
        )
        trials.append(_summarize(evaluation, ppy))

    edges = np.array([t["edge_vs_universe"] for t in trials], dtype=float)
    totals = np.array([t["net_total"] or 0.0 for t in trials], dtype=float)
    sharpes = np.array([t["sharpe"] if t["sharpe"] is not None else np.nan for t in trials], dtype=float)
    ics = np.array([t["ic_mean"] if t["ic_mean"] is not None else np.nan for t in trials], dtype=float)

    return {
        "n_trials": n_trials,
        "drop_frac": drop_frac,
        "keep": keep_n,
        "edge_positive_share": float((edges > 0).mean()) if len(edges) else 0.0,
        "edge_mean": _r(float(np.nanmean(edges))),
        "edge_median": _r(float(np.nanmedian(edges))),
        "edge_std": _r(float(np.nanstd(edges, ddof=1))) if len(edges) > 1 else None,
        "edge_p5": _r(float(np.nanpercentile(edges, 5))) if len(edges) else None,
        "edge_p95": _r(float(np.nanpercentile(edges, 95))) if len(edges) else None,
        "edge_min": _r(float(np.nanmin(edges))) if len(edges) else None,
        "edge_max": _r(float(np.nanmax(edges))) if len(edges) else None,
        "net_total_mean": _r(float(np.nanmean(totals))) if len(totals) else None,
        "sharpe_mean": _r(float(np.nanmean(sharpes))) if len(sharpes) else None,
        "sharpe_std": _r(float(np.nanstd(sharpes, ddof=1))) if len(sharpes) > 1 else None,
        "ic_mean_of_means": _r(float(np.nanmean(ics))) if len(ics) else None,
        "trials": [{k: _r(v) if isinstance(v, (int, float)) else v for k, v in t.items()} for t in trials],
    }


# --------------------------------------------------------------------------- #
# Test 2: parameter plateau
# --------------------------------------------------------------------------- #

def parameter_sweep(
    frames: dict[str, pd.DataFrame],
    benchmark: str,
    universe: Sequence[str],
    weights: dict[str, float],
    screening: dict[str, Any],
    base_config: BacktestConfig,
    top_n_grid: Sequence[int] = (5, 10, 20, 30),
    rebalance_grid: Sequence[int] = (5, 21, 63),
    cost_bps: float = 10.0,
    weight_mode: str = "equal",
    progress: ProgressFn | None = None,
    dv_frame: pd.DataFrame | None = None,
    pool_size: int = 1000,
    liquidity_days: int = 20,
) -> dict[str, Any]:
    """Sweep top_n x rebalance_days. A real edge is a broad plateau, not a spike."""
    progress = progress or _noop
    cells: list[dict[str, Any]] = []
    pairs = [(r, r) for r in rebalance_grid]
    for idx, (reb, hold) in enumerate(pairs):
        progress(f"Parameter sweep {reb}d rebalance…", idx / max(1, len(pairs)))
        cfg = BacktestConfig(**{**base_config.to_dict(), "rebalance_days": int(reb), "holding_days": int(hold)})
        try:
            panel = build_panel(frames, benchmark, cfg, universe)
        except RuntimeError:
            continue
        ppy = 252.0 / max(1, hold)
        active = (
            active_by_period_for(panel, dv_frame, pool_size, liquidity_days, benchmark)
            if dv_frame is not None
            else None
        )
        for top_n in top_n_grid:
            evaluation = evaluate_panel(
                panel, weights, screening, top_n=int(top_n), cost_bps=cost_bps,
                weight_mode=weight_mode, active_by_period=active,
            )
            summary = _summarize(evaluation, ppy)
            cells.append(
                {
                    "rebalance_days": int(reb),
                    "holding_days": int(hold),
                    "top_n": int(top_n),
                    **summary,
                }
            )

    edges = np.array([c["edge_vs_universe"] for c in cells], dtype=float)
    return {
        "cells": cells,
        "n_cells": len(cells),
        "edge_positive_share": float((edges > 0).mean()) if len(edges) else 0.0,
        "edge_min": _r(float(np.nanmin(edges))) if len(edges) else None,
        "edge_median": _r(float(np.nanmedian(edges))) if len(edges) else None,
        "edge_max": _r(float(np.nanmax(edges))) if len(edges) else None,
        "edge_spread": _r(float(np.nanmax(edges) - np.nanmin(edges))) if len(edges) else None,
        "sharpe_median": _r(float(np.nanmedian([c["sharpe"] for c in cells if c["sharpe"] is not None])))
        if cells else None,
    }


# --------------------------------------------------------------------------- #
# Test 3: cost sensitivity
# --------------------------------------------------------------------------- #

def cost_sensitivity(
    panel: Panel,
    weights: dict[str, float],
    screening: dict[str, Any],
    top_n: int = 10,
    cost_grid: Sequence[float] = (0.0, 10.0, 30.0, 50.0),
    weight_mode: str = "equal",
    active_by_period: Sequence[set[str]] | None = None,
) -> dict[str, Any]:
    ppy = 252.0 / max(1, panel.schedule[1][1] - panel.schedule[1][0]) if panel.n_periods else 12.0
    rows: list[dict[str, Any]] = []
    for bps in cost_grid:
        evaluation = evaluate_panel(
            panel, weights, screening, top_n=top_n, cost_bps=float(bps),
            weight_mode=weight_mode, active_by_period=active_by_period,
        )
        summary = _summarize(evaluation, ppy)
        rows.append({"cost_bps": float(bps), **summary})

    max_cost_row = max(rows, key=lambda r: r["cost_bps"]) if rows else None
    return {
        "rows": rows,
        "edge_positive_at_max_cost": bool(max_cost_row and (max_cost_row["edge_vs_universe"] or 0.0) > 0),
        "max_cost_bps": max_cost_row["cost_bps"] if max_cost_row else None,
        "edge_at_max_cost": max_cost_row["edge_vs_universe"] if max_cost_row else None,
    }


# --------------------------------------------------------------------------- #
# Suite + verdict
# --------------------------------------------------------------------------- #

def run_robustness_suite(
    settings: dict[str, Any] | None = None,
    config: BacktestConfig | None = None,
    synthetic: bool = False,
    use_cache: bool = True,
    bootstrap_trials: int = 30,
    drop_frac: float = 0.30,
    top_n_grid: Sequence[int] = (5, 10, 20, 30),
    rebalance_grid: Sequence[int] = (5, 21, 63),
    cost_grid: Sequence[float] = (0.0, 10.0, 30.0, 50.0),
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    settings = settings or load_settings()
    config = config or BacktestConfig()
    progress = progress or _noop

    weights = settings["scoring_weights"]
    screening = settings["screening"]
    benchmark = settings["benchmark"]
    ingest = settings.get("ingest", {}) or {}
    bt_universe_cfg = (settings.get("universe_config", {}) or {}).get("backtest", {}) or {}
    universe_mode = str(bt_universe_cfg.get("mode", "static")).strip().lower()
    pool_size = int(bt_universe_cfg.get("pool_size", 1000))
    liquidity_days = int(bt_universe_cfg.get("lookback_days", 20))

    resolution = resolve_universe(settings, allow_network=not synthetic, use_cache=use_cache)
    universe = list(resolution.tickers)
    if not universe:
        raise ValueError("universe is empty")

    progress("Fetching history…", 0.02)
    tickers = [benchmark] + universe
    prices, errors = fetch_prices(
        tickers,
        period=config.lookback,
        interval="1d",
        use_cache=use_cache,
        synthetic=synthetic,
        retry_periods=ingest.get("retry_periods"),
        on_missing=ingest.get("on_missing", "drop"),
        batch_size=int(ingest.get("batch_size", 40)),
        batch_pause_seconds=float(ingest.get("batch_pause_seconds", 0.5)),
    )
    errors = list(resolution.errors) + list(errors)
    frames = {t: (df if "sma20" in df.columns else add_indicators(df)) for t, df in prices.items()}
    if benchmark not in frames:
        raise RuntimeError(f"missing benchmark {benchmark}")

    # Point-in-time universe support (must match the engine).
    dv_frame = dollar_volume_frame(frames) if universe_mode == "point_in_time_liquidity" else None

    progress("Building point-in-time panel…", 0.12)
    panel = build_panel(frames, benchmark, config, universe)
    base_active = (
        active_by_period_for(panel, dv_frame, pool_size, liquidity_days, benchmark)
        if dv_frame is not None
        else None
    )

    progress("Bootstrap: random universe subsets…", 0.25)
    bootstrap = universe_bootstrap(
        panel, weights, screening, top_n=config.top_n, cost_bps=config.cost_bps,
        weight_mode=config.weight_mode, n_trials=bootstrap_trials, drop_frac=drop_frac,
        active_by_period=base_active,
    )

    progress("Cost sensitivity…", 0.65)
    costs = cost_sensitivity(
        panel, weights, screening, top_n=config.top_n, cost_grid=cost_grid,
        weight_mode=config.weight_mode, active_by_period=base_active,
    )

    progress("Parameter sweep…", 0.75)
    sweep = parameter_sweep(
        frames, benchmark, universe, weights, screening, config,
        top_n_grid=top_n_grid, rebalance_grid=rebalance_grid,
        cost_bps=config.cost_bps, weight_mode=config.weight_mode,
        progress=lambda m, p: progress(m, 0.75 + 0.2 * p),
        dv_frame=dv_frame, pool_size=pool_size, liquidity_days=liquidity_days,
    )

    baseline = _summarize(
        evaluate_panel(
            panel, weights, screening, top_n=config.top_n, cost_bps=config.cost_bps,
            weight_mode=config.weight_mode, active_by_period=base_active,
        ),
        252.0 / max(1, config.holding_days),
    )

    verdict = _build_verdict(bootstrap, sweep, costs)

    result: dict[str, Any] = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "config": config.to_dict(),
        "benchmark": benchmark,
        "universe": resolution.to_dict(),
        "universe_size": len(universe),
        "baseline": baseline,
        "bootstrap": bootstrap,
        "sweep": sweep,
        "costs": costs,
        "verdict": verdict,
        "errors": errors,
    }

    from core.backtest.robustness_report import render_robustness_markdown

    result["markdown"] = render_robustness_markdown(result)
    progress("Done.", 1.0)
    return result


def _build_verdict(bootstrap: dict, sweep: dict, costs: dict) -> dict[str, Any]:
    checks = [
        {
            "name": "Universe bootstrap",
            "passed": bootstrap.get("edge_positive_share", 0.0) >= 0.80,
            "detail": (
                f"{bootstrap.get('edge_positive_share', 0):.0%} of "
                f"{bootstrap.get('n_trials', 0)} random subsets kept a positive edge"
            ),
        },
        {
            "name": "Parameter plateau",
            "passed": sweep.get("edge_positive_share", 0.0) >= 0.70,
            "detail": (
                f"{sweep.get('edge_positive_share', 0):.0%} of {sweep.get('n_cells', 0)} "
                f"parameter combos had a positive edge (spread "
                f"{(sweep.get('edge_spread') or 0) * 100:.0f}pp)"
            ),
        },
        {
            "name": "Cost survival",
            "passed": bool(costs.get("edge_positive_at_max_cost")),
            "detail": (
                f"edge at {costs.get('max_cost_bps')} bps = "
                f"{(costs.get('edge_at_max_cost') or 0) * 100:+.1f}%"
            ),
        },
    ]
    n_passed = sum(1 for c in checks if c["passed"])
    overall = "PASS" if n_passed == 3 else ("PARTIAL" if n_passed == 2 else "FAIL")
    return {"checks": checks, "n_passed": n_passed, "overall": overall}


def _r(value: Any, digits: int = 4) -> Any:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if (np.isnan(f) or np.isinf(f)) else round(f, digits)
