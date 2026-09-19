"""Walk-forward backtester for the trend screen.

Design goals
------------
* **No look-ahead.** Metrics at each rebalance come from ``metrics_at(df, pos)``,
  which only sees data up to that date.
* **No survivorship shortcut.** Delisted/renamed tickers use their last
  available print; tickers that IPO'd mid-sample simply don't exist before then.
* **Costs are real.** Turnover-based commission + slippage is charged every
  rebalance for both the sell and the buy.
* **Evidence, not just returns.** We report the information coefficient (rank
  correlation between trend score and forward return) and the top-vs-bottom
  quintile spread, which is what actually tells you if the *signal* works.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np
import pandas as pd

from core.config import load_settings
from core.features.indicators import add_indicators
from core.features.metrics import metrics_at, price_at, trend_quality
from core.features.trending import rank_trending
from core.ingest.prices import fetch_prices
from core.universe.point_in_time import dollar_volume_frame, universe_at_from_frame
from core.universe import resolve_universe

log = logging.getLogger(__name__)

ProgressFn = Callable[[str, float], None]


@dataclass
class BacktestConfig:
    start: str | None = None          # YYYY-MM-DD (default: earliest available)
    end: str | None = None
    lookback: str = "10y"
    rebalance_days: int = 21          # ~monthly
    holding_days: int = 21
    top_n: int = 10
    cost_bps: float = 10.0            # one-way commission + slippage, basis points
    min_history_days: int = 130
    folds: int = 4                    # walk-forward sub-periods
    weight_mode: str = "equal"        # equal | score
    random_draws: int = 25            # Monte-Carlo control: random baskets per period

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _noop(_m: str, _p: float) -> None:  # pragma: no cover
    return None


def build_schedule(
    full_index: pd.DatetimeIndex, config: "BacktestConfig"
) -> list[tuple[int, int, pd.Timestamp, pd.Timestamp]]:
    """Rebalance dates as (entry_pos, exit_pos, entry_date, exit_date) tuples."""
    n_full = len(full_index)
    start_ts = pd.Timestamp(config.start) if config.start else full_index[0]
    end_ts = pd.Timestamp(config.end) if config.end else full_index[-1]

    schedule: list[tuple[int, int, pd.Timestamp, pd.Timestamp]] = []
    i = config.min_history_days
    while i + config.holding_days < n_full:
        as_of = full_index[i]
        if as_of > end_ts:
            break
        if as_of >= start_ts:
            schedule.append((i, i + config.holding_days, as_of, full_index[i + config.holding_days]))
        i += config.rebalance_days
    return schedule


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    total = weights.sum()
    if total <= 0:
        return float(np.mean(values))
    return float(np.sum(values * weights) / total)


def _max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    return float(dd.min())


def _stats(returns: np.ndarray, bench_returns: np.ndarray, ppy: float) -> dict[str, Any]:
    returns = np.asarray(returns, dtype=float)
    bench_returns = np.asarray(bench_returns, dtype=float)
    n = len(returns)
    if n == 0:
        return {"periods": 0}

    excess = returns - bench_returns
    equity = np.cumprod(1.0 + returns)
    total = float(equity[-1] - 1.0) if n else 0.0
    years = n / ppy if ppy else 0.0
    cagr = float((1.0 + total) ** (1.0 / years) - 1.0) if years > 0 and total > -1 else float("nan")

    vol = float(returns.std(ddof=1) * np.sqrt(ppy)) if n > 1 else float("nan")
    sharpe = (
        float(returns.mean() / returns.std(ddof=1) * np.sqrt(ppy))
        if n > 1 and returns.std(ddof=1) > 0
        else float("nan")
    )
    downside = returns[returns < 0]
    sortino = (
        float(returns.mean() / downside.std(ddof=1) * np.sqrt(ppy))
        if len(downside) > 1 and downside.std(ddof=1) > 0
        else float("nan")
    )
    ir = (
        float(excess.mean() / excess.std(ddof=1) * np.sqrt(ppy))
        if n > 1 and excess.std(ddof=1) > 0
        else float("nan")
    )
    bench_equity = np.cumprod(1.0 + bench_returns)

    return {
        "periods": n,
        "total_return": _round(total),
        "benchmark_total_return": _round(float(bench_equity[-1] - 1.0)),
        "alpha": _round(total - float(bench_equity[-1] - 1.0)),
        "cagr": _round(cagr),
        "ann_vol": _round(vol),
        "sharpe": _round(sharpe),
        "sortino": _round(sortino),
        "max_drawdown": _round(_max_drawdown(equity)),
        "hit_rate": _round(float((returns > 0).mean())),
        "win_rate_vs_bench": _round(float((excess > 0).mean())),
        "mean_excess_per_period": _round(float(excess.mean())),
        "information_ratio": _round(ir),
        "best_period": _round(float(returns.max())),
        "worst_period": _round(float(returns.min())),
    }


def _round(value: Any, digits: int = 4) -> Any:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if (np.isnan(f) or np.isinf(f)) else round(f, digits)


def run_backtest(
    settings: dict[str, Any] | None = None,
    config: BacktestConfig | None = None,
    synthetic: bool = False,
    use_cache: bool = True,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    config = config or BacktestConfig()
    settings = settings or load_settings()
    progress = progress or _noop

    screening = settings["screening"]
    weights = settings["scoring_weights"]
    benchmark = settings["benchmark"]
    ingest = settings.get("ingest", {}) or {}
    bt_universe_cfg = (settings.get("universe_config", {}) or {}).get("backtest", {}) or {}
    universe_mode = str(bt_universe_cfg.get("mode", "static")).strip().lower()
    pool_size = int(bt_universe_cfg.get("pool_size", 1000))
    liquidity_days = int(bt_universe_cfg.get("lookback_days", 20))

    # The pool is everything we hold prices for; the tradeable set may then be
    # re-derived per rebalance date (point-in-time) rather than fixed up front.
    resolution = resolve_universe(settings, allow_network=not synthetic, use_cache=use_cache)
    universe = list(resolution.tickers)
    if not universe:
        raise ValueError("universe is empty")

    errors: list[str] = list(resolution.errors)

    # --- Data -------------------------------------------------------------
    progress("Fetching history…", 0.02)
    tickers = [benchmark] + universe
    prices, ingest_errors = fetch_prices(
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
    errors.extend(ingest_errors)

    frames: dict[str, pd.DataFrame] = {}
    for ticker, df in prices.items():
        frames[ticker] = df if "sma20" in df.columns else add_indicators(df)

    if benchmark not in frames or len(frames[benchmark]) < config.min_history_days + config.holding_days + 10:
        raise RuntimeError(f"insufficient benchmark history for {benchmark}")

    bench = frames[benchmark]
    bench_close = bench["Close"]
    full_index = bench.index

    # --- Rebalance schedule (on the benchmark calendar) -------------------
    schedule = build_schedule(full_index, config)

    if not schedule:
        raise RuntimeError("no rebalance dates in range — widen the window or reduce min_history_days")

    log.info("backtest: %d rebalances over %d tickers", len(schedule), len(frames) - 1)

    # Point-in-time universe: precompute dollar volume once, then slice per date
    # so the universe is knowable on each rebalance date (no look-ahead).
    dv_frame = dollar_volume_frame(frames) if universe_mode == "point_in_time_liquidity" else None

    # --- Walk the schedule ------------------------------------------------
    periods: list[dict[str, Any]] = []
    gross_returns: list[float] = []
    net_returns: list[float] = []
    bench_returns: list[float] = []
    universe_returns: list[float] = []   # control: equal-weight all eligible names
    random_returns: list[float] = []     # control: random N-name baskets
    ics: list[float] = []
    quintile_returns: dict[int, list[float]] = {q: [] for q in range(5)}
    prev_picks: set[str] = set()

    for step, (i, j, as_of, exit_date) in enumerate(schedule):
        progress(f"Simulating {as_of.date()} → {exit_date.date()}", 0.05 + 0.85 * (step / len(schedule)))

        bench_ret_3m = (
            float(bench_close.iloc[i] / bench_close.iloc[i - 63] - 1.0) if i >= 63 else np.nan
        )
        bench_fwd = float(bench_close.iloc[j] / bench_close.iloc[i] - 1.0)

        rows: list[dict[str, Any]] = []
        fwd: dict[str, float] = {}

        if dv_frame is not None:
            active = universe_at_from_frame(
                dv_frame, as_of, pool_size, lookback_days=liquidity_days, exclude={benchmark}
            )
            if not active:
                active = universe  # fail soft rather than producing an empty book
        else:
            active = universe

        for ticker in active:
            df = frames.get(ticker)
            if df is None or len(df) < config.min_history_days:
                continue
            entry_pos, entry_price = price_at(df, as_of)
            exit_pos, exit_price = price_at(df, exit_date)
            if entry_pos is None or exit_pos is None or exit_pos <= entry_pos:
                continue
            if entry_pos < config.min_history_days:
                continue
            metrics = metrics_at(df, entry_pos, bench_ret_3m=bench_ret_3m)
            if metrics is None:
                continue
            metrics["ticker"] = ticker
            metrics["trend_quality_raw"] = trend_quality(metrics)
            rows.append(metrics)
            fwd[ticker] = float(exit_price / entry_price - 1.0)

        period: dict[str, Any] = {
            "date": as_of.date().isoformat(),
            "exit_date": exit_date.date().isoformat(),
            "benchmark_return": _round(bench_fwd),
            "picks": [],
            "gross_return": 0.0,
            "net_return": 0.0,
            "turnover": 0.0,
            "ic": None,
        }

        if rows:
            table = pd.DataFrame(rows).set_index("ticker")
            ranked = rank_trending(table, weights, screening)
        else:
            ranked = pd.DataFrame()

        if not ranked.empty:
            ranked = ranked.copy()
            ranked["fwd_return"] = [fwd[t] for t in ranked.index]

            # Information coefficient: does the score rank forward returns?
            # Spearman == Pearson on ranks (avoids a scipy dependency).
            if len(ranked) >= 8 and ranked["fwd_return"].std() > 0:
                ic = (
                    ranked["trend_score"].rank()
                    .corr(ranked["fwd_return"].rank())
                )
                if pd.notna(ic):
                    ics.append(float(ic))
                    period["ic"] = _round(float(ic))

            # Quintile spread across the whole eligible universe.
            try:
                quintiles = pd.qcut(ranked["trend_score"], 5, labels=False, duplicates="drop")
                for q, group in ranked.groupby(quintiles):
                    quintile_returns[int(q)].append(float(group["fwd_return"].mean()))
            except ValueError:  # too few distinct scores
                pass

            picks = ranked.head(config.top_n)
            pick_tickers = list(picks.index)
            returns = picks["fwd_return"].to_numpy(dtype=float)
            if config.weight_mode == "score":
                w = picks["trend_score"].to_numpy(dtype=float)
                gross = _weighted_mean(returns, w)
            else:
                gross = float(returns.mean())

            current = set(pick_tickers)
            if prev_picks:
                turnover = len(current - prev_picks) / max(1, len(current))
            else:
                turnover = 1.0
            prev_picks = current

            cost = turnover * 2.0 * (config.cost_bps / 10_000.0)
            net = gross - cost

            period.update(
                {
                    "picks": pick_tickers,
                    "gross_return": _round(gross),
                    "net_return": _round(net),
                    "turnover": _round(turnover),
                }
            )
            gross_returns.append(gross)
            net_returns.append(net)

            # --- Controls: is the ranking adding anything? -----------------
            eligible = ranked["fwd_return"].to_numpy(dtype=float)
            universe_returns.append(float(eligible.mean()))
            rng = np.random.default_rng(10_000 + step)
            k = min(config.top_n, len(eligible))
            draws = [float(rng.choice(eligible, size=k, replace=False).mean()) for _ in range(config.random_draws)]
            random_returns.append(float(np.mean(draws)))
        else:
            gross_returns.append(0.0)
            net_returns.append(0.0)
            universe_returns.append(0.0)
            random_returns.append(0.0)

        bench_returns.append(bench_fwd)
        periods.append(period)

    ppy = 252.0 / config.holding_days

    # --- Aggregate --------------------------------------------------------
    progress("Computing statistics…", 0.92)
    summary = {
        "net": _stats(np.array(net_returns), np.array(bench_returns), ppy),
        "gross": _stats(np.array(gross_returns), np.array(bench_returns), ppy),
        "benchmark": _stats(np.array(bench_returns), np.array(bench_returns), ppy),
        "random": _stats(np.array(random_returns), np.array(bench_returns), ppy),
        "universe": _stats(np.array(universe_returns), np.array(bench_returns), ppy),
    }

    net_arr = np.array(net_returns)
    gross_arr = np.array(gross_returns)
    bench_arr = np.array(bench_returns)
    random_arr = np.array(random_returns)
    universe_arr = np.array(universe_returns)

    signal = _signal_quality(ics, quintile_returns)

    folds = _fold_stats(net_returns, bench_returns, schedule, config.folds, ppy)

    result: dict[str, Any] = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "as_of": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "config": config.to_dict(),
        "benchmark": benchmark,
        "universe": resolution.to_dict(),
        "universe_size": len(universe),
        "schedule": {
            "rebalances": len(schedule),
            "first": schedule[0][2].date().isoformat(),
            "last": schedule[-1][3].date().isoformat(),
            "holding_days": config.holding_days,
        },
        "summary": summary,
        "signal_quality": signal,
        "folds": folds,
        "equity_curve": {
            "dates": [p["exit_date"] for p in periods],
            "net": [float(x) for x in np.cumprod(1.0 + net_arr)],
            "gross": [float(x) for x in np.cumprod(1.0 + gross_arr)],
            "benchmark": [float(x) for x in np.cumprod(1.0 + bench_arr)],
            "random": [float(x) for x in np.cumprod(1.0 + random_arr)],
            "universe": [float(x) for x in np.cumprod(1.0 + universe_arr)],
        },
        "periods": periods,
        "errors": errors,
    }

    from core.backtest.report import render_backtest_markdown

    result["markdown"] = render_backtest_markdown(result)
    progress("Done.", 1.0)
    return result


def _signal_quality(ics: list[float], quintile_returns: dict[int, list[float]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if ics:
        arr = np.array(ics)
        out["ic_mean"] = _round(float(arr.mean()))
        out["ic_std"] = _round(float(arr.std(ddof=1))) if len(arr) > 1 else None
        out["ic_ir"] = (
            _round(float(arr.mean() / arr.std(ddof=1) * np.sqrt(12)))
            if len(arr) > 1 and arr.std(ddof=1) > 0
            else None
        )
        out["ic_positive_share"] = _round(float((arr > 0).mean()))
        out["ic_periods"] = len(arr)

    q_means = {q: (float(np.mean(v)) if v else None) for q, v in quintile_returns.items()}
    out["quintile_mean_forward_return"] = {f"q{q + 1}": _round(v) for q, v in q_means.items()}
    top, bottom = q_means.get(4), q_means.get(0)
    if top is not None and bottom is not None:
        out["top_minus_bottom_spread"] = _round(top - bottom)
    return out


def _fold_stats(
    net_returns: list[float],
    bench_returns: list[float],
    schedule: list[tuple[int, int, pd.Timestamp, pd.Timestamp]],
    folds: int,
    ppy: float,
) -> list[dict[str, Any]]:
    """Split the timeline into contiguous out-of-sample windows."""
    n = len(net_returns)
    if folds < 2 or n < folds * 2:
        return []
    edges = np.linspace(0, n, folds + 1).astype(int)
    out: list[dict[str, Any]] = []
    for k in range(folds):
        a, b = edges[k], edges[k + 1]
        if b <= a:
            continue
        stats = _stats(np.array(net_returns[a:b]), np.array(bench_returns[a:b]), ppy)
        stats["fold"] = k + 1
        stats["start"] = schedule[a][2].date().isoformat()
        stats["end"] = schedule[b - 1][3].date().isoformat()
        out.append(stats)
    return out
