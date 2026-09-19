"""Same-day round-trip ("buy at open, sell at close") simulator.

For each trading day D:

1. Rank the universe using **only data through the previous close (D-1)** —
   ``metrics_at(df, pos)`` with ``pos`` pointing at D-1. No look-ahead.
2. Buy ``shares`` of the top ``picks`` names at day D's **open**.
3. Sell them at day D's **close**.
4. Record gross P&L, costs and net P&L.

Why the app's other backtest can't answer this: ``core/backtest/engine.py``
holds positions for whole periods and works in returns. This module models a
literal cash P&L for a fixed share count.

**Approximation.** The app only stores daily OHLCV, so "sell before the close"
is modelled as the official close and no intraday path is simulated. Real fills
would be worse (spread, slippage, the stock moving between your order and the
bell), so treat the output as an optimistic upper bound — especially given the
project's standing verdict that the underlying signal has no demonstrated edge.
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
from core.ingest.company_names import get_company_names
from core.ingest.prices import fetch_prices
from core.universe import resolve_universe

log = logging.getLogger(__name__)

ProgressFn = Callable[[str, float], None]


@dataclass
class DailySimConfig:
    start: str | None = None          # YYYY-MM-DD (inclusive)
    end: str | None = None            # YYYY-MM-DD (inclusive)
    lookback: str = "3y"              # history window to fetch
    picks: int = 2                    # how many names to buy each day
    shares: int = 100                 # fixed shares per pick
    cost_bps: float = 10.0            # one-way commission + slippage, basis points
    min_history_days: int = 130       # warm-up required before a name is eligible

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _noop(_m: str, _p: float) -> None:  # pragma: no cover
    return None


def _round(value: Any, digits: int = 4) -> Any:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if (np.isnan(f) or np.isinf(f)) else round(f, digits)


def simulate_frames(
    frames: dict[str, pd.DataFrame],
    benchmark: str,
    settings: dict[str, Any],
    config: DailySimConfig,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """Run the same-day simulation over already-loaded OHLCV frames.

    Split out from :func:`run_daily_sim` so tests can inject frames directly.
    """
    progress = progress or _noop
    screening = settings["screening"]
    weights = settings["scoring_weights"]

    if benchmark not in frames:
        raise RuntimeError(f"benchmark {benchmark} not in frames")
    bench = frames[benchmark]
    if "sma20" not in bench.columns:
        frames = {t: (df if "sma20" in df.columns else add_indicators(df)) for t, df in frames.items()}
        bench = frames[benchmark]

    bench_close = bench["Close"]
    full_index = bench.index
    n = len(full_index)
    if n < config.min_history_days + 2:
        raise RuntimeError("not enough history for the requested warm-up")

    start_ts = pd.Timestamp(config.start) if config.start else None
    end_ts = pd.Timestamp(config.end) if config.end else None
    tickers = [t for t in frames if t != benchmark]

    days: list[dict[str, Any]] = []
    all_picks: list[str] = []

    for i in range(config.min_history_days, n):
        trade_date = full_index[i]
        if start_ts is not None and trade_date < start_ts:
            continue
        if end_ts is not None and trade_date > end_ts:
            break

        signal_idx = i - 1
        signal_date = full_index[signal_idx]
        if i % 25 == 0:
            progress(f"Simulating {trade_date.date()}", 0.05 + 0.9 * (i / n))

        bench_ret_3m = (
            float(bench_close.iloc[signal_idx] / bench_close.iloc[signal_idx - 63] - 1.0)
            if signal_idx >= 63
            else np.nan
        )

        rows: list[dict[str, Any]] = []
        quotes: dict[str, tuple[float, float]] = {}

        for ticker in tickers:
            df = frames.get(ticker)
            if df is None or len(df) < config.min_history_days:
                continue
            sig_pos, _ = price_at(df, signal_date)
            if sig_pos is None or sig_pos < config.min_history_days - 1:
                continue
            trade_pos, close_price = price_at(df, trade_date)
            # Require an *exact* print on the trade date (price_at can ffill
            # across a holiday) and a valid open.
            if trade_pos is None or df.index[trade_pos] != trade_date:
                continue
            open_price = float(df["Open"].iloc[trade_pos])
            if not np.isfinite(open_price) or open_price <= 0 or not np.isfinite(close_price):
                continue
            metrics = metrics_at(df, sig_pos, bench_ret_3m=bench_ret_3m)
            if metrics is None:
                continue
            metrics["ticker"] = ticker
            metrics["trend_quality_raw"] = trend_quality(metrics)
            rows.append(metrics)
            quotes[ticker] = (open_price, float(close_price))

        day: dict[str, Any] = {
            "date": trade_date.date().isoformat(),
            "signal_date": signal_date.date().isoformat(),
            "picks": [],
            "capital": 0.0,
            "gross_pnl": 0.0,
            "cost": 0.0,
            "net_pnl": 0.0,
            "ret_on_capital": None,
        }

        if rows:
            table = pd.DataFrame(rows).set_index("ticker")
            ranked = rank_trending(table, weights, screening)
            if not ranked.empty:
                pick_tickers = list(ranked.head(config.picks).index)
                all_picks.extend(pick_tickers)
                capital = 0.0
                for ticker in pick_tickers:
                    entry, exit_ = quotes[ticker]
                    shares = int(config.shares)
                    gross = shares * (exit_ - entry)
                    cost = shares * (entry + exit_) * (config.cost_bps / 10_000.0)
                    net = gross - cost
                    capital += shares * entry
                    day["picks"].append(
                        {
                            "ticker": ticker,
                            "entry": _round(entry, 2),
                            "exit": _round(exit_, 2),
                            "shares": shares,
                            "gross_pnl": _round(gross, 2),
                            "cost": _round(cost, 2),
                            "net_pnl": _round(net, 2),
                            "ret": _round(exit_ / entry - 1.0, 6),
                        }
                    )
                    day["gross_pnl"] += gross
                    day["cost"] += cost
                    day["net_pnl"] += net
                day["capital"] = _round(capital, 2)
                day["gross_pnl"] = _round(day["gross_pnl"], 2)
                day["cost"] = _round(day["cost"], 2)
                day["net_pnl"] = _round(day["net_pnl"], 2)
                day["ret_on_capital"] = _round(day["net_pnl"] / capital, 6) if capital else None

        days.append(day)

    # Attach company names for display (offline static map; falls back to ticker).
    names = get_company_names(sorted(set(all_picks)), allow_network=False)
    for day in days:
        for pick in day["picks"]:
            pick["name"] = names.get(pick["ticker"], pick["ticker"])

    summary = _summarize(days)
    progress("Done.", 1.0)
    return {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "as_of": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "config": config.to_dict(),
        "benchmark": benchmark,
        "days": days,
        "summary": summary,
        "universe": settings.get("universe", []),
    }


def _summarize(days: list[dict[str, Any]]) -> dict[str, Any]:
    traded = [d for d in days if d["picks"]]
    if not traded:
        return {"days": 0, "traded_days": 0}

    nets = np.array([d["net_pnl"] for d in traded], dtype=float)
    gross = float(sum(d["gross_pnl"] for d in traded))
    cost = float(sum(d["cost"] for d in traded))
    capitals = np.array([d["capital"] for d in traded], dtype=float)
    rets = np.array([d["ret_on_capital"] or 0.0 for d in traded], dtype=float)
    equity = np.cumprod(1.0 + rets)

    win = int((nets > 0).sum())
    loss = int((nets < 0).sum())
    flat = int((nets == 0).sum())
    decided = win + loss
    best_i = int(np.argmax(nets))
    worst_i = int(np.argmin(nets))

    return {
        "days": len(days),
        "traded_days": len(traded),
        "first": traded[0]["date"],
        "last": traded[-1]["date"],
        "total_gross": _round(gross, 2),
        "total_cost": _round(cost, 2),
        "total_net": _round(float(nets.sum()), 2),
        "win_days": win,
        "loss_days": loss,
        "flat_days": flat,
        "win_rate": _round(win / decided, 4) if decided else None,
        "avg_net_per_day": _round(float(nets.mean()), 2),
        "best_day": {"date": traded[best_i]["date"], "net": _round(float(nets[best_i]), 2)},
        "worst_day": {"date": traded[worst_i]["date"], "net": _round(float(nets[worst_i]), 2)},
        "avg_capital": _round(float(capitals.mean()), 2),
        "total_return_on_capital": _round(float(nets.sum() / capitals.mean()), 4) if capitals.mean() else None,
        "mean_daily_return": _round(float(rets.mean()), 6),
        "cum_net": [ _round(float(x), 2) for x in np.cumsum(nets) ],
        "equity": [_round(float(x), 6) for x in equity],
        "dates": [d["date"] for d in traded],
    }


def run_daily_sim(
    settings: dict[str, Any] | None = None,
    config: DailySimConfig | None = None,
    synthetic: bool = False,
    use_cache: bool = True,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """Fetch prices, then run the same-day simulator."""
    config = config or DailySimConfig()
    settings = settings or load_settings()
    progress = progress or _noop

    benchmark = settings["benchmark"]
    ingest = settings.get("ingest", {}) or {}

    resolution = resolve_universe(settings, allow_network=not synthetic, use_cache=use_cache)
    universe = list(resolution.tickers)
    if not universe:
        raise ValueError("universe is empty")
    errors: list[str] = list(resolution.errors)

    progress("Fetching history…", 0.02)
    prices, ingest_errors = fetch_prices(
        [benchmark] + universe,
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

    frames = {
        t: (df if "sma20" in df.columns else add_indicators(df)) for t, df in prices.items()
    }
    if benchmark not in frames:
        raise RuntimeError(f"benchmark {benchmark} did not load")

    result = simulate_frames(frames, benchmark, settings, config, progress=progress)
    result["universe"] = resolution.to_dict()
    result["errors"] = errors
    result["markdown"] = render_daily_sim_markdown(result)
    return result


# --------------------------------------------------------------------------- #
# Markdown report
# --------------------------------------------------------------------------- #
def _money(v: Any) -> str:
    try:
        return f"${float(v):+,.2f}"
    except (TypeError, ValueError):
        return "n/a"


def render_daily_sim_markdown(result: dict[str, Any]) -> str:
    cfg = result["config"]
    s = result["summary"]
    lines = [
        f"# Same-day round-trip simulator — {result['run_id']}",
        "",
        f"*{result['as_of']} · benchmark {result['benchmark']} · "
        f"{cfg['picks']} picks/day × {cfg['shares']} shares · {cfg['cost_bps']} bps/side · "
        f"lookback {cfg['lookback']}*",
        "",
    ]
    if not s.get("traded_days"):
        lines.append("No tradable days in range.")
        return "\n".join(lines) + "\n"

    verdict = "made money" if (s["total_net"] or 0) > 0 else "lost money"
    lines += [
        "## Verdict",
        "",
        f"Over **{s['traded_days']}** trading days the strategy **{verdict}**: "
        f"net **{_money(s['total_net'])}** "
        f"({s['win_days']} winning days vs {s['loss_days']} losing).",
        "",
        "## Headline",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Net P&L | {_money(s['total_net'])} |",
        f"| Gross P&L | {_money(s['total_gross'])} |",
        f"| Costs | {_money(-s['total_cost']) if s['total_cost'] else '$0.00'} |",
        f"| Winning / losing days | {s['win_days']} / {s['loss_days']} |",
        f"| Win rate | {s['win_rate'] * 100:.1f}% |" if s.get("win_rate") is not None else "| Win rate | n/a |",
        f"| Average net / day | {_money(s['avg_net_per_day'])} |",
        f"| Best day | {_money(s['best_day']['net'])} ({s['best_day']['date']}) |",
        f"| Worst day | {_money(s['worst_day']['net'])} ({s['worst_day']['date']}) |",
        f"| Average capital deployed | {_money(s['avg_capital']).lstrip('+')} |",
        f"| Return on capital | {s['total_return_on_capital'] * 100:+.2f}% |"
        if s.get("total_return_on_capital") is not None
        else "| Return on capital | n/a |",
        "",
        "## Last 20 tradable days",
        "",
        "| Date | Picks | Capital | Gross | Cost | **Net** |",
        "|---|---|---|---|---|---|",
    ]
    for day in [d for d in result["days"] if d["picks"]][-20:]:
        picks = ", ".join(p["ticker"] for p in day["picks"])
        lines.append(
            f"| {day['date']} | {picks} | {_money(day['capital']).lstrip('+')} | "
            f"{_money(day['gross_pnl'])} | {_money(-day['cost']) if day['cost'] else '$0.00'} | "
            f"**{_money(day['net_pnl'])}** |"
        )

    lines += [
        "",
        "---",
        "",
        "**Caveats.** Daily bars only — 'sell before the close' is modelled as the official "
        "close, so fills are optimistic and slippage/spread are not modelled beyond bps. "
        "Fixed 100-share sizing means the capital at risk varies by price. Most importantly, "
        "the project's research found **no demonstrated edge** in this signal, so expect a "
        "losing or random result; this is a research exercise, not a tradable system.",
        "",
    ]
    return "\n".join(lines)
