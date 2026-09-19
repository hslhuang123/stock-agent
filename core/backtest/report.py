"""Markdown rendering for backtest results."""
from __future__ import annotations

from typing import Any


def _pct(v: Any, digits: int = 1) -> str:
    if v is None:
        return "n/a"
    try:
        return f"{float(v) * 100:+.{digits}f}%"
    except (TypeError, ValueError):
        return "n/a"


def _num(v: Any, digits: int = 2) -> str:
    if v is None:
        return "n/a"
    try:
        return f"{float(v):,.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def render_backtest_markdown(result: dict[str, Any]) -> str:
    cfg = result.get("config", {})
    sched = result.get("schedule", {})
    summary = result.get("summary", {})
    net = summary.get("net", {})
    gross = summary.get("gross", {})
    bench = summary.get("benchmark", {})
    sig = result.get("signal_quality", {})

    L: list[str] = []
    L.append(f"# Walk-Forward Backtest — {result.get('as_of', '')}")
    L.append("")
    L.append(
        f"*Run `{result.get('run_id', '')}` · {sched.get('first')} → {sched.get('last')} · "
        f"{sched.get('rebalances')} rebalances · {cfg.get('rebalance_days')}-day rebalance · "
        f"{cfg.get('holding_days')}-day hold · top {cfg.get('top_n')} · "
        f"cost {cfg.get('cost_bps')} bps one-way · benchmark {result.get('benchmark')}*"
    )
    L.append("")
    L.append("> The strategy is always compared against buy-and-hold on the benchmark. "
             "Positive absolute return with negative alpha is not edge — it is beta.")
    L.append("")

    L.append("## Headline")
    L.append("")
    L.append("| Metric | Strategy (net) | Strategy (gross) | Benchmark |")
    L.append("|---|---|---|---|")
    rows = [
        ("Total return", "total_return", True),
        ("CAGR", "cagr", True),
        ("Annualized vol", "ann_vol", True),
        ("Sharpe", "sharpe", False),
        ("Sortino", "sortino", False),
        ("Max drawdown", "max_drawdown", True),
        ("Hit rate (periods > 0)", "hit_rate", True),
        ("Win rate vs benchmark", "win_rate_vs_bench", True),
        ("Mean excess / period", "mean_excess_per_period", True),
        ("Information ratio", "information_ratio", False),
        ("Best period", "best_period", True),
        ("Worst period", "worst_period", True),
    ]
    for label, key, is_pct in rows:
        fmt = _pct if is_pct else _num
        L.append(f"| {label} | {fmt(net.get(key))} | {fmt(gross.get(key))} | {fmt(bench.get(key))} |")
    L.append("")

    L.append("## Controls — is the ranking adding anything?")
    L.append("")
    L.append("A trend score is only useful if the top-N it selects beats (a) a random "
             "N-name basket and (b) the equal-weight average of the whole eligible universe. "
             "If it does not, the strategy is just beta plus concentration.")
    L.append("")
    L.append("| Metric | Strategy (net) | Random basket | Universe equal-weight | Benchmark |")
    L.append("|---|---|---|---|---|")
    rnd = summary.get("random", {})
    uni = summary.get("universe", {})
    for label, key, is_pct in [
        ("Total return", "total_return", True),
        ("CAGR", "cagr", True),
        ("Annualized vol", "ann_vol", True),
        ("Sharpe", "sharpe", False),
        ("Max drawdown", "max_drawdown", True),
        ("Hit rate", "hit_rate", True),
        ("Win rate vs benchmark", "win_rate_vs_bench", True),
    ]:
        fmt = _pct if is_pct else _num
        L.append(
            f"| {label} | {fmt(net.get(key))} | {fmt(rnd.get(key))} | {fmt(uni.get(key))} | {fmt(bench.get(key))} |"
        )
    L.append("")
    strat_total = net.get("total_return")
    rnd_total = rnd.get("total_return")
    uni_total = uni.get("total_return")
    if strat_total is not None and rnd_total is not None and uni_total is not None:
        edge_random = strat_total - rnd_total
        edge_universe = strat_total - uni_total
        L.append(f"- Strategy minus **random basket**: **{_pct(edge_random)}**")
        L.append(f"- Strategy minus **universe equal-weight**: **{_pct(edge_universe)}**")
        L.append("")
        if edge_universe <= 0:
            L.append("> ⚠️ The top-N selection did **not** beat the equal-weight universe. "
                     "The ranking is not adding value over this sample.")
            L.append("")

    L.append("## Signal quality")
    L.append("")
    L.append("The information coefficient (IC) is the rank correlation between the trend "
             "score and the subsequent holding-period return. Anything persistently above "
             "~0.03 is meaningful; a positive top-minus-bottom spread means the ranking "
             "orders outcomes correctly.")
    L.append("")
    L.append(f"- IC mean: **{_num(sig.get('ic_mean'), 4)}**")
    L.append(f"- IC std: {_num(sig.get('ic_std'), 4)}")
    L.append(f"- IC IR (annualized): {_num(sig.get('ic_ir'), 2)}")
    L.append(f"- Share of periods with positive IC: {_pct(sig.get('ic_positive_share'))}")
    L.append(f"- IC observations: {sig.get('ic_periods', 0)}")
    L.append(f"- Top-minus-bottom quintile spread: **{_pct(sig.get('top_minus_bottom_spread'))}** per period")
    L.append("")

    q = sig.get("quintile_mean_forward_return") or {}
    if q:
        L.append("| Quintile (by trend score) | Mean forward return |")
        L.append("|---|---|")
        L.append(f"| Q5 (highest) | {_pct(q.get('q5'))} |")
        L.append(f"| Q4 | {_pct(q.get('q4'))} |")
        L.append(f"| Q3 | {_pct(q.get('q3'))} |")
        L.append(f"| Q2 | {_pct(q.get('q2'))} |")
        L.append(f"| Q1 (lowest) | {_pct(q.get('q1'))} |")
        L.append("")

    folds = result.get("folds") or []
    if folds:
        L.append("## Walk-forward folds (out-of-sample windows)")
        L.append("")
        L.append("| Fold | Window | Periods | Net return | Benchmark | Alpha | Sharpe | Max DD | Win vs bench |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for f in folds:
            L.append(
                f"| {f.get('fold')} | {f.get('start')} → {f.get('end')} | {f.get('periods')} | "
                f"{_pct(f.get('total_return'))} | {_pct(f.get('benchmark_total_return'))} | "
                f"{_pct(f.get('alpha'))} | {_num(f.get('sharpe'))} | {_pct(f.get('max_drawdown'))} | "
                f"{_pct(f.get('win_rate_vs_bench'))} |"
            )
        L.append("")
        L.append("A strategy that only works in one fold is a curve-fit, not an edge.")
        L.append("")

    L.append("## Recent rebalances")
    L.append("")
    L.append("| As of | Exit | Picks | Gross | Net | Bench | Excess | Turnover | IC |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for p in result.get("periods", [])[-12:]:
        picks = ", ".join(p.get("picks", [])[:6])
        if len(p.get("picks", [])) > 6:
            picks += "…"
        excess = None
        if p.get("net_return") is not None and p.get("benchmark_return") is not None:
            excess = p["net_return"] - p["benchmark_return"]
        L.append(
            f"| {p.get('date')} | {p.get('exit_date')} | {picks or '—'} | "
            f"{_pct(p.get('gross_return'))} | {_pct(p.get('net_return'))} | "
            f"{_pct(p.get('benchmark_return'))} | {_pct(excess)} | "
            f"{_num(p.get('turnover'))} | {_num(p.get('ic'), 3)} |"
        )
    L.append("")

    if result.get("errors"):
        L.append(f"## Data notes ({len(result['errors'])})")
        L.append("")
        for err in result["errors"][:15]:
            L.append(f"- {err}")
        L.append("")

    L.append("---")
    L.append("")
    L.append("**Caveats.** In-sample weights, no borrow costs, no liquidity/impact model "
             "beyond bps, and a fixed universe (which is itself a form of survivorship bias "
             "if chosen with hindsight). Treat these numbers as an upper bound.")
    return "\n".join(L)
