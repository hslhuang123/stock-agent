"""Markdown rendering for the robustness suite."""
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


def render_robustness_markdown(result: dict[str, Any]) -> str:
    cfg = result.get("config", {})
    base = result.get("baseline", {})
    bs = result.get("bootstrap", {})
    sweep = result.get("sweep", {})
    costs = result.get("costs", {})
    verdict = result.get("verdict", {})

    L: list[str] = []
    L.append(f"# Robustness Gate — {result.get('run_id', '')}")
    L.append("")
    L.append(
        f"*{result.get('universe_size')} tickers · benchmark {result.get('benchmark')} · "
        f"{cfg.get('lookback')} history · top {cfg.get('top_n')} · "
        f"{cfg.get('cost_bps')} bps/side · bootstrap {bs.get('n_trials')} trials "
        f"(drop {bs.get('drop_frac', 0):.0%})*"
    )
    L.append("")
    L.append("Every test is judged against the **equal-weight universe**, not SPY, "
             "so universe selection is not credited as skill.")
    L.append("")

    overall = verdict.get("overall", "?")
    icon = {"PASS": "✅", "PARTIAL": "⚠️", "FAIL": "❌"}.get(overall, "❔")
    L.append(f"## Verdict: {icon} {overall} ({verdict.get('n_passed', 0)}/3 checks passed)")
    L.append("")
    L.append("| Check | Result | Detail |")
    L.append("|---|---|---|")
    for c in verdict.get("checks", []):
        L.append(f"| {c['name']} | {'PASS' if c['passed'] else 'FAIL'} | {c['detail']} |")
    L.append("")

    L.append("## Baseline")
    L.append("")
    L.append("| Metric | Value |")
    L.append("|---|---|")
    L.append(f"| Net total return | {_pct(base.get('net_total'))} |")
    L.append(f"| Gross total return | {_pct(base.get('gross_total'))} |")
    L.append(f"| Universe equal-weight | {_pct(base.get('universe_total'))} |")
    L.append(f"| Benchmark (SPY) | {_pct(base.get('benchmark_total'))} |")
    L.append(f"| **Edge vs universe** | **{_pct(base.get('edge_vs_universe'))}** |")
    L.append(f"| Sharpe | {_num(base.get('sharpe'))} |")
    L.append(f"| Max drawdown | {_pct(base.get('max_drawdown'))} |")
    L.append(f"| IC mean | {_num(base.get('ic_mean'), 4)} |")
    L.append("")

    # --- Test 1 -----------------------------------------------------------
    L.append("## Test 1 — Universe bootstrap")
    L.append("")
    L.append(
        f"Ran the strategy on {bs.get('n_trials')} random subsets of "
        f"{bs.get('keep')} tickers (dropping {bs.get('drop_frac', 0):.0%} each time). "
        "A real signal keeps working when you remove names; a mirage does not."
    )
    L.append("")
    L.append(f"- Share of trials with **positive edge**: **{_pct(bs.get('edge_positive_share'))}**")
    L.append(f"- Edge mean {_pct(bs.get('edge_mean'))} · median {_pct(bs.get('edge_median'))} · std {_pct(bs.get('edge_std'))}")
    L.append(f"- Edge 5th–95th percentile: {_pct(bs.get('edge_p5'))} to {_pct(bs.get('edge_p95'))}")
    L.append(f"- Edge worst/best: {_pct(bs.get('edge_min'))} / {_pct(bs.get('edge_max'))}")
    L.append(f"- Mean Sharpe across trials: {_num(bs.get('sharpe_mean'))} (std {_num(bs.get('sharpe_std'))})")
    L.append(f"- Mean IC across trials: {_num(bs.get('ic_mean_of_means'), 4)}")
    L.append("")
    share = bs.get("edge_positive_share") or 0.0
    if share >= 0.9:
        L.append("> The edge is broad-based: almost any subset reproduces it. That is the good outcome.")
    elif share >= 0.8:
        L.append("> The edge survives subsetting, though with meaningful variance across subsets.")
    elif share >= 0.5:
        L.append("> ⚠️ The edge is **unstable** — roughly a coin flip on which names you include. "
                 "Treat it as unproven.")
    else:
        L.append("> ❌ The edge is **concentrated in a minority of names**. Most subsets lose to the "
                 "equal-weight universe. This is a mirage, not a signal.")
    L.append("")

    # --- Test 2 -----------------------------------------------------------
    L.append("## Test 2 — Parameter plateau")
    L.append("")
    L.append(f"Swept {sweep.get('n_cells')} combinations of positions and rebalance frequency. "
             "Real signals degrade gracefully; curve-fits spike at one setting.")
    L.append("")
    L.append(f"- Share of combos with positive edge: **{_pct(sweep.get('edge_positive_share'))}**")
    L.append(f"- Edge min {_pct(sweep.get('edge_min'))} · median {_pct(sweep.get('edge_median'))} · max {_pct(sweep.get('edge_max'))}")
    L.append(f"- Edge spread (max − min): {_pct(sweep.get('edge_spread'))}")
    L.append("")
    cells = sweep.get("cells", [])
    if cells:
        top_ns = sorted({c["top_n"] for c in cells})
        rebs = sorted({c["rebalance_days"] for c in cells})
        L.append("Edge vs universe by configuration:")
        L.append("")
        L.append("| Rebalance \\ Positions | " + " | ".join(f"{n}" for n in top_ns) + " |")
        L.append("|---" * (len(top_ns) + 1) + "|")
        for reb in rebs:
            row = []
            for n in top_ns:
                match = next((c for c in cells if c["rebalance_days"] == reb and c["top_n"] == n), None)
                row.append(_pct(match["edge_vs_universe"]) if match else "n/a")
            L.append(f"| {reb}d | " + " | ".join(row) + " |")
        L.append("")
    spread = sweep.get("edge_spread")
    if spread is not None and spread > 3.0:
        L.append("> ⚠️ Large spread across configurations suggests the result is sensitive to "
                 "arbitrary settings — a sign of curve-fitting risk.")
    else:
        L.append("> The result is reasonably stable across parameter choices.")
    L.append("")

    # --- Test 3 -----------------------------------------------------------
    L.append("## Test 3 — Cost sensitivity")
    L.append("")
    L.append("Momentum is turnover-heavy. An edge that dies at 30 bps is not tradable at size.")
    L.append("")
    L.append("| Cost (bps/side) | Net total | Universe EW | Edge | Sharpe |")
    L.append("|---|---|---|---|---|")
    for row in costs.get("rows", []):
        L.append(
            f"| {_num(row.get('cost_bps'), 0)} | {_pct(row.get('net_total'))} | "
            f"{_pct(row.get('universe_total'))} | {_pct(row.get('edge_vs_universe'))} | "
            f"{_num(row.get('sharpe'))} |"
        )
    L.append("")
    if costs.get("edge_positive_at_max_cost"):
        L.append(f"> Edge survives {_num(costs.get('max_cost_bps'), 0)} bps per side.")
    else:
        L.append(f"> ❌ Edge does not survive {_num(costs.get('max_cost_bps'), 0)} bps per side.")
    L.append("")

    if result.get("errors"):
        L.append(f"## Data notes ({len(result['errors'])})")
        L.append("")
        for err in result["errors"][:10]:
            L.append(f"- {err}")
        L.append("")

    L.append("---")
    L.append("")
    L.append("**What this does not fix.** The universe is still selected with hindsight. "
             "A PASS here means the signal is robust to *composition*; it does not mean the "
             "absolute returns are achievable. Point-in-time index membership remains the "
             "next required step.")
    return "\n".join(L)
