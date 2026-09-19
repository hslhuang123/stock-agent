"""Markdown rendering for the attribution suite."""
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


def render_attribution_markdown(result: dict[str, Any]) -> str:
    cfg = result.get("config", {})
    base = result.get("baseline", {})
    market = result.get("market", {})
    uni_market = result.get("universe_market", {})
    sector_attr = result.get("sector_neutral", {})
    sector_weights = result.get("sector_weights", {})
    variants = result.get("turnover_variants", [])
    verdict = result.get("verdict", {})

    L: list[str] = []
    L.append(f"# Factor Attribution — {result.get('run_id', '')}")
    L.append("")
    L.append(
        f"*{result.get('universe_size')} tickers · benchmark {result.get('benchmark')} · "
        f"{cfg.get('lookback')} history · top {cfg.get('top_n')} · "
        f"{cfg.get('cost_bps')} bps/side · Newey-West HAC errors*"
    )
    L.append("")
    L.append("The question: **is this skill, or just beta and sector exposure?**")
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

    # --- Baseline ---------------------------------------------------------
    L.append("## Baseline")
    L.append("")
    L.append("| Metric | Value |")
    L.append("|---|---|")
    L.append(f"| Net total | {_pct(base.get('net_total'))} |")
    L.append(f"| Universe equal-weight | {_pct(base.get('universe_total'))} |")
    L.append(f"| Benchmark | {_pct(base.get('benchmark_total'))} |")
    L.append(f"| Edge vs universe | {_pct(base.get('edge_vs_universe'))} |")
    L.append(f"| Sharpe | {_num(base.get('sharpe'))} |")
    L.append("")

    # --- Market attribution ----------------------------------------------
    L.append("## Market attribution (Jensen's alpha)")
    L.append("")
    L.append("Regressing strategy returns on the benchmark, with Newey-West standard errors. "
             "Alpha is what is left after paying for market exposure.")
    L.append("")
    L.append("| Estimate | Value | t-stat |")
    L.append("|---|---|---|")
    L.append(f"| Alpha (annualized) | **{_pct(market.get('alpha_annualized'))}** | {_num(market.get('alpha_t'))} |")
    L.append(f"| Beta | {_num(market.get('beta'))} | {_num(market.get('beta_t'))} |")
    L.append(f"| R² | {_num(market.get('r_squared'))} | — |")
    L.append(f"| Observations | {market.get('n')} (HAC lags {market.get('hac_lags')}) | — |")
    L.append("")
    if market.get("significant_alpha"):
        L.append("> Alpha is positive and statistically significant (t ≥ 2). The strategy is "
                 "not merely a repackaged index position.")
    else:
        L.append("> ⚠️ Alpha is **not** statistically significant once market beta is accounted for. "
                 "The strategy may largely be a levered or concentrated market position.")
    L.append("")
    L.append(
        f"For reference, the *universe itself* has beta {_num(uni_market.get('beta'))} and "
        f"annualized alpha {_pct(uni_market.get('alpha_annualized'))} "
        f"(t={_num(uni_market.get('alpha_t'))}) vs the benchmark. High R² there confirms most of "
        "the universe's return is market exposure."
    )
    L.append("")

    # --- Sector neutral ---------------------------------------------------
    L.append("## Sector-neutral benchmark")
    L.append("")
    L.append("A benchmark that holds the **same sector weights** as the strategy but picks names "
             "at random within each sector. Beating it requires stock selection, not allocation.")
    L.append("")
    L.append("| Metric | Value |")
    L.append("|---|---|")
    L.append(f"| Mean excess (annualized) | **{_pct(sector_attr.get('mean_annualized'))}** |")
    L.append(f"| t-stat | {_num(sector_attr.get('t_stat'))} |")
    L.append(f"| Share of periods positive | {_pct(sector_attr.get('positive_share'))} |")
    L.append(f"| Observations | {sector_attr.get('n')} |")
    L.append("")
    if sector_attr.get("significant"):
        L.append("> Stock selection adds value beyond sector allocation.")
    else:
        L.append("> ⚠️ Once sector allocation is matched, the excess is not statistically "
                 "significant. Much of the edge may be **sector tilting**.")
    L.append("")

    if sector_weights:
        stats = (result.get("rotation") or {}).get("weight_stats") or {}
        L.append("### Sector exposure over time")
        L.append("")
        if stats:
            L.append("| Sector | Mean | Std | Min | Max |")
            L.append("|---|---|---|---|---|")
            for sector, s in list(stats.items())[:10]:
                L.append(
                    f"| {sector} | {_pct(s.get('mean'), 1)} | {_pct(s.get('std'), 1)} | "
                    f"{_pct(s.get('min'), 1)} | {_pct(s.get('max'), 1)} |"
                )
            L.append("")
        else:
            L.append("| Sector | Average weight |")
            L.append("|---|---|")
            for sector, w in list(sector_weights.items())[:10]:
                L.append(f"| {sector} | {_pct(w, 1)} |")
            L.append("")

        rotation = result.get("rotation") or {}
        if rotation:
            L.append("### Rotation or static tilt?")
            L.append("")
            L.append(
                "Correlating each period's sector weight with that same period's sector return. "
                "Positive and meaningful means the tilts are *predictive* — real rotation skill. "
                "Near zero means the weights move but carry no forecasting content."
            )
            L.append("")
            corr = rotation.get("corr_weight_vs_sector_return")
            L.append(f"- Correlation (weight vs same-period sector return): **{_num(corr, 3)}** (n={rotation.get('n_pairs')})")
            L.append(f"- Rotation turbulence (mean weight std): {_pct(rotation.get('rotation_turbulence'), 1)}")
            L.append(f"- Dynamic? {'yes' if rotation.get('is_dynamic') else 'no'} · "
                     f"Predictive? {'yes' if rotation.get('is_predictive') else 'no'}")
            L.append("")
            if rotation.get("is_dynamic") and not rotation.get("is_predictive"):
                L.append(
                    "> ❌ The sector weights vary substantially but have **no predictive power**. "
                    "This is a *static-ish regime tilt* — exposure to whatever sector the sample "
                    "favoured — not sector rotation skill. It is not repeatable."
                )
            elif rotation.get("is_predictive"):
                L.append("> ✅ Sector tilts carry forecasting content — genuine rotation skill.")
            else:
                L.append("> The book does not rotate meaningfully; treat the sector result as a tilt.")
            L.append("")

    # --- Turnover ---------------------------------------------------------
    if variants:
        L.append("## Turnover reduction and cost survival")
        L.append("")
        L.append("`hysteresis` keeps holdings until their rank falls below a threshold "
                 "(`top_n × exit_multiple`), cutting turnover. Edge is measured vs the "
                 "equal-weight universe at each cost level.")
        L.append("")

        costs = sorted({int(k) for v in variants for k in (v.get("edge_by_cost") or {})})
        header = "| Rebalance | Scoring | Selection | Exit× | Turnover | " + " | ".join(
            f"Edge @ {c}bps" for c in costs
        ) + " | Alpha (ann.) | Beta | Sector-neutral | Sharpe |"
        L.append(header)
        L.append("|---" * (8 + len(costs)) + "|")
        for v in variants:
            edge_cells = []
            for c in costs:
                edge_cells.append(_pct((v.get("edge_by_cost") or {}).get(str(c))))
            sel = v.get("selection")
            label = "top-N" if sel == "topn" else "hysteresis"
            ex = "—" if sel == "topn" else _num(v.get("exit_multiple"), 1)
            scoring = v.get("scoring", "pooled")
            scorable = "sector-neutral" if scoring == "sector_neutral" else "pooled"
            sec_cell = _pct(v.get("sector_neutral_annualized"))
            if v.get("sector_neutral_t") is not None:
                sec_cell += f" (t={_num(v.get('sector_neutral_t'))})"
            L.append(
                f"| {v.get('rebalance_days')}d | {scorable} | {label} | {ex} | {_pct(v.get('avg_turnover'), 0)} | "
                + " | ".join(edge_cells)
                + f" | {_pct(v.get('alpha_annualized'))} | {_num(v.get('beta'))} | {sec_cell} | {_num(v.get('sharpe'))} |"
            )
        L.append("")

        comparison = (verdict.get("scoring_comparison") or {})
        pooled = comparison.get("pooled")
        neutral = comparison.get("sector_neutral")
        if pooled and neutral:
            L.append("### Pooled vs sector-neutral scoring")
            L.append("")
            L.append("| Approach | Variants | Mean turnover | Mean sector-neutral alpha | Best t | Significant |")
            L.append("|---|---|---|---|---|---|")
            L.append(
                f"| Pooled | {pooled.get('variants')} | {_pct(pooled.get('mean_turnover'), 0)} | "
                f"{_pct(pooled.get('mean_sector_neutral_annualized'))} | {_num(pooled.get('best_sector_neutral_t'))} | "
                f"{pooled.get('significant_variants')} |"
            )
            L.append(
                f"| Sector-neutral | {neutral.get('variants')} | {_pct(neutral.get('mean_turnover'), 0)} | "
                f"{_pct(neutral.get('mean_sector_neutral_annualized'))} | {_num(neutral.get('best_sector_neutral_t'))} | "
                f"{neutral.get('significant_variants')} |"
            )
            L.append("")
            if (neutral.get("significant_variants") or 0) == 0:
                L.append(
                    "> ❌ Forcing sector neutrality does **not** produce a significant stock-selection "
                    "edge. The signal has no measurable within-sector content. The pooled result was "
                    "sector rotation."
                )
            else:
                L.append(
                    "> ✅ Sector-neutral scoring produces a significant stock-selection edge. The signal "
                    "has genuine within-sector content."
                )
            L.append("")
        sig_sec = [v for v in variants if v.get("sector_neutral_significant")]
        if variants and not sig_sec:
            L.append("> ⚠️ **No variant retains a statistically significant sector-neutral excess.** "
                     "Every improvement traced back to sector allocation. This is the decisive finding: "
                     "the system is a sector-rotation engine, not a stock picker.")
        elif sig_sec:
            best_sec = max(sig_sec, key=lambda v: v.get("sector_neutral_annualized") or -9)
            L.append(
                f"> ✅ Variant **{best_sec.get('rebalance_days')}d / {best_sec.get('selection')}** retains a "
                f"significant sector-neutral excess of {_pct(best_sec.get('sector_neutral_annualized'))} "
                f"(t={_num(best_sec.get('sector_neutral_t'))}) — genuine stock selection on top of sector tilting."
            )
        L.append("")

        best = None
        max_cost = str(max(costs)) if costs else None
        if max_cost:
            viable = [v for v in variants if (v.get("edge_by_cost") or {}).get(max_cost, -1) > 0]
            if viable:
                best = max(viable, key=lambda v: v["edge_by_cost"][max_cost])
                L.append(
                    f"> Best cost-viable variant at {max_cost} bps: **{best.get('rebalance_days')}d / "
                    f"{best.get('selection')}** with turnover {_pct(best.get('avg_turnover'), 0)} "
                    f"and edge {_pct(best['edge_by_cost'][max_cost])}."
                )
            else:
                L.append(f"> ❌ No variant keeps a positive edge at {max_cost} bps.")
            L.append("")

    if result.get("errors"):
        L.append(f"## Data notes ({len(result['errors'])})")
        L.append("")
        for err in result["errors"][:10]:
            L.append(f"- {err}")
        L.append("")

    L.append("---")
    L.append("")
    L.append("**Caveat.** Sectors are static GICS labels, not point-in-time, and the universe "
             "is still selected with hindsight. Attribution narrows *what* the edge is; it does "
             "not make the sample unbiased.")
    return "\n".join(L)
