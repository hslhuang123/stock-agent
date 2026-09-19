"""Streamlit attribution page.

Auto-discovered by Streamlit when running `streamlit run web/app.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.backtest.attribution import run_attribution_suite  # noqa: E402
from core.backtest.engine import BacktestConfig  # noqa: E402
from core.config import load_settings  # noqa: E402
from web.ui import (  # noqa: E402
    answer,
    explain,
    glossary,
    how_to_read,
    page_header,
    research_caveat,
)
from web.universe_controls import universe_caption, universe_selector  # noqa: E402

st.set_page_config(page_title="Attribution · Stock Agent", page_icon=":material/science:", layout="wide")
page_header(
    ":material/science:",
    "Factor attribution",
    "When the strategy made money, where did it come from — picking good stocks, or just the "
    "overall market and one hot industry going up?",
    "This separates real stock-picking skill from lucky exposure. A strategy that only wins because "
    "it holds a lot of technology stocks is not really picking stocks. We also check whether trading "
    "less often (to save on fees) changes the answer.",
)
how_to_read(
    [
        "**Jensen's alpha** strips out market beta. Significant positive alpha means the strategy "
        "is not just a repackaged index fund.",
        "**Sector-neutral excess** compares against a benchmark holding the *same sectors* but "
        "picking names at random. Beating it requires stock selection, not sector tilting.",
        "**Turnover variants** test hysteresis bands, which trade less. The cost columns show "
        "whether each variant survives realistic fees.",
        "Read the **t-statistics**, not just the point estimates. |t| below 2 means \"cannot rule "
        "out chance\".",
    ]
)
glossary()


def pct(v, digits=1):
    try:
        return f"{float(v) * 100:+.{digits}f}%"
    except (TypeError, ValueError):
        return "n/a"


def num(v, digits=2):
    try:
        return f"{float(v):,.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


settings = load_settings()

with st.sidebar:
    st.header("Attribution settings")
    run_settings = universe_selector(settings, key_prefix="at", show_backtest_mode=False)
    st.divider()
    lookback = st.selectbox("History", ["3y", "5y", "10y"], index=2)
    top_n = st.slider("Positions", 3, 30, 10)
    cost = st.slider("Baseline cost (bps/side)", 0.0, 50.0, 10.0, step=2.5)
    synthetic = st.toggle("Offline / synthetic data", value=False)
    run = st.button("Run attribution", icon=":material/play_arrow:", type="primary", width="stretch")

if run:
    bar = st.progress(0.0, text="Fetching history…")

    def on_progress(message: str, p: float) -> None:
        bar.progress(min(max(p, 0.0), 1.0), text=message)

    try:
        with st.spinner("Running attribution…"):
            st.session_state["at"] = run_attribution_suite(
                settings=run_settings,
                config=BacktestConfig(lookback=lookback, top_n=int(top_n), cost_bps=float(cost)),
                synthetic=synthetic,
                progress=on_progress,
            )
        bar.empty()
    except Exception as exc:
        bar.empty()
        st.error(f"Attribution failed: {exc}")

at = st.session_state.get("at")
if not at:
    st.info("Configure in the sidebar and press **Run attribution**.")
    st.stop()

verdict = at["verdict"]
_uni_caption = universe_caption(at)
if _uni_caption:
    st.caption(_uni_caption)

_level, _headline, _detail = {
    "PASS": (
        "good",
        "All three attribution checks passed.",
        "There is evidence of return beyond market beta, sector tilting and costs.",
    ),
    "PARTIAL": (
        "warn",
        f"Only {verdict['n_passed']} of 3 attribution checks passed.",
        "Part of the apparent edge is explained away by market beta, sector exposure or "
        "costs — read the checks below.",
    ),
    "FAIL": (
        "bad",
        "The attribution checks failed.",
        "The apparent edge is explained by market beta, sector exposure or costs. It is not skill.",
    ),
}.get(verdict["overall"], ("info", f"Verdict: {verdict['overall']}.", ""))
answer(_level, _headline, _detail)

cols = st.columns(3)
for col, check in zip(cols, verdict["checks"]):
    with col:
        col.metric(check["name"], "PASS" if check["passed"] else "FAIL", border=True)
        col.caption(check["detail"])

st.divider()

# --- Market attribution ---------------------------------------------------
market = at["market"]
st.subheader("Market attribution — Jensen's alpha", icon=":material/functions:")
st.markdown(
    "Regressing strategy returns on the benchmark with Newey-West (HAC) errors. "
    "Alpha is what remains after paying for market exposure."
)
m1, m2, m3, m4 = st.columns(4)
m1.metric(
    "Alpha (annualised)", pct(market.get("alpha_annualized")), f"t={num(market.get('alpha_t'))}",
    border=True, help=explain("Jensen's alpha"),
)
m2.metric(
    "Beta", num(market.get("beta")), f"t={num(market.get('beta_t'))}",
    border=True, help=explain("Beta"),
)
m3.metric(
    "R²", num(market.get("r_squared")), border=True,
    help="Share of the strategy's return explained by the market. High R² means it mostly just moves with the market.",
)
m4.metric(
    "Observations", market.get("n", 0), border=True,
    help="Number of return periods in the regression.",
)

if market.get("significant_alpha"):
    st.success("Alpha is positive and statistically significant (t ≥ 2). Not just a repackaged index.")
else:
    st.warning("Alpha is not statistically significant once market beta is accounted for.")

# --- Sector neutral -------------------------------------------------------
sec = at["sector_neutral"]
st.subheader("Sector-neutral benchmark", icon=":material/groups:")
st.markdown(
    "A benchmark holding the **same sector weights** as the strategy but picking names "
    "at random within each sector. Beating it requires stock selection, not allocation."
)
s1, s2, s3 = st.columns(3)
s1.metric(
    "Excess (annualised)", pct(sec.get("mean_annualized")), f"t={num(sec.get('t_stat'))}",
    border=True, help=explain("Sector-neutral excess"),
)
s2.metric(
    "Periods positive", pct(sec.get("positive_share"), 0), border=True,
    help="Share of months the strategy beat the sector-matched benchmark.",
)
s3.metric("Observations", sec.get("n", 0), border=True, help="Number of monthly return periods.")

if sec.get("significant"):
    st.success("Stock selection adds value beyond sector allocation.")
else:
    st.error(
        "Once sector allocation is matched, the excess is **not** statistically significant. "
        "Much of the apparent edge is sector tilting."
    )

weights = at.get("sector_weights") or {}
if weights:
    st.markdown("**Average sector exposure**")
    wdf = pd.DataFrame(
        {"weight": [w for w in weights.values()]},
        index=list(weights.keys()),
    )
    st.bar_chart(wdf, horizontal=True, height=min(320, 40 * len(weights) + 80))

# --- Turnover variants ----------------------------------------------------
st.subheader("Turnover reduction and cost survival", icon=":material/repeat:")
st.markdown(
    "`hysteresis` keeps holdings until rank falls below `top_n × exit_multiple`, cutting turnover. "
    "The **Sector-neutral** column is the decisive one — it shows whether each variant has real "
    "stock-selection edge."
)
variants = at.get("turnover_variants") or []
if variants:
    rows = []
    for v in variants:
        ego = v.get("edge_by_cost") or {}
        row = {
            "Rebalance": f"{v['rebalance_days']}d",
            "Scoring": "sector-neutral" if v.get("scoring") == "sector_neutral" else "pooled",
            "Selection": "top-N" if v["selection"] == "topn" else "hysteresis",
            "Exit×": "—" if v["selection"] == "topn" else num(v.get("exit_multiple"), 1),
            "Turnover": pct(v.get("avg_turnover"), 0),
            "Alpha (ann.)": pct(v.get("alpha_annualized")),
            "Beta": num(v.get("beta")),
            "Sector-neutral": pct(v.get("sector_neutral_annualized")),
            "Sector-neutral t": num(v.get("sector_neutral_t")),
            "Sharpe": num(v.get("sharpe")),
        }
        for k in sorted(ego, key=int):
            row[f"Edge @ {k}bps"] = pct(ego[k])
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    comp = verdict.get("scoring_comparison") or {}
    if comp.get("pooled") and comp.get("sector_neutral"):
        st.markdown("**Pooled vs sector-neutral scoring**")
        cdf = pd.DataFrame(
            {
                "Variants": [comp["pooled"].get("variants"), comp["sector_neutral"].get("variants")],
                "Mean turnover": [pct(comp["pooled"].get("mean_turnover"), 0), pct(comp["sector_neutral"].get("mean_turnover"), 0)],
                "Mean sector-neutral alpha": [
                    pct(comp["pooled"].get("mean_sector_neutral_annualized")),
                    pct(comp["sector_neutral"].get("mean_sector_neutral_annualized")),
                ],
                "Best t": [num(comp["pooled"].get("best_sector_neutral_t")), num(comp["sector_neutral"].get("best_sector_neutral_t"))],
                "Significant variants": [
                    comp["pooled"].get("significant_variants"),
                    comp["sector_neutral"].get("significant_variants"),
                ],
            },
            index=["Pooled", "Sector-neutral"],
        )
        st.dataframe(cdf, width="stretch")

    sig = [v for v in variants if v.get("sector_neutral_significant")]
    if not sig:
        st.error(
            "**No variant retains a statistically significant sector-neutral excess** — including the "
            "sector-neutral-scored ones. The signal has no measurable within-sector content; the edge "
            "is sector rotation, not stock picking."
        )
    else:
        best = max(sig, key=lambda v: v.get("sector_neutral_annualized") or -9)
        st.success(
            f"Variant {best['rebalance_days']}d/{best.get('scoring')}/{best['selection']} keeps a "
            f"significant sector-neutral excess of {pct(best.get('sector_neutral_annualized'))} "
            f"(t={num(best.get('sector_neutral_t'))})."
        )

st.divider()
st.info(
    "**Caveat.** Sectors are static GICS labels (not point-in-time), and the universe is still "
    "hindsight-selected. Attribution narrows *what* the edge is; it does not make the sample unbiased.",
    icon=":material/info:",
)

st.download_button(
    "Download attribution report",
    data=at["markdown"],
    file_name=f"attribution-{at['run_id']}.md",
    mime="text/markdown",
    icon=":material/download:",
    width="stretch",
)

research_caveat()
