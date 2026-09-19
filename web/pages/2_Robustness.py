"""Streamlit robustness page.

Auto-discovered by Streamlit when running `streamlit run web/app.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.backtest.engine import BacktestConfig  # noqa: E402
from core.backtest.robustness import run_robustness_suite  # noqa: E402
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

st.set_page_config(page_title="Robustness · Stock Agent", page_icon=":material/shield:", layout="wide")
page_header(
    ":material/shield:",
    "Robustness gate",
    "Is the past result real, or did we just get lucky?",
    "Three luck checks: does it still work if we remove some stocks, if we change the settings, and "
    "if we add realistic trading fees? Every check compares against simply owning all the stocks "
    "equally, not against the S&P 500.",
)
how_to_read(
    [
        "**Test 1 — bootstrap:** re-runs the strategy on many random subsets of the universe. "
        "A real signal keeps working when names are removed.",
        "**Test 2 — parameter plateau:** re-runs across many position/rebalance settings. "
        "A real signal degrades gracefully; a curve-fit spikes at one setting.",
        "**Test 3 — cost sensitivity:** momentum trades a lot. An edge that dies at 30 bps "
        "per side is not tradable in practice.",
        "The overall verdict is a *robustness* grade, not a profit forecast. Even a PASS does "
        "not make the absolute returns achievable.",
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
    st.header("Robustness settings")
    run_settings = universe_selector(settings, key_prefix="rb", show_backtest_mode=True)
    st.divider()
    lookback = st.selectbox("History", ["3y", "5y", "10y"], index=2)
    top_n = st.slider("Positions", 3, 30, 10)
    cost_bps = st.slider("Cost per side (bps)", 0.0, 50.0, 10.0, step=2.5)
    trials = st.slider("Bootstrap trials", 10, 100, 30, step=10)
    drop = st.slider("Fraction of universe dropped", 0.1, 0.5, 0.30, step=0.05)
    synthetic = st.toggle("Offline / synthetic data", value=False)
    run = st.button("Run robustness gate", icon=":material/play_arrow:", type="primary", width="stretch")

if run:
    bar = st.progress(0.0, text="Fetching history…")

    def on_progress(message: str, p: float) -> None:
        bar.progress(min(max(p, 0.0), 1.0), text=message)

    try:
        with st.spinner("Running tests…"):
            st.session_state["rb"] = run_robustness_suite(
                settings=run_settings,
                config=BacktestConfig(lookback=lookback, top_n=int(top_n), cost_bps=float(cost_bps)),
                synthetic=synthetic,
                bootstrap_trials=int(trials),
                drop_frac=float(drop),
                progress=on_progress,
            )
        bar.empty()
    except Exception as exc:
        bar.empty()
        st.error(f"Robustness suite failed: {exc}")

rb = st.session_state.get("rb")
if not rb:
    st.info("Configure the suite in the sidebar and press **Run robustness gate**.")
    st.stop()

verdict = rb["verdict"]
overall = verdict["overall"]
_uni_caption = universe_caption(rb)
if _uni_caption:
    st.caption(_uni_caption)

_level, _headline, _detail = {
    "PASS": (
        "good",
        "All three robustness checks passed.",
        "The edge is broad, stable across settings and survives the tested costs. "
        "That is still not proof of profit — see the caveat at the bottom.",
    ),
    "PARTIAL": (
        "warn",
        f"Only {verdict['n_passed']} of 3 robustness checks passed.",
        "The edge is fragile: read the failing check below before trusting it.",
    ),
    "FAIL": (
        "bad",
        "The robustness checks failed.",
        "The edge does not survive subsetting, settings or costs — do not treat it as tradable.",
    ),
}.get(overall, ("info", f"Verdict: {overall}.", ""))
answer(_level, _headline, _detail)

cols = st.columns(3)
for col, check in zip(cols, verdict["checks"]):
    with col:
        col.metric(check["name"], "PASS" if check["passed"] else "FAIL", border=True)
        col.caption(check["detail"])

st.divider()

st.subheader("The baseline run", icon=":material/straighten:")
base = rb["baseline"]
with st.container(horizontal=True):
    st.metric(
        "Edge vs universe", pct(base.get("edge_vs_universe")),
        border=True, help=explain("Edge vs universe"),
    )
    st.metric(
        "Universe equal-weight", pct(base.get("universe_total")),
        border=True, help="Total return of holding every eligible name equally. The bar the strategy must clear.",
    )
    st.metric("Sharpe", num(base.get("sharpe")), border=True, help=explain("Sharpe"))
    st.metric("IC mean", num(base.get("ic_mean"), 4), border=True, help=explain("IC (information coefficient)"))

st.subheader("Test 1 — Universe bootstrap", icon=":material/casino:")
bs = rb["bootstrap"]
st.markdown(
    f"Ran the strategy on **{bs['n_trials']}** random subsets of **{bs['keep']}** tickers "
    f"(dropping {bs['drop_frac']:.0%} each time). A real signal keeps working when you remove names."
)

m1, m2, m3, m4 = st.columns(4)
m1.metric(
    "Positive-edge share", pct(bs.get("edge_positive_share"), 0), border=True,
    help="Share of random subsets where the strategy still beat the universe. Higher is better.",
)
m2.metric("Median edge", pct(bs.get("edge_median")), border=True, help="Typical edge across subsets.")
m3.metric("5th percentile", pct(bs.get("edge_p5")), border=True, help="Bad-case subset edge — how much worse it can get.")
m4.metric("Worst case", pct(bs.get("edge_min")), border=True, help="The single worst subset.")

edges = [t["edge_vs_universe"] for t in bs.get("trials", [])]
if edges:
    hist = pd.DataFrame({"edge": edges})
    st.bar_chart(hist, height=200)
    st.caption("Distribution of edge vs equal-weight universe across random subsets.")

share = bs.get("edge_positive_share") or 0
if share >= 0.9:
    st.success("The edge is broad-based — almost any subset reproduces it.")
elif share >= 0.8:
    st.info("The edge survives subsetting, with meaningful variance.")
elif share >= 0.5:
    st.warning("The edge is unstable — roughly a coin flip on which names you include.")
else:
    st.error("The edge is concentrated in a minority of names. Most subsets lose to the universe.")

# --- Test 2 ---------------------------------------------------------------
st.subheader("Test 2 — Parameter plateau", icon=":material/tune:")
sweep = rb["sweep"]
st.markdown(
    f"Swept **{sweep['n_cells']}** combinations of positions and rebalance frequency. "
    "Real signals degrade gracefully; curve-fits spike at one setting."
)

cells = sweep.get("cells", [])
if cells:
    top_ns = sorted({c["top_n"] for c in cells})
    rebs = sorted({c["rebalance_days"] for c in cells})
    grid = pd.DataFrame(index=[f"{r}d" for r in rebs], columns=[str(n) for n in top_ns], dtype=float)
    for c in cells:
        grid.loc[f"{c['rebalance_days']}d", str(c["top_n"])] = (c["edge_vs_universe"] or 0) * 100
    with st.expander("Edge vs universe (%) by configuration", icon=":material/grid_on:"):
        st.caption("Rows = rebalance frequency, columns = number of positions. A real signal looks smooth across the grid.")
        st.dataframe(grid, width="stretch")

s1, s2, s3 = st.columns(3)
s1.metric(
    "Configs with positive edge", pct(sweep.get("edge_positive_share"), 0), border=True,
    help="Share of the tested settings where the strategy still beat the universe.",
)
s2.metric(
    "Edge spread", pct(sweep.get("edge_spread")), border=True,
    help="Best minus worst configuration. A very large spread signals curve-fit risk.",
)
s3.metric("Median edge", pct(sweep.get("edge_median")), border=True, help="Typical edge across settings.")

if (sweep.get("edge_spread") or 0) > 3.0:
    st.warning("Large spread across settings suggests sensitivity to arbitrary choices — curve-fit risk.")
else:
    st.success("The result is reasonably stable across parameter choices.")

# --- Test 3 ---------------------------------------------------------------
st.subheader("Test 3 — Cost sensitivity", icon=":material/payments:")
st.markdown("Momentum is turnover-heavy. An edge that dies at 30 bps is not tradable at size.")
costs = rb["costs"]
cost_df = pd.DataFrame(costs.get("rows", []))
if not cost_df.empty:
    show = cost_df[["cost_bps", "net_total", "universe_total", "edge_vs_universe", "sharpe"]].copy()
    for col in ["net_total", "universe_total", "edge_vs_universe"]:
        show[col] = show[col].map(lambda v: pct(v))
    show.columns = ["Cost (bps/side)", "Net total", "Universe EW", "Edge", "Sharpe"]
    st.dataframe(show, width="stretch", hide_index=True)

    chart = pd.DataFrame(
        {"Edge": [r["edge_vs_universe"] for r in costs["rows"]]},
        index=[f"{r['cost_bps']:.0f} bps" for r in costs["rows"]],
    )
    st.bar_chart(chart, height=200)

if costs.get("edge_positive_at_max_cost"):
    st.success(f"Edge survives {costs['max_cost_bps']:.0f} bps per side.")
else:
    st.error(f"Edge does not survive {costs['max_cost_bps']:.0f} bps per side.")

st.divider()
st.info(
    "**What this does not fix.** The universe is still selected with hindsight. A PASS means the "
    "signal is robust to *composition* — not that the absolute returns are achievable. "
    "Point-in-time index membership remains the next required step.",
    icon=":material/info:",
)

st.download_button(
    "Download robustness report",
    data=rb["markdown"],
    file_name=f"robustness-{rb['run_id']}.md",
    mime="text/markdown",
    icon=":material/download:",
    width="stretch",
)

research_caveat()
