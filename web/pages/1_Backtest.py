"""Streamlit backtest page.

Auto-discovered by Streamlit when running `streamlit run web/app.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.backtest.engine import BacktestConfig, run_backtest  # noqa: E402
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

st.set_page_config(page_title="Backtest · Stock Agent", page_icon=":material/history:", layout="wide")
page_header(
    ":material/history:",
    "Walk-forward backtest",
    "If we had followed this list for the past few years, would we have ended up with more money "
    "than just buying a small amount of every stock we are allowed to pick?",
    "We compare against 'buy a bit of everything' (and against a random pick), not just the S&P 500. "
    "The list of allowed stocks is already a good one, so beating the S&P on its own would not prove "
    "the ranking works. Costs are charged every time the list changes.",
)
how_to_read(
    [
        "**Edge** = strategy return minus the equal-weight universe. Beating SPY alone proves "
        "nothing, because the universe itself beats SPY.",
        "**Controls** compares the strategy with (a) a random N-name basket and (b) the whole "
        "universe held equally. If it cannot beat those, the score adds nothing.",
        "**Signal quality** asks whether a higher trend score really led to a higher forward return.",
        "**Folds** split the history into out-of-sample windows. Edge in only one fold is a curve-fit.",
        "Costs are charged on turnover at every rebalance, so the net line is the honest one.",
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
    st.header("Backtest settings")
    run_settings = universe_selector(settings, key_prefix="bt", show_backtest_mode=True)
    st.divider()
    lookback = st.selectbox("History", ["3y", "5y", "10y", "max"], index=2)
    rebalance = st.select_slider("Rebalance (trading days)", [5, 10, 21, 42, 63], value=21)
    hold = st.select_slider("Holding period (days)", [5, 10, 21, 42, 63], value=rebalance)
    top_n = st.slider("Positions", 3, 30, 10)
    cost_bps = st.slider("Cost per side (bps)", 0.0, 50.0, 10.0, step=2.5)
    folds = st.slider("Walk-forward folds", 2, 6, 4)
    weight_mode = st.segmented_control(
        "Weighting",
        options=["equal", "score"],
        default="equal",
        help="`equal` gives each pick the same weight; `score` weights by trend score.",
    ) or "equal"
    synthetic = st.toggle("Offline / synthetic data", value=False)
    run = st.button("Run backtest", icon=":material/play_arrow:", type="primary", width="stretch")

if run:
    bar = st.progress(0.0, text="Fetching history…")

    def on_progress(message: str, p: float) -> None:
        bar.progress(min(max(p, 0.0), 1.0), text=message)

    config = BacktestConfig(
        lookback=lookback,
        rebalance_days=int(rebalance),
        holding_days=int(hold),
        top_n=int(top_n),
        cost_bps=float(cost_bps),
        folds=int(folds),
        weight_mode=weight_mode,
    )
    try:
        with st.spinner("Simulating…"):
            st.session_state["bt"] = run_backtest(
                settings=run_settings, config=config, synthetic=synthetic, progress=on_progress
            )
        bar.empty()
    except Exception as exc:
        bar.empty()
        st.error(f"Backtest failed: {exc}")

bt = st.session_state.get("bt")
if not bt:
    st.info("Configure the backtest in the sidebar and press **Run backtest**.")
    st.stop()

summary = bt["summary"]
net, rnd, uni, bm = summary["net"], summary["random"], summary["universe"], summary["benchmark"]
sched = bt["schedule"]

st.caption(
    f"Run `{bt['run_id']}` · {sched['first']} → {sched['last']} · "
    f"{sched['rebalances']} rebalances · {bt['config']['rebalance_days']}d rebalance · "
    f"top {bt['config']['top_n']} · {bt['config']['cost_bps']} bps/side"
)
_uni_caption = universe_caption(bt)
if _uni_caption:
    st.caption(_uni_caption)

edge_uni = (net.get("total_return") or 0) - (uni.get("total_return") or 0)
edge_rnd = (net.get("total_return") or 0) - (rnd.get("total_return") or 0)
if edge_uni <= 0:
    answer(
        "bad",
        "The ranking did not beat simply holding the universe.",
        f"Edge {pct(edge_uni)} — the score subtracted value versus the equal-weight universe.",
    )
elif edge_uni < 0.5:
    answer(
        "warn",
        "The ranking beat the universe, but only slightly.",
        f"Edge {pct(edge_uni)} vs the equal-weight universe and {pct(edge_rnd)} vs a random basket — "
        "at this size it is hard to separate from noise.",
    )
else:
    answer(
        "good",
        "The ranking beat simply holding the universe.",
        f"Edge {pct(edge_uni)} vs the equal-weight universe and {pct(edge_rnd)} vs a random basket.",
    )

with st.container(horizontal=True):
    st.metric(
        "Total return (net)", pct(net.get("total_return")), border=True,
        help="Cumulative return after trading costs.",
    )
    st.metric("CAGR", pct(net.get("cagr")), border=True, help="Compound annual growth rate.")
    st.metric("Sharpe", num(net.get("sharpe")), border=True, help=explain("Sharpe"))
    st.metric(
        "Max drawdown", pct(net.get("max_drawdown")), border=True, help=explain("Max drawdown")
    )

st.subheader("Is the ranking adding value?", icon=":material/balance:")
st.markdown(
    "A trend score only matters if top-N beats a **random basket** and the "
    "**equal-weight universe**. Otherwise it is beta plus concentration. "
    "Each column is the same window measured four ways."
)

ctrl = pd.DataFrame(
    {
        "Strategy (net)": [
            pct(net.get("total_return")), pct(net.get("cagr")), pct(net.get("ann_vol")),
            num(net.get("sharpe")), pct(net.get("max_drawdown")), pct(net.get("hit_rate")),
        ],
        "Random basket": [
            pct(rnd.get("total_return")), pct(rnd.get("cagr")), pct(rnd.get("ann_vol")),
            num(rnd.get("sharpe")), pct(rnd.get("max_drawdown")), pct(rnd.get("hit_rate")),
        ],
        "Universe EW": [
            pct(uni.get("total_return")), pct(uni.get("cagr")), pct(uni.get("ann_vol")),
            num(uni.get("sharpe")), pct(uni.get("max_drawdown")), pct(uni.get("hit_rate")),
        ],
        "Benchmark": [
            pct(bm.get("total_return")), pct(bm.get("cagr")), pct(bm.get("ann_vol")),
            num(bm.get("sharpe")), pct(bm.get("max_drawdown")), pct(bm.get("hit_rate")),
        ],
    },
    index=["Total return", "CAGR", "Ann. vol", "Sharpe", "Max DD", "Hit rate"],
)
st.dataframe(ctrl, width="stretch")

st.subheader("Equity curves", icon=":material/show_chart:")
st.caption("Growth of $1 over the backtest window for strategy, controls and benchmark.")
eq = bt["equity_curve"]
curve = pd.DataFrame(
    {
        "Strategy (net)": eq["net"],
        "Random basket": eq["random"],
        "Universe EW": eq["universe"],
        "Benchmark": eq["benchmark"],
    },
    index=pd.to_datetime(eq["dates"]),
)
st.line_chart(curve, height=380, width="stretch")

left, right = st.columns(2)
with left:
    st.subheader("Signal quality", icon=":material/insights:")
    st.caption("Did a higher score actually lead to a higher forward return?")
    sig = bt["signal_quality"]
    st.metric(
        "IC mean", num(sig.get("ic_mean"), 4), border=True,
        help=explain("IC (information coefficient)"),
    )
    st.metric(
        "IC IR (annualised)", num(sig.get("ic_ir"), 2), border=True,
        help="Information ratio of the IC: its average divided by its volatility.",
    )
    st.metric(
        "Top − bottom quintile", pct(sig.get("top_minus_bottom_spread")), border=True,
        help="Highest-scoring fifth minus lowest-scoring fifth forward return. Positive = the ranking orders outcomes correctly.",
    )
    ic_mean = sig.get("ic_mean")
    if ic_mean is not None:
        if ic_mean > 0.03:
            st.success("The score has a meaningful positive relationship with future returns.")
        elif ic_mean > 0:
            st.warning("The relationship is weakly positive — close to noise.")
        else:
            st.error("The score does not rank future returns (IC is ~0 or negative).")
    q = sig.get("quintile_mean_forward_return") or {}
    qdf = pd.DataFrame(
        {"Mean forward return": [q.get(f"q{i}") for i in range(1, 6)]},
        index=["Q1 (lowest score)", "Q2", "Q3", "Q4", "Q5 (highest score)"],
    )
    st.bar_chart(qdf, height=220)

with right:
    st.subheader("Walk-forward folds", icon=":material/grid_view:")
    st.caption("Out-of-sample sub-periods. Edge in only one fold is a curve-fit, not a signal.")
    folds_df = pd.DataFrame(bt["folds"])
    if not folds_df.empty:
        show = folds_df[["fold", "start", "end", "total_return", "benchmark_total_return", "alpha", "sharpe", "max_drawdown"]].copy()
        for col in ["total_return", "benchmark_total_return", "alpha", "max_drawdown"]:
            show[col] = show[col].map(lambda v: pct(v))
        st.dataframe(show, width="stretch", hide_index=True)
        st.caption("Edge concentrated in a single fold is a curve-fit, not a signal.")

st.subheader("Recent rebalances", icon=":material/receipt_long:")
st.caption("What the strategy actually held, and how it did over the following period.")
with st.expander("Show the last 15 rebalances", icon=":material/receipt_long:"):
    periods = pd.DataFrame(bt["periods"]).tail(15).copy()
    if not periods.empty:
        periods["picks"] = periods["picks"].map(lambda x: ", ".join(x[:6]) + ("…" if len(x) > 6 else ""))
        periods["excess"] = periods["net_return"] - periods["benchmark_return"]
        st.dataframe(
            periods[["date", "exit_date", "picks", "gross_return", "net_return", "benchmark_return", "excess", "turnover", "ic"]],
            width="stretch",
            hide_index=True,
        )

st.download_button(
    "Download backtest report",
    data=bt["markdown"],
    file_name=f"backtest-{bt['run_id']}.md",
    mime="text/markdown",
    icon=":material/download:",
    width="stretch",
)

research_caveat()
