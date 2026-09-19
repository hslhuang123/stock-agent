"""Same-day round-trip simulator page.

Auto-discovered by Streamlit when running `streamlit run web/app.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.backtest.daily_sim import DailySimConfig, run_daily_sim  # noqa: E402
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

st.set_page_config(page_title="Simulator · Stock Agent", page_icon=":material/schedule:", layout="wide")
page_header(
    ":material/schedule:",
    "Same-day simulator",
    "If we bought 100 shares of the top picks at today's open and sold them at today's close, "
    "would we have made or lost money that day?",
    "A literal cash-P&L simulation, day by day. Signals are computed from the PREVIOUS close only, "
    "so there is no look-ahead. It is an approximation — see the caveats at the bottom.",
)
how_to_read(
    [
        "Each day the app ranks the universe using data up to **yesterday's close**.",
        "It buys **100 shares of the top picks at today's open**, then sells at **today's close**.",
        "**Net P&L** = (exit − entry) × shares, minus commission/slippage charged on both sides.",
        "**Win rate** = share of days that ended up. A high win rate can still lose money if the "
        "losses are bigger than the wins.",
        "The app only has **daily** bars, so 'sell before the close' is modelled as the closing "
        "price — real fills would be worse.",
    ]
)
glossary()

settings = load_settings()

with st.sidebar:
    st.header("Simulator settings")
    run_settings = universe_selector(settings, key_prefix="sim", show_backtest_mode=False)
    st.divider()
    lookback = st.selectbox("History", ["3mo", "6mo", "1y", "3y", "5y"], index=3)
    picks = st.slider("Stocks bought per day", 1, 5, 2)
    shares = st.number_input("Shares per pick", min_value=1, max_value=10_000, value=100, step=100)
    cost_bps = st.slider("Cost per side (bps)", 0.0, 50.0, 10.0, step=2.5)
    synthetic = st.toggle("Offline / synthetic data", value=False)
    run = st.button("Run simulation", icon=":material/play_arrow:", type="primary", width="stretch")

if run:
    bar = st.progress(0.0, text="Fetching history…")

    def on_progress(message: str, p: float) -> None:
        bar.progress(min(max(p, 0.0), 1.0), text=message)

    config = DailySimConfig(
        lookback=lookback,
        picks=int(picks),
        shares=int(shares),
        cost_bps=float(cost_bps),
    )
    try:
        with st.spinner("Simulating day by day…"):
            st.session_state["sim"] = run_daily_sim(
                settings=run_settings,
                config=config,
                synthetic=synthetic,
                use_cache=True,
                progress=on_progress,
            )
        bar.empty()
    except Exception as exc:
        bar.empty()
        st.error(f"Simulation failed: {exc}")

sim = st.session_state.get("sim")
if not sim:
    st.info("Configure the simulation in the sidebar and press **Run simulation**.")
    st.stop()

s = sim["summary"]
_uni_caption = universe_caption(sim)
if _uni_caption:
    st.caption(_uni_caption)
st.caption(
    f"Run `{sim['run_id']}` · {sim['config']['lookback']} lookback · "
    f"{sim['config']['picks']} picks × {sim['config']['shares']} shares · "
    f"{sim['config']['cost_bps']} bps/side"
)

if not s.get("traded_days"):
    st.warning("No tradable days in range — widen the lookback or check the data.")
    st.stop()

total_net = s["total_net"] or 0.0
if total_net > 0:
    answer(
        "good",
        f"This made money over the period: {total_net:+,.2f} USD net.",
        f"{s['win_days']} winning days vs {s['loss_days']} losing "
        f"(win rate {(s.get('win_rate') or 0) * 100:.0f}%).",
    )
elif total_net < 0:
    answer(
        "bad",
        f"This lost money over the period: {total_net:+,.2f} USD net.",
        f"{s['win_days']} winning days vs {s['loss_days']} losing "
        f"(win rate {(s.get('win_rate') or 0) * 100:.0f}%).",
    )
else:
    answer("info", "Break-even over the period.")

with st.container(horizontal=True):
    st.metric("Net P&L", f"${total_net:+,.0f}", border=True, help="Total profit/loss after costs, in USD.")
    st.metric("Winning days", s["win_days"], border=True, help="Days that ended positive.")
    st.metric("Losing days", s["loss_days"], border=True, help="Days that ended negative.")
    st.metric(
        "Win rate",
        f"{(s.get('win_rate') or 0) * 100:.0f}%",
        border=True,
        help="Winning days ÷ (winning + losing). It can be high and still lose money if losses are larger.",
    )
    st.metric(
        "Avg net / day",
        f"${s['avg_net_per_day']:+,.2f}",
        border=True,
        help="Average net P&L per traded day.",
    )

st.subheader("Cumulative P&L", icon=":material/show_chart:")
cum = pd.DataFrame(
    {"Cumulative net P&L ($)": s["cum_net"]},
    index=pd.to_datetime(s["dates"]),
)
st.line_chart(cum, height=320, width="stretch")

st.subheader("Daily P&L", icon=":material/bar_chart:")
traded = [d for d in sim["days"] if d["picks"]]
daily = pd.DataFrame(
    {"Daily net P&L ($)": [d["net_pnl"] for d in traded]},
    index=pd.to_datetime([d["date"] for d in traded]),
)
st.bar_chart(daily, height=260, width="stretch")

st.subheader("Day-by-day results", icon=":material/table_chart:")
rows = []
for d in traded:
    rows.append(
        {
            "Date": d["date"],
            "Picks (entry → exit)": ", ".join(
                f"{p['ticker']} {p['entry']:.2f}→{p['exit']:.2f}" for p in d["picks"]
            ),
            "Capital": d["capital"],
            "Gross": d["gross_pnl"],
            "Cost": -d["cost"],
            "Net": d["net_pnl"],
            "Return": d["ret_on_capital"],
        }
    )
df = pd.DataFrame(rows).iloc[::-1]
st.dataframe(
    df,
    width="stretch",
    hide_index=True,
    column_config={
        "Capital": st.column_config.NumberColumn(format="$%,.0f", help="Total cost of the shares bought that day."),
        "Gross": st.column_config.NumberColumn(format="$%+,.2f", help="Before costs."),
        "Cost": st.column_config.NumberColumn(format="$%+,.2f", help="Commission + slippage on both sides."),
        "Net": st.column_config.NumberColumn(format="$%+,.2f", help="Gross minus costs — the bottom line."),
        "Return": st.column_config.NumberColumn(format="%+.3f%%", help="Net P&L as a fraction of capital deployed."),
    },
)

with st.expander("What this simulation does and does not tell you", icon=":material/info:"):
    st.markdown(
        "- **No look-ahead.** The pick for each day uses only data through the previous close.\n"
        "- **Daily bars only.** 'Sell before the close' is modelled as the official closing price, "
        "and there is no intraday path, spread or slippage beyond the bps you set — so the numbers "
        "are **optimistic**.\n"
        "- **Fixed 100 shares.** Capital deployed varies with price; this is not equal-risk sizing.\n"
        "- **No proven edge.** The project's research found the signal does not beat the universe "
        "after costs. Expect this to lose or be random; it is a research exercise, not a system."
    )

st.download_button(
    "Download simulation report",
    data=sim["markdown"],
    file_name=f"daily-sim-{sim['run_id']}.md",
    mime="text/markdown",
    icon=":material/download:",
    width="stretch",
)

research_caveat()
