"""Help & guide page.

Auto-discovered by Streamlit when running `streamlit run web/app.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from web.ui import glossary, page_header, research_caveat  # noqa: E402

st.set_page_config(page_title="Help · Stock Agent", page_icon=":material/help:", layout="wide")

page_header(
    ":material/help:",
    "Help & guide",
    "How does this app work, what does each page tell me, and what do the numbers mean?",
    "Plain-English explanations. The short version: this is a research tool, not investment advice.",
)

# ---------------------------------------------------------------------------
st.subheader("The big picture", icon=":material/school:")
with st.container(border=True):
    st.markdown(
        "This app ranks a list of stocks by how strong their recent price trend is, then writes a "
        "short plan for the top ones — what to buy, where to cut losses, and how big a position to "
        "take.\n\n"
        "**The catch:** a list that *looks* good is not necessarily a list that makes money. So the "
        "other pages are tests of the same list, and so far those tests have **not** found a "
        "reliable advantage. Treat everything here as research, never as advice."
    )

# ---------------------------------------------------------------------------
st.subheader("What each page does", icon=":material/map:")
st.markdown(
    """
| Page | Question it answers | In one word |
|---|---|---|
| **Dashboard** | Which stocks does the app like today, and what is the plan for each? | The output |
| **Backtest** | Would this list have beaten simply owning everything? | History test |
| **Robustness** | Is the past result real, or did we just get lucky? | Stability test |
| **Attribution** | Is any edge skill, or just the market and one hot industry? | Skill test |
| **Simulator** | Buy 100 shares of the top picks at the open and sell at the close — profit or loss? | Daily P&L test |
"""
)

# ---------------------------------------------------------------------------
st.subheader("How to use it", icon=":material/rocket_launch:")
with st.container(border=True):
    st.markdown(
        "1. In the **sidebar**, leave *Universe source* on `static` to screen your configured list, "
        "or switch to `dynamic` to screen ~250 discovered market-wide names.\n"
        "2. Leave **Offline / synthetic data** ON the first time — it needs no internet and always "
        "works. Turn it off for live prices.\n"
        "3. Press **Run screen** on the Dashboard.\n"
        "4. Read the ranked table, then open **See all N names that were screened** and search for "
        "any ticker you care about.\n"
        "5. Use the **Backtest / Robustness / Attribution** pages to ask whether the ranking actually "
        "works. Use the **Simulator** to see a literal daily profit-and-loss."
    )

# ---------------------------------------------------------------------------
st.subheader("Dashboard columns, in plain English", icon=":material/table_chart:")
st.markdown(
    """
| Column | Meaning |
|---|---|
| **Ticker** | Stock symbol. A `.TO` suffix means the Toronto exchange. |
| **Company** | Company name. |
| **Price** | Latest close, in the stock's own currency (`.TO` names are CAD). |
| **Trend** | 0–100 rank of trend strength **within this list**. 100 = strongest here. |
| **Conv.** | Conviction 0–0.95: how strongly the rules like the setup. Below 0.45 → vetoed. |
| **Dir** | `long` (buy candidate), `watch` (wait), or `avoid`. |
| **3m** | Price return over the last ~3 months. |
| **vs Bench** | 3-month return minus the benchmark's (relative strength). |
| **RSI** | 14-day momentum gauge. Above ~70 is often overbought; above 82 is a veto. |
| **ATR%** | Average daily move as a share of price — how volatile it is. |
| **Stop** | Suggested stop-loss (price − 2 × ATR). |
| **Shares / Value** | Position size and its dollar value from the risk model. |
| **Valid** | ✅ passed the risk checks · ❌ vetoed (see the thesis card for why). |
"""
)
st.caption("Every column header also has a hover tooltip, and the glossary below defines every term.")

# ---------------------------------------------------------------------------
st.subheader("The trend score, briefly", icon=":material/functions:")
with st.container(border=True):
    st.markdown(
        "The **Trend** score is built from seven factors — 3-month momentum, 6-month momentum, "
        "strength versus the benchmark, trend quality (moving-average structure), volume surge, "
        "closeness to the 52-week high, and risk-adjusted momentum. Each is ranked as a percentile "
        "across the list, then weighted and combined into a single 0–100 score.\n\n"
        "Because it is a **percentile rank**, it is relative: a score of 80 means *top of this "
        "list*, not *strong in the whole market*."
    )

# ---------------------------------------------------------------------------
glossary()

# ---------------------------------------------------------------------------
st.subheader("Common questions", icon=":material/quiz:")

with st.expander("Why isn't my stock in the table?"):
    st.markdown(
        "The table shows only the **top shortlist** (a handful of names), not the whole universe. "
        "Open **See all N names that were screened** on the Dashboard and search for your ticker to "
        "confirm whether it was screened and where it ranked."
    )

with st.expander("Is this financial advice?"):
    st.markdown(
        "**No.** It is a research and educational tool. The project's own tests found **no "
        "demonstrated edge** — the ranking did not beat simply owning the universe once costs and "
        "sector exposure were accounted for. Never trade real money based on this."
    )

with st.expander("Where does the data come from, and why does it sometimes fail?"):
    st.markdown(
        "Prices come from **Yahoo Finance** (via the `yfinance` library) and are cached in `data/`. "
        "Yahoo revises prices, omits delisted tickers, and rate-limits — especially from cloud "
        "servers. If a live run fails or hangs, switch on **Offline / synthetic data**, which "
        "generates prices locally and always works."
    )

with st.expander("Why do the numbers change between runs?"):
    st.markdown(
        "Live prices move, so the trend score changes daily. It is also a **relative** rank: adding "
        "or removing a name from the universe re-ranks everyone, so the same stock can score "
        "differently from one configuration to the next."
    )

with st.expander("Why does the same-day simulator often lose money after costs?"):
    st.markdown(
        "It round-trips a position **every day**: buy at the open, sell at the close, paying "
        "commission and slippage on both sides. On a momentum screen that turnover is large, so "
        "costs frequently exceed the gross edge — which is exactly what the project's cost-sensitivity "
        "test showed."
    )

with st.expander("Can it suggest a stock outside my list?"):
    st.markdown(
        "Yes — set **Universe source** to `dynamic` in the sidebar and it will screen ~250 "
        "market-wide names discovered from Yahoo's screener; the top of the ranked list is then your "
        "pick. The API also has `POST /analyze/{ticker}` to analyse any ticker you name yourself."
    )

with st.expander("Does the app remember my portfolio or my results?"):
    st.markdown(
        "No. There is **no portfolio feature**, and run results live only in the current browser "
        "session — they are lost when you refresh, close the tab, or the app restarts. The one thing "
        "that persists is the configured ticker list in `config/settings.yaml`, which is part of the "
        "code."
    )

# ---------------------------------------------------------------------------
st.subheader("Learn more", icon=":material/menu_book:")
st.markdown(
    "- **README.md** — what the project is and how to run it.\n"
    "- **FINDINGS.md** — the full research record and the honest verdict.\n"
    "- **AGENT.md** — the design invariants (no look-ahead, deterministic numbers).\n"
    "- **DEPLOY.md** — how the hosted app is deployed.\n\n"
    "All four live in the repository root."
)

research_caveat()
