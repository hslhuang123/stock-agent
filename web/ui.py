"""Plain-language UI helpers shared by every page.

The four pages are one narrative, not four unrelated tools:

1. **Dashboard** shows today's screen.
2. **Backtest / Robustness / Attribution** are three increasingly strict attempts
   to falsify it.

These helpers keep the framing, metric tooltips and glossary identical everywhere,
so a reader always knows *what a number means* and *whether it is good or bad*.
No custom CSS — only native Streamlit elements, so it survives upgrades.
"""
from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Glossary — one plain-English definition per term used in the UI.
# ---------------------------------------------------------------------------
GLOSSARY: dict[str, str] = {
    "Trend score": (
        "0–100 percentile rank of a stock versus the rest of the universe on "
        "momentum, trend quality, volume surge and 52-week-high proximity. "
        "100 = strongest relative trend."
    ),
    "Conviction": (
        "0–1 confidence the deterministic rules assign to a thesis. Below the "
        "configured floor (default 0.45) the risk validator vetoes the trade."
    ),
    "Edge vs universe": (
        "Strategy return minus the return of simply equal-weighting every "
        "eligible name. The only fair test of the *ranking* — beating SPY alone "
        "proves nothing because the universe itself beats SPY."
    ),
    "IC (information coefficient)": (
        "Rank correlation between the trend score and the NEXT holding period's "
        "return. Near 0 means the score does not rank outcomes; above ~+0.03 is "
        "meaningful."
    ),
    "Sharpe": (
        "Annualised return per unit of volatility. Higher is better, but compare "
        "it with the random basket and the universe — not just with 1.0."
    ),
    "Max drawdown": (
        "Worst peak-to-trough loss over the period, shown negative. Smaller in "
        "magnitude is better."
    ),
    "Turnover": (
        "Share of the portfolio replaced at each rebalance. Higher turnover means "
        "more trading cost and more tax drag."
    ),
    "Cost bps": (
        "One-way trading cost in basis points (1 bp = 0.01%). A momentum edge "
        "that disappears by 30 bps is not tradable in practice."
    ),
    "Bootstrap": (
        "Re-running the strategy on many random subsets of the universe. A real "
        "signal keeps working when names are removed; a mirage does not."
    ),
    "Parameter plateau": (
        "Re-running across many position-count / rebalance settings. A real signal "
        "degrades gracefully; a curve-fit spikes at one lucky setting."
    ),
    "Jensen's alpha": (
        "Return left over after removing the return explained by market beta. "
        "Significant alpha (|t| ≥ 2) would mean the strategy is not a repackaged "
        "index fund."
    ),
    "Beta": (
        "Sensitivity to the market. 1.0 moves with the market; above 1 amplifies "
        "it; below 1 dampens it."
    ),
    "Newey-West / HAC": (
        "A correction to t-statistics for overlapping return windows, which would "
        "otherwise make results look more significant than they really are."
    ),
    "t-statistic": (
        "How many standard errors an estimate sits from zero. Roughly, |t| ≥ 2 "
        "means 'unlikely to be chance'."
    ),
    "Sector-neutral excess": (
        "Excess return versus a benchmark holding the same sector weights but "
        "picking names at random. Beating it requires stock picking, not sector "
        "tilting."
    ),
    "Hysteresis": (
        "Holding a name until its rank falls below a threshold "
        "(top_n × exit_multiple), which cuts turnover versus always re-picking "
        "the top N."
    ),
    "Point-in-time universe": (
        "Rebuilding the tradeable universe at each historical date using only data "
        "available at that date, which prevents look-ahead bias."
    ),
    "Look-ahead bias": (
        "Accidentally using information that would not have been available at the "
        "time. It inflates backtest results."
    ),
    "Long / watch / avoid": (
        "The thesis direction. *long* = actionable buy candidate, *watch* = wait "
        "for confirmation, *avoid* = do not trade."
    ),
}


def explain(term: str) -> str:
    """Return the plain-English definition for a glossary term ('' if unknown)."""
    return GLOSSARY.get(term, "")


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------
def page_header(icon: str, title: str, question: str, blurb: str) -> None:
    """Consistent page opener: title, the plain-English question, and a caveat line."""
    st.title(title, icon=icon)
    st.markdown(f"**In plain English:** {question}")
    st.caption(blurb)


def big_picture(*, expanded: bool = False) -> None:
    """The premise of the whole app, in everyday language."""
    with st.expander(
        "Start here — what is this app actually testing?",
        icon=":material/school:",
        expanded=expanded,
    ):
        st.markdown(
            "This app ranks US stocks by how strong their recent price trend is, then writes a short "
            "plan for the top ones: what to buy, where to cut losses, and how large a position to take.\n\n"
            "**The catch:** a list that *looks* good is not necessarily a list that makes money. So the "
            "other three pages are tests of the same list:\n\n"
            "1. **Backtest** — would this list have done better than just owning a bit of everything?\n"
            "2. **Robustness** — is that result real, or just luck?\n"
            "3. **Attribution** — if it is real, is it skill, or just the market and one hot industry?\n\n"
            "So far these tests have **not** found a reliable advantage. Treat this as a research tool, "
            "not investment advice."
        )


def how_to_read(points: list[str], *, expanded: bool = False) -> None:
    """A short 'what am I looking at' guide, collapsed by default."""
    with st.expander("How to read this page", icon=":material/help:", expanded=expanded):
        for point in points:
            st.markdown(f"- {point}")


def glossary(*, expanded: bool = False) -> None:
    """Render the full plain-English glossary in the active container."""
    with st.expander(
        "Glossary — every term in plain English",
        icon=":material/menu_book:",
        expanded=expanded,
    ):
        for term, definition in GLOSSARY.items():
            st.markdown(f"**{term}** — {definition}")


def answer(level: str, headline: str, detail: str = "") -> None:
    """One plain-language verdict banner.

    ``level`` maps to a native status element so colour carries the meaning:
    ``good`` (green) · ``warn`` (amber) · ``bad`` (red) · ``info`` (blue).
    """
    icons = {
        "good": ":material/check_circle:",
        "warn": ":material/warning:",
        "bad": ":material/error:",
        "info": ":material/info:",
    }
    banners = {"good": st.success, "warn": st.warning, "bad": st.error, "info": st.info}
    banners.get(level, st.info)(f"**{headline}**" + (f" {detail}" if detail else ""), icon=icons.get(level, ":material/info:"))


def research_caveat(text: str | None = None) -> None:
    """The standing honesty note, identical on every page."""
    st.caption(
        text
        or (
            "Research output only — not financial advice, and no orders are placed. "
            "The strategy has **no demonstrated edge**; the Backtest, Robustness and "
            "Attribution pages are the tests that established that."
        )
    )
