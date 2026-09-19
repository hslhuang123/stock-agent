"""Streamlit dashboard.

Run from the project root:
    streamlit run web/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import load_settings  # noqa: E402
from core.pipeline import run_pipeline  # noqa: E402
from core.universe import discover_universe  # noqa: E402
from web.ui import (  # noqa: E402
    answer,
    big_picture,
    explain,
    glossary,
    how_to_read,
    page_header,
    research_caveat,
)

MODES = ["static", "dynamic", "hybrid"]

st.set_page_config(page_title="Stock Agent", page_icon=":material/query_stats:", layout="wide")


@st.cache_data(show_spinner=False)
def _settings_snapshot() -> dict:
    s = load_settings()
    s.pop("_path", None)
    return s


def _pct(v) -> str:
    try:
        return f"{float(v) * 100:+.1f}%"
    except (TypeError, ValueError):
        return "—"


def _num(v, digits=2) -> str:
    try:
        return f"{float(v):,.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def main() -> None:
    page_header(
        ":material/query_stats:",
        "Stock Trend Agent",
        "Which stocks does the app like today, and what is the plan for each one?",
        "This page shows today's list and the reasoning behind it. It does not prove the list is any "
        "good — the Backtest, Robustness and Attribution pages are where we test that. Research "
        "output only, not financial advice.",
    )
    big_picture()
    how_to_read(
        [
            "**Ranked candidates** is today's shortlist, ordered by conviction (validated names first). "
            "The **Trend** column is the 0–100 score. Hover any column header for a definition.",
            "**Trend** is a 0–100 rank, **Conviction** is the rules' confidence, and "
            "**Valid** means the risk validator did not veto the trade plan.",
            "The **Analysis** cards explain *why* each name is shortlisted, with entry, stop and targets.",
            "A high score is **not** a validated edge — this page only shows the screen. "
            "The three analysis pages test whether the screen actually works.",
        ]
    )

    settings = load_settings()
    seed_universe = settings.get("universe", [])

    uc = dict(settings.get("universe_config", {}) or {})
    disc = dict(uc.get("discovery", {}) or {})
    configured_mode = str(uc.get("mode", "static")).lower()
    max_results = int(uc.get("max_results", 250))

    with st.sidebar:
        st.header("Run settings")
        provider = st.selectbox(
            "Analyst",
            options=["rules", "llm"],
            index=0,
            help="`rules` runs offline. `llm` adds narrative synthesis (needs OPENAI_API_KEY).",
        )

        mode = st.selectbox(
            "Universe source",
            options=MODES,
            index=MODES.index(configured_mode) if configured_mode in MODES else 0,
            help=(
                "`static` screens the configured seed list. `dynamic` discovers names "
                "from Yahoo's screener. `hybrid` unions both."
            ),
        )

        if mode != "static":
            with st.expander("Discovery gates", expanded=False):
                mc_billions = st.number_input(
                    "Min market cap ($B)",
                    min_value=0.0,
                    max_value=5000.0,
                    value=float(disc.get("min_market_cap", 2_000_000_000)) / 1e9,
                    step=0.5,
                )
                disc["min_market_cap"] = mc_billions * 1e9
                disc["min_price"] = st.number_input(
                    "Min price ($)",
                    min_value=0.0,
                    max_value=1000.0,
                    value=float(disc.get("min_price", 5.0)),
                    step=1.0,
                )
                disc["min_avg_volume_3m"] = st.number_input(
                    "Min 3m avg volume (shares)",
                    min_value=0,
                    max_value=100_000_000,
                    value=int(disc.get("min_avg_volume_3m", 500_000)),
                    step=100_000,
                )
                max_results = st.slider(
                    "Max discovered names", min_value=25, max_value=250, value=max_results, step=25
                )

            if st.button("Refresh universe", icon=":material/refresh:", width="stretch"):
                tmp = {
                    **settings,
                    "universe_config": {**uc, "mode": mode, "discovery": disc, "max_results": max_results},
                }
                with st.spinner("Querying Yahoo screener…"):
                    found = discover_universe(tmp, max_results=max_results, allow_network=True)
                st.session_state["discovery"] = found.to_dict()

            snapshot = st.session_state.get("discovery")
            if snapshot:
                st.caption(
                    f"Discovered **{len(snapshot.get('tickers', []))}** names · "
                    f"source `{snapshot.get('source')}` · {snapshot.get('discovered_at') or '—'}"
                )
                with st.expander("Preview discovered names", expanded=False):
                    preview = pd.DataFrame(
                        [
                            {"Ticker": t, "Company": (snapshot.get("names") or {}).get(t, "")}
                            for t in (snapshot.get("tickers") or [])[:50]
                        ]
                    )
                    st.dataframe(preview, width="stretch", hide_index=True)
            else:
                st.caption("No discovery yet — press **Refresh universe**.")

        if mode == "static":
            available = len(seed_universe)
        else:
            available = len((st.session_state.get("discovery") or {}).get("tickers", [])) or 50
        max_top = max(5, min(50, available or 5))
        default_top = min(int(settings["screening"]["top_n"]), max_top)
        top_n = st.slider("Shortlist size", 5, max_top, default_top)

        synthetic = st.toggle(
            "Offline / synthetic data",
            value=False,
            help="Use generated prices instead of live market data. Useful for demos and tests.",
        )
        use_cache = st.toggle("Use price cache", value=True)
        run = st.button("Run screen", icon=":material/play_arrow:", type="primary", width="stretch")

        st.divider()
        with st.expander("How the four pages fit together", icon=":material/account_tree:"):
            st.markdown(
                "1. **Dashboard** — today's ranked shortlist and trade plans (this page).\n"
                "2. **Backtest** — would the ranking have beaten simply holding the universe?\n"
                "3. **Robustness** — is any edge broad and stable, or a fluke of a few names?\n"
                "4. **Attribution** — is the edge skill, or just market and sector exposure?"
            )
        glossary()
        st.caption(f"Universe source: **{mode}** · Benchmark: **{settings['benchmark']}**")
        st.caption(f"Account: **${settings['risk']['account_size']:,}** · risk/trade **{settings['risk']['risk_per_trade_pct']}%**")

    run_settings = {
        **settings,
        "universe_config": {**uc, "mode": mode, "discovery": disc, "max_results": max_results},
    }

    if run:
        progress_bar = st.progress(0.0, text="Starting…")

        def on_progress(message: str, pct: float) -> None:
            progress_bar.progress(min(max(pct, 0.0), 1.0), text=message)

        try:
            with st.spinner("Running pipeline…"):
                result = run_pipeline(
                    settings=run_settings,
                    synthetic=synthetic,
                    provider=provider,
                    top_n=top_n,
                    use_cache=use_cache,
                    progress=on_progress,
                )
            st.session_state["result"] = result
            progress_bar.empty()
        except Exception as exc:
            progress_bar.empty()
            st.error(f"Pipeline failed: {exc}")

    result = st.session_state.get("result")
    if not result:
        st.info("Configure the run in the sidebar and press **Run screen**.")
        return

    # --- Market regime ----------------------------------------------------
    regime = result.get("regime") or {}
    passed = sum(1 for x in result.get("candidates", []) if x["validation"]["passed"])
    if passed == 0 and result.get("candidates"):
        answer(
            "warn",
            "No shortlisted name passed the risk validator.",
            "That is a normal outcome in a weak tape — the screen found trends, but none offered an acceptable risk plan.",
        )

    st.markdown("**At a glance**")
    with st.container(horizontal=True):
        st.metric(
            "Names screened",
            result.get("universe_size", 0),
            border=True,
            help="Liquid names that passed the price and dollar-volume filters.",
        )
        st.metric(
            "Shortlisted",
            len(result.get("candidates", [])),
            border=True,
            help="Names that received a full thesis and trade plan.",
        )
        st.metric(
            "Passed risk check",
            passed,
            border=True,
            help="Did not get vetoed by the ATR / conviction risk validator.",
        )
        if regime.get("available"):
            st.metric(
                f"{regime['benchmark']} 3-month",
                _pct(regime.get("ret_3m")),
                "above 200-day average" if regime.get("above_sma200") else "below 200-day average",
                border=True,
                help="Market regime: the benchmark's recent trend, used to judge whether the environment favours longs.",
            )

    st.caption(f"Run `{result['run_id']}` · as of {result['as_of']} · analyst `{result['provider']}`")

    universe_info = result.get("universe") or {}
    if universe_info:
        st.caption(
            f"Universe source: `{universe_info.get('source')}` · mode `{universe_info.get('mode')}` · "
            f"**{universe_info.get('size', 0)}** tradeable names"
            + (f" · discovered {universe_info['discovered_at']}" if universe_info.get("discovered_at") else "")
        )

    # --- Ranked table -----------------------------------------------------
    st.subheader("Ranked candidates", icon=":material/format_list_numbered:")
    st.caption(
        "Today's shortlist — validated names first, then highest conviction (not sorted by trend score). "
        "Hover a column header for a definition."
    )
    rows = []
    for t in result.get("candidates", []):
        tech = t.get("technicals", {})
        rel = t.get("relative_strength", {})
        pos = t.get("position", {})
        rows.append(
            {
                "Ticker": t["ticker"],
                "Company": t.get("name") or "",
                "Price": t.get("price"),
                "Trend": t.get("trend_score"),
                "Conv.": t.get("conviction"),
                "Dir": t.get("direction"),
                "3m": rela(tech.get("ret_3m"), rel.get("ret_3m")),
                "vs Bench": rel.get("rel_strength_3m"),
                "RSI": tech.get("rsi14"),
                "ATR%": tech.get("atr_pct"),
                "Stop": pos.get("stop"),
                "Shares": pos.get("shares"),
                "Value": pos.get("position_value"),
                "Valid": "✅" if t["validation"]["passed"] else "❌",
            }
        )
    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        width="stretch",
        hide_index=True,
        column_config={
            "Ticker": st.column_config.TextColumn(help="Ticker symbol."),
            "Company": st.column_config.TextColumn(help="Company name."),
            "Price": st.column_config.NumberColumn(format="%.2f", help="Last close."),
            "Trend": st.column_config.ProgressColumn(
                min_value=0, max_value=100, format="%.1f", help=explain("Trend score")
            ),
            "Conv.": st.column_config.NumberColumn(format="%.2f", help=explain("Conviction")),
            "Dir": st.column_config.TextColumn(help=explain("Long / watch / avoid")),
            "3m": st.column_config.NumberColumn(format="%+.1f%%", help="3-month price return."),
            "vs Bench": st.column_config.NumberColumn(
                format="%+.1f%%", help="3-month return minus the benchmark's — relative strength."
            ),
            "RSI": st.column_config.NumberColumn(
                format="%.0f", help="14-day Relative Strength Index. Above ~70 is often overbought; below ~30 oversold."
            ),
            "ATR%": st.column_config.NumberColumn(
                format="%.1f%%", help="Average True Range as % of price — volatility used to size the stop."
            ),
            "Stop": st.column_config.NumberColumn(format="%.2f", help="Suggested stop (entry − 2×ATR)."),
            "Shares": st.column_config.NumberColumn(format="%d", help="Position size from the risk model."),
            "Value": st.column_config.NumberColumn(format="$%,.0f", help="Dollar value of the position."),
            "Valid": st.column_config.TextColumn(
                help="✅ passed the risk validator · ❌ vetoed — see the thesis card for the reason."
            ),
        },
    )

    # --- Full ranked universe --------------------------------------------
    rankings = result.get("rankings") or []
    with st.expander(
        f"See all {result.get('universe_size', 0)} names that were screened",
        icon=":material/format_list_bulleted:",
        expanded=True,
    ):
        st.caption(
            "The table above shows only the top shortlist. This is the full ranked universe — "
            "type a ticker or company to check whether (and where) it was screened."
        )
        if rankings:
            rdf = pd.DataFrame(rankings).rename(
                columns={"ticker": "Ticker", "name": "Company", "trend_score": "Trend"}
            )
            query = st.text_input(
                "Find a ticker or company",
                placeholder="e.g. TELUS, XYZ, CVE.TO",
                key="ranking_search",
            )
            if query:
                q = query.strip().lower()
                rdf = rdf[
                    rdf["Ticker"].str.lower().str.contains(q, regex=False)
                    | rdf["Company"].str.lower().str.contains(q, regex=False)
                ]
            if rdf.empty:
                st.warning(f"No screened name matches {query!r}.")
            else:
                st.dataframe(
                    rdf,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Trend": st.column_config.ProgressColumn(
                            min_value=0,
                            max_value=100,
                            format="%.1f",
                            help=explain("Trend score"),
                        ),
                    },
                )
        else:
            st.caption("Re-run the screen to populate the full ranking.")

    # --- Thesis cards -----------------------------------------------------
    st.subheader("Analysis — why each name is here", icon=":material/description:")
    st.caption("Each card is a thesis with its own levels and risk plan. The first card starts expanded.")
    for i, t in enumerate(result.get("candidates", [])):
        name = t.get("name") or ""
        label = t["ticker"] + (f" — {name}" if name and name != t["ticker"] else "")
        title = f"{label} · {t['direction'].upper()} · conviction {t['conviction']:.2f} · trend {t['trend_score']:.0f}"
        with st.expander(title, expanded=(i == 0), icon=":material/description:"):
            badge_color = {"long": "green", "watch": "orange", "avoid": "red"}.get(t["direction"], "gray")
            st.badge(t["direction"].upper(), color=badge_color)
            left, right = st.columns([2, 1])
            with left:
                st.markdown(f"**Thesis** — {t['thesis']}")
                if t.get("catalysts"):
                    st.markdown("**Catalysts / bull case**")
                    for c in t["catalysts"]:
                        st.markdown(f"- {c}")
                if t.get("risks"):
                    st.markdown("**Risks / bear case**")
                    for r in t["risks"]:
                        st.markdown(f"- {r}")
                st.markdown(f"**Invalidation:** {t.get('invalidation', '—')}")
                if not t["validation"]["passed"]:
                    st.warning("Validator vetoed: " + "; ".join(t["validation"]["reasons"]))
            with right:
                lv = t.get("levels", {})
                pos = t.get("position", {})
                st.markdown("**Levels**")
                st.write(f"Support: {_num(lv.get('support_20d'))}")
                st.write(f"Resistance: {_num(lv.get('resistance_20d'))}")
                st.write(f"52w high: {_num(lv.get('high_52w'))}")
                st.markdown("**Trade plan**")
                st.write(f"Entry ~{_num(t.get('price'))}")
                st.write(f"Stop {_num(pos.get('stop'))}")
                st.write(f"Target 1 {_num(pos.get('target_1'))}")
                st.write(f"Target 2 {_num(pos.get('target_2'))}")
                st.write(f"R:R {pos.get('risk_reward_1')}")
                st.write(f"Size {pos.get('shares', 0)} sh (${_num(pos.get('position_value'), 0)})")
            st.caption(f"Source: {t.get('source')} · citations: {', '.join(t.get('citations', [])) or '—'}")

    # --- Downloads --------------------------------------------------------
    st.download_button(
        "Download markdown report",
        data=result.get("markdown", ""),
        file_name=f"stock-agent-{result['run_id']}.md",
        mime="text/markdown",
        icon=":material/download:",
        width="stretch",
    )

    if result.get("errors"):
        with st.expander(f"Data notes ({len(result['errors'])})", icon=":material/database:"):
            for err in result["errors"]:
                st.text(err)

    research_caveat()


def rela(a, b):
    return a if a is not None else b


if __name__ == "__main__":
    main()
