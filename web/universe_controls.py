"""Shared sidebar control for choosing the screened universe.

The Backtest / Robustness / Attribution pages previously always screened the
static seed list from ``config/settings.yaml``. This mirrors the dashboard's
universe selector so every page can run ``static`` / ``dynamic`` / ``hybrid``
resolution — and, for the backtest paths, choose a point-in-time liquidity
universe — with the same provenance guarantees as the CLI.

Place this module outside ``web/pages/`` so Streamlit does not treat it as a page.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from core.universe import discover_universe

UNIVERSE_MODES = ["static", "dynamic", "hybrid"]
BACKTEST_MODES = ["point_in_time_liquidity", "static"]


def universe_selector(
    settings: dict[str, Any],
    *,
    key_prefix: str,
    show_backtest_mode: bool = False,
) -> dict[str, Any]:
    """Render universe controls in the active container; return overridden settings.

    ``key_prefix`` keeps widget and discovery state independent between pages.
    ``show_backtest_mode`` adds the point-in-time vs static backtest-universe
    choice (only meaningful for pages that run a historical simulation).
    """
    uc = dict(settings.get("universe_config", {}) or {})
    disc = dict(uc.get("discovery", {}) or {})
    configured_mode = str(uc.get("mode", "static")).strip().lower()
    configured_backtest = str((uc.get("backtest") or {}).get("mode", "static")).strip().lower()
    max_results = int(uc.get("max_results", 250))

    mode = st.segmented_control(
        "Universe source",
        options=UNIVERSE_MODES,
        default=configured_mode if configured_mode in UNIVERSE_MODES else "static",
        key=f"{key_prefix}_universe_mode",
        help=(
            "`static` screens the configured seed list. `dynamic` discovers names "
            "from Yahoo's screener. `hybrid` unions both."
        ),
    ) or "static"

    if mode != "static":
        with st.expander("Discovery gates", expanded=False):
            disc["min_market_cap"] = (
                st.number_input(
                    "Min market cap ($B)",
                    min_value=0.0,
                    max_value=5000.0,
                    value=float(disc.get("min_market_cap", 2_000_000_000)) / 1e9,
                    step=0.5,
                    key=f"{key_prefix}_min_mcap",
                )
                * 1e9
            )
            disc["min_price"] = st.number_input(
                "Min price ($)",
                min_value=0.0,
                max_value=1000.0,
                value=float(disc.get("min_price", 5.0)),
                step=1.0,
                key=f"{key_prefix}_min_price",
            )
            disc["min_avg_volume_3m"] = st.number_input(
                "Min 3m avg volume (shares)",
                min_value=0,
                max_value=100_000_000,
                value=int(disc.get("min_avg_volume_3m", 500_000)),
                step=100_000,
                key=f"{key_prefix}_min_volume",
            )
            max_results = st.slider(
                "Max discovered names",
                min_value=25,
                max_value=250,
                value=min(max(max_results, 25), 250),
                step=25,
                key=f"{key_prefix}_max_results",
            )

        if st.button(
            "Refresh universe",
            icon=":material/refresh:",
            key=f"{key_prefix}_refresh",
        ):
            tmp = {
                **settings,
                "universe_config": {
                    **uc,
                    "mode": mode,
                    "discovery": disc,
                    "max_results": max_results,
                },
            }
            with st.spinner("Querying Yahoo screener…"):
                found = discover_universe(tmp, max_results=max_results, allow_network=True)
            st.session_state[f"{key_prefix}_discovery"] = found.to_dict()

        snapshot = st.session_state.get(f"{key_prefix}_discovery")
        if snapshot:
            st.caption(
                f"Discovered **{len(snapshot.get('tickers', []))}** names · "
                f"source `{snapshot.get('source')}` · {snapshot.get('discovered_at') or '—'}"
            )
        else:
            st.caption("No discovery this session — press **Refresh universe** (a recent cache is reused).")

    backtest_mode: str | None = None
    if show_backtest_mode:
        default_bt = configured_backtest if configured_backtest in BACKTEST_MODES else "static"
        if mode != "static" and default_bt == "static":
            default_bt = "point_in_time_liquidity"
        backtest_mode = st.segmented_control(
            "Backtest universe",
            options=BACKTEST_MODES,
            default=default_bt,
            key=f"{key_prefix}_backtest_mode",
            help=(
                "`point_in_time_liquidity` rebuilds the universe from price history at "
                "each rebalance (no look-ahead) — required for a discovered pool."
            ),
        ) or default_bt
        if mode != "static" and backtest_mode != "point_in_time_liquidity":
            st.warning(
                "Screening a discovered pool with a **static** backtest universe leaks "
                "look-ahead. Prefer `point_in_time_liquidity`."
            )

    overridden = {**uc, "mode": mode, "max_results": max_results, "discovery": disc}
    if show_backtest_mode and backtest_mode:
        overridden["backtest"] = {**(uc.get("backtest") or {}), "mode": backtest_mode}
    return {**settings, "universe_config": overridden}


def universe_caption(result: dict[str, Any]) -> str:
    """One-line provenance string for a result produced by a core suite."""
    uni = result.get("universe") or {}
    if not uni:
        return ""
    parts = [
        f"Universe: `{uni.get('source')}` · mode `{uni.get('mode')}` · **{uni.get('size', 0)}** names"
    ]
    if uni.get("discovered_at"):
        parts.append(f"discovered {uni['discovered_at']}")
    if uni.get("benchmark_excluded"):
        parts.append("benchmark excluded")
    return " · ".join(parts)
