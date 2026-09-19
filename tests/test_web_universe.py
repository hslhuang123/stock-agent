"""Streamlit UI tests for the shared universe selector.

Uses Streamlit's first-party headless ``AppTest`` — no browser, no network, no
server. Guards the three analysis pages against render regressions and pins the
look-ahead warning on a discovered pool with a static backtest universe.
"""
from __future__ import annotations

import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from web.universe_controls import universe_caption  # noqa: E402


def test_universe_caption_reports_provenance():
    cap = universe_caption(
        {
            "universe": {
                "source": "cache-fresh",
                "mode": "dynamic",
                "size": 248,
                "discovered_at": "2026-09-11T20:48:16Z",
                "benchmark_excluded": True,
            }
        }
    )
    assert "cache-fresh" in cap and "dynamic" in cap and "248" in cap
    assert "benchmark excluded" in cap
    assert universe_caption({}) == ""


def test_universe_selector_warns_on_static_backtest_universe():
    script = (
        "import streamlit as st\n"
        "from core.config import load_settings\n"
        "from web.universe_controls import universe_selector\n"
        "universe_selector(load_settings(), key_prefix='t', show_backtest_mode=True)\n"
    )
    at = AppTest.from_string(script).run()
    assert not at.exception

    # Static by default: no warning.
    assert not at.warning

    at.segmented_control(key="t_universe_mode").set_value("dynamic").run()
    assert not at.exception
    assert at.segmented_control(key="t_universe_mode").value == "dynamic"
    assert any("look-ahead" in w.value for w in at.warning)

    at.segmented_control(key="t_backtest_mode").set_value("point_in_time_liquidity").run()
    assert not at.exception
    assert not at.warning


def test_analysis_pages_render_without_error():
    at = AppTest.from_file(str(ROOT / "web" / "app.py"), default_timeout=30).run()
    assert not at.exception

    for page in [
        "pages/1_Backtest.py",
        "pages/2_Robustness.py",
        "pages/3_Attribution.py",
        "pages/4_Simulator.py",
    ]:
        at.switch_page(page).run()
        assert not at.exception, f"{page} raised: {[str(e.value) for e in at.exception]}"
        labels = [c.label for c in at.segmented_control]
        assert "Universe source" in labels
