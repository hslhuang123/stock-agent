"""Configuration loading with sane defaults."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"
DATA_DIR = PROJECT_ROOT / "data"
REPORT_DIR = PROJECT_ROOT / "reports"

DEFAULTS: dict[str, Any] = {
    "benchmark": "SPY",
    "universe": [],
    "universe_config": {
        # static  - use `universe` exactly (previous behaviour)
        # dynamic - discover from the screener, ignoring `universe`
        # hybrid  - discovered names unioned with `universe`
        "mode": "static",
        "max_results": 250,
        # Reuse a discovery cache younger than this many minutes (0 = always
        # re-query). Lets the UI refresh and the screen run share one lookup.
        "cache_minutes": 30,
        "discovery": {
            "region": "us",
            "exchanges": ["NYQ", "NMS", "NGM", "NCM", "ASE"],
            "min_price": 5.0,
            "min_market_cap": 2_000_000_000,
            "min_avg_volume_3m": 500_000,
            "screeners": [],
            "dedupe_companies": True,
            "sort_field": "intradaymarketcap",
            "sort_asc": False,
        },
        # Universe used by the walk-forward backtest.
        "backtest": {
            # point_in_time_liquidity - rebuild each rebalance date from price
            #                           history (look-ahead free). Required for
            #                           honest dynamic-universe backtests.
            # static                  - use the resolved list for every date.
            "mode": "static",
            "pool_size": 1000,
            "lookback_days": 20,
        },
    },
    "ingest": {
        # If a ticker returns no rows for the requested period, retry wider.
        "retry_periods": ["2y", "5y", "max"],
        # What to do when every retry fails: "drop" (record an error) or
        # "synthetic" (fabricate prices - tests/smoke runs only).
        "on_missing": "drop",
        "batch_size": 40,
        "batch_pause_seconds": 0.5,
    },
    "screening": {
        "lookback": "1y",
        "interval": "1d",
        "top_n": 15,
        "min_price": 5.0,
        "min_avg_dollar_volume": 5_000_000,
        "min_history_days": 130,
    },
    "scoring_weights": {
        "momentum_3m": 0.22,
        "momentum_6m": 0.13,
        "relative_strength_3m": 0.18,
        "trend_quality": 0.15,
        "volume_surge": 0.10,
        "proximity_52w_high": 0.09,
        "risk_adjusted_momentum": 0.13,
    },
    "risk": {
        "account_size": 100_000,
        "risk_per_trade_pct": 1.0,
        "atr_stop_multiple": 2.0,
        "max_position_pct": 10.0,
        "min_conviction": 0.45,
    },
    "agent": {
        "provider": "rules",
        "model": "gpt-4o-mini",
        "temperature": 0.2,
        "top_candidates": 8,
        "horizon": "2-6 weeks",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_settings(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML settings, merged over defaults."""
    path = Path(path) if path else DEFAULT_SETTINGS_PATH
    raw: dict[str, Any] = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    settings = _deep_merge(DEFAULTS, raw)
    settings["_path"] = str(path)
    return settings


def env_llm_config(settings: dict[str, Any]) -> dict[str, Any]:
    """Resolve LLM credentials from the environment.

    Returns an empty dict when no API key is present so callers can fall back
    to the deterministic rule-based analyst.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return {}
    return {
        "api_key": api_key,
        "base_url": os.getenv("OPENAI_BASE_URL", "").strip() or None,
        "model": os.getenv("AGENT_MODEL", "").strip()
        or settings.get("agent", {}).get("model", "gpt-4o-mini"),
    }
