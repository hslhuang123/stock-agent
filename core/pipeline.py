"""End-to-end pipeline: ingest -> features -> screen -> analyze -> validate."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from core.agent.analyst import analyze
from core.agent.tools import MarketContext
from core.config import REPORT_DIR, load_settings
from core.features.indicators import add_indicators
from core.features.trending import compute_feature_table, rank_trending
from core.ingest.company_names import get_company_names
from core.ingest.prices import fetch_prices
from core.report.render import render_markdown
from core.risk.rules import validate_thesis
from core.universe import resolve_universe

log = logging.getLogger(__name__)

ProgressFn = Callable[[str, float], None]


def _noop(_msg: str, _pct: float) -> None:  # pragma: no cover
    return None


def run_pipeline(
    settings: dict[str, Any] | None = None,
    synthetic: bool = False,
    provider: str | None = None,
    top_n: int | None = None,
    use_cache: bool = True,
    progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """Run the full screen and return a JSON-serializable result."""
    settings = settings or load_settings()
    progress = progress or _noop

    screening = settings["screening"]
    benchmark = settings["benchmark"]
    ingest = settings.get("ingest", {}) or {}

    # Resolve the tradeable universe (static / dynamic / hybrid). The benchmark
    # is excluded from the tradeable set here and fetched separately below.
    resolution = resolve_universe(settings, allow_network=not synthetic, use_cache=use_cache)
    universe = resolution.tickers
    if not universe:
        raise ValueError(
            "universe is empty — check universe_config and config/settings.yaml"
        )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    errors: list[str] = list(resolution.errors)

    # 1. Ingest ------------------------------------------------------------
    progress("Fetching price data…", 0.05)
    all_tickers = list(dict.fromkeys(universe + [benchmark]))
    prices, ingest_errors = fetch_prices(
        all_tickers,
        period=screening.get("lookback", "1y"),
        interval=screening.get("interval", "1d"),
        use_cache=use_cache,
        synthetic=synthetic,
        retry_periods=ingest.get("retry_periods"),
        on_missing=ingest.get("on_missing", "drop"),
        batch_size=int(ingest.get("batch_size", 40)),
        batch_pause_seconds=float(ingest.get("batch_pause_seconds", 0.5)),
    )
    errors.extend(ingest_errors)

    if benchmark not in prices or len(prices[benchmark]) < 60:
        errors.append(f"{benchmark}: benchmark data insufficient — relative strength disabled")
        bench_frame = None
    else:
        bench_frame = add_indicators(prices.pop(benchmark))
        prices[benchmark] = bench_frame

    # 2. Features ----------------------------------------------------------
    progress("Computing indicators and trend features…", 0.3)
    table = compute_feature_table(
        prices,
        benchmark=bench_frame,
        min_history_days=int(screening.get("min_history_days", 130)),
    )
    if table.empty:
        raise RuntimeError("no tickers produced features — check data availability")

    # 3. Screen ------------------------------------------------------------
    progress("Ranking universe by trend score…", 0.5)
    ranked = rank_trending(table, settings["scoring_weights"], screening)
    if ranked.empty:
        raise RuntimeError("no tickers passed the liquidity filters")

    n = int(top_n or screening.get("top_n", 15))

    # The benchmark is fetched for relative strength and the regime read, but it
    # is not a tradeable candidate (invariant: AGENT.md §5.4). Keep it in
    # `ranked` so `_regime_summary` and `get_market_regime` still work, and
    # expose a separate tradeable view for selection and reporting.
    tradeable = ranked.drop(index=benchmark, errors="ignore")
    if tradeable.empty:
        raise RuntimeError("no tradeable candidates after excluding the benchmark")

    shortlist = tradeable.head(n)

    ctx = MarketContext(prices=prices, features=ranked, benchmark=benchmark)

    # 4. Analyze + validate ------------------------------------------------
    candidates: list[dict[str, Any]] = []
    n_candidates = int(settings.get("agent", {}).get("top_candidates", n))
    for i, ticker in enumerate(shortlist.index[:n_candidates]):
        progress(f"Analyzing {ticker}…", 0.5 + 0.45 * (i / max(1, n_candidates)))
        thesis = analyze(ctx, ticker, settings, provider=provider)
        passed, reasons = validate_thesis(thesis.to_dict(), settings["risk"])
        thesis.validation = {"passed": passed, "reasons": reasons}
        candidates.append(thesis.to_dict())

    candidates.sort(key=lambda c: (c["validation"]["passed"], c["conviction"], c["trend_score"]), reverse=True)

    # Attach company names: prefer curated/static or fetched names, then the
    # names the screener already returned for free, then the ticker itself.
    name_tickers = list(dict.fromkeys([c["ticker"] for c in candidates] + list(shortlist.index)))
    fetched_names = get_company_names(name_tickers, allow_network=not synthetic)
    names: dict[str, str] = {}
    for ticker in name_tickers:
        name = fetched_names.get(ticker, ticker)
        if name == ticker:
            name = resolution.names.get(ticker, ticker)
        names[ticker] = name
    for candidate in candidates:
        candidate["name"] = names.get(candidate["ticker"], candidate["ticker"])

    progress("Rendering report…", 0.97)

    all_names = get_company_names(list(tradeable.index), allow_network=False, persist=False)

    regime = _regime_summary(ranked, benchmark)
    result: dict[str, Any] = {
        "run_id": run_id,
        "as_of": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "provider": (provider or settings["agent"]["provider"]),
        "benchmark": benchmark,
        "universe": resolution.to_dict(),
        "universe_size": len(tradeable),
        "shortlist": [
            {
                "ticker": t,
                "name": names.get(t, t),
                "trend_score": float(shortlist.loc[t, "trend_score"]),
                "conviction_rank": float(shortlist.loc[t, "trend_score"]),
            }
            for t in shortlist.index
        ],
        "rankings": [
            {
                "ticker": t,
                "name": (
                    names.get(t)
                    if names.get(t) and names.get(t) != t
                    else (all_names.get(t) or t)
                ),
                "trend_score": float(ranked.loc[t, "trend_score"]),
            }
            for t in tradeable.index
        ],
        "regime": regime,
        "candidates": candidates,
        "errors": errors,
        "settings_path": settings.get("_path"),
    }
    result["markdown"] = render_markdown(result)

    progress("Done.", 1.0)
    return result


def _regime_summary(ranked: pd.DataFrame, benchmark: str) -> dict[str, Any]:
    if benchmark not in ranked.index:
        return {"benchmark": benchmark, "available": False}
    row = ranked.loc[benchmark]
    return {
        "benchmark": benchmark,
        "available": True,
        "close": round(float(row["close"]), 2),
        "ret_1m": _num(row.get("ret_1m")),
        "ret_3m": _num(row.get("ret_3m")),
        "ann_vol": _num(row.get("ann_vol")),
        "above_sma50": bool(row["close"] > row["sma50"]) if pd.notna(row.get("sma50")) else None,
        "above_sma200": bool(row["close"] > row["sma200"]) if pd.notna(row.get("sma200")) else None,
    }


def _num(value: Any) -> Any:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(f) else round(f, 4)


def save_result(result: dict[str, Any]) -> tuple[Path, Path]:
    """Persist JSON + markdown. Returns (json_path, md_path)."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / f"{result['run_id']}.json"
    md_path = REPORT_DIR / f"{result['run_id']}.md"
    latest_md = REPORT_DIR / "latest.md"
    latest_json = REPORT_DIR / "latest.json"

    payload = json.dumps(result, default=str, indent=2)
    json_path.write_text(payload, encoding="utf-8")
    md_path.write_text(result.get("markdown", ""), encoding="utf-8")
    latest_json.write_text(payload, encoding="utf-8")
    latest_md.write_text(result.get("markdown", ""), encoding="utf-8")
    return json_path, md_path
