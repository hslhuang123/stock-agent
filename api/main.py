"""FastAPI layer over the core pipeline.

Long-running screens are dispatched to a background worker; clients poll for
status. The core pipeline is never modified here — this is a thin shell.

Run:
    uvicorn api.main:app --reload --port 8000
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from threading import Lock
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core.agent.analyst import analyze
from core.agent.tools import MarketContext, dispatch
from core.config import REPORT_DIR, load_settings
from core.ingest.prices import fetch_prices
from core.pipeline import run_pipeline, save_result

app = FastAPI(title="Stock Agent API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_RUNS: dict[str, dict[str, Any]] = {}
_LOCK = Lock()


class ScreenRequest(BaseModel):
    top_n: int | None = Field(default=None, ge=1, le=200)
    synthetic: bool = False
    provider: str | None = Field(default=None, pattern="^(rules|llm)$")


class ChatRequest(BaseModel):
    ticker: str
    message: str = "Give me your current read on this name."
    synthetic: bool = False


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}


def _execute_run(run_id: str, req: ScreenRequest) -> None:
    try:
        with _LOCK:
            _RUNS[run_id]["status"] = "running"
        result = run_pipeline(
            settings=load_settings(),
            synthetic=req.synthetic,
            provider=req.provider,
            top_n=req.top_n,
        )
        json_path, md_path = save_result(result)
        with _LOCK:
            _RUNS[run_id].update(
                {
                    "status": "completed",
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "result": result,
                    "report_path": str(md_path),
                    "json_path": str(json_path),
                }
            )
    except Exception as exc:  # pragma: no cover
        with _LOCK:
            _RUNS[run_id].update(
                {"status": "failed", "error": str(exc), "finished_at": datetime.now(timezone.utc).isoformat()}
            )


@app.post("/runs")
def create_run(req: ScreenRequest, background: BackgroundTasks) -> dict[str, Any]:
    run_id = uuid.uuid4().hex[:12]
    with _LOCK:
        _RUNS[run_id] = {
            "run_id": run_id,
            "status": "queued",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "request": req.model_dump(),
        }
    background.add_task(_execute_run, run_id, req)
    return {"run_id": run_id, "status": "queued"}


@app.get("/runs")
def list_runs() -> list[dict[str, Any]]:
    with _LOCK:
        return [
            {k: v for k, v in run.items() if k != "result"}
            for run in _RUNS.values()
        ]


@app.get("/runs/{run_id}")
def get_run(run_id: str, include_result: bool = True) -> dict[str, Any]:
    with _LOCK:
        run = _RUNS.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run not found")
    if not include_result:
        return {k: v for k, v in run.items() if k != "result"}
    return run


@app.get("/reports/latest")
def latest_report() -> dict[str, str]:
    path = REPORT_DIR / "latest.md"
    if not path.exists():
        raise HTTPException(status_code=404, detail="no report yet")
    return {"markdown": path.read_text(encoding="utf-8")}


# --------------------------------------------------------------------------- #
# Interactive single-ticker analysis (the "chat with the analyst" route)
# --------------------------------------------------------------------------- #

@app.post("/analyze/{ticker}")
def analyze_ticker(ticker: str, provider: str = "rules", synthetic: bool = False) -> dict[str, Any]:
    ticker = ticker.upper()
    settings = load_settings()
    benchmark = settings["benchmark"]
    prices, _ = fetch_prices([ticker, benchmark], synthetic=synthetic)
    if ticker not in prices:
        raise HTTPException(status_code=404, detail=f"no data for {ticker}")

    # Build a minimal context (no cross-sectional scan needed for one name).
    from core.features.indicators import add_indicators
    from core.features.trending import compute_feature_table, rank_trending

    bench = add_indicators(prices.pop(benchmark))
    prices[benchmark] = bench
    table = compute_feature_table(prices, benchmark=bench)
    ranked = rank_trending(table, settings["scoring_weights"], settings["screening"])
    if ticker not in ranked.index:
        ranked = table  # skip filters so single names still respond

    ctx = MarketContext(prices=prices, features=ranked, benchmark=benchmark)
    thesis = analyze(ctx, ticker, settings, provider=provider)
    return thesis.to_dict()


@app.post("/chat")
def chat(req: ChatRequest) -> dict[str, Any]:
    """Introspect the tools directly — useful for debugging/debug UI.

    (Wire an LLM here if you want free-form conversation; the deterministic
    tool results are what an LLM would receive.)
    """
    settings = load_settings()
    benchmark = settings["benchmark"]
    ticker = req.ticker.upper()
    prices, _ = fetch_prices([ticker, benchmark], synthetic=req.synthetic)

    from core.features.indicators import add_indicators
    from core.features.trending import compute_feature_table, rank_trending

    bench = add_indicators(prices.pop(benchmark))
    prices[benchmark] = bench
    table = compute_feature_table(prices, benchmark=bench)
    ranked = rank_trending(table, settings["scoring_weights"], settings["screening"])
    ctx = MarketContext(prices=prices, features=ranked if ticker in ranked.index else table, benchmark=benchmark)

    return {
        "ticker": ticker,
        "message": req.message,
        "tools": {
            "trend_score": dispatch(ctx, "get_trend_score", {"ticker": ticker}),
            "technicals": dispatch(ctx, "get_technicals", {"ticker": ticker}),
            "levels": dispatch(ctx, "get_levels", {"ticker": ticker}),
            "relative_strength": dispatch(ctx, "get_relative_strength", {"ticker": ticker}),
            "market_regime": dispatch(ctx, "get_market_regime", {}),
        },
    }
