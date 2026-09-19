# AGENT.md — guidance for AI coding agents

Instructions for any agent (or human) working in this repository. Read this before
changing code.

---

## 1. What this project is

A research harness for US equity screening: it detects trending stocks, produces
structured, risk-bounded trade theses, and — most importantly — **tries to falsify
itself**.

**Honest status: the strategy has no demonstrated edge.** Do not describe it as
profitable, validated, or ready for capital. The current evidence:

| Source of return | Verdict |
|---|---|
| Stock selection (within sector) | +3.3%/yr, t = 0.90 → not significant |
| Sector rotation timing | corr +0.037 → no predictive power |
| Hysteresis / turnover control | genuine operational win (49% → 22%) |
| Static tech/growth tilt | present, but a regime bet |
| Hindsight-selected universe | inflates all absolute results |

The verdict was re-confirmed on a **screener-discovered 248-name universe**
(P0c, 2026-09-12): edge vs universe fell +241% → **+36%**, IC +0.025 → **−0.005**,
and market alpha lost significance (t 2.59 → **1.41**). See `FINDINGS.md` F6. When
run on a dynamic universe, normalise the sector taxonomy first (ISSUE-009).

The **harness** is the asset, not the strategy. If asked to "improve returns",
prefer work that increases the strength of the falsification tests over work that
tunes parameters.

See **[FINDINGS.md](FINDINGS.md)** for the full research record, the open-issue
register (including a known benchmark-as-candidate bug with repro and fix), and
the prioritised work plan. Consult it before starting new work.

---

## 2. Environment

```bash
cd stock-agent
source .venv/bin/activate      # Python 3.11 — REQUIRED
```

- The system default is Python **3.14**, which lacks wheels for several deps.
  Always use `.venv` (3.11). Never `pip install` outside it.
- Installed: pandas 3.0, numpy 2.x, yfinance, streamlit, fastapi, openai, pytest.
- No API key is needed for anything except the optional LLM analyst
  (`OPENAI_API_KEY` in `.env`).

## 3. Commands

```bash
pytest -q                                        # 87 tests, ~4 min, fully offline
python scripts/daily_run.py --synthetic          # screen, no network
python scripts/daily_run.py                      # screen, live data
python scripts/backtest.py --lookback 10y
python scripts/robustness.py --trials 30
python scripts/attribution.py --lookback 10y
streamlit run web/app.py                         # dashboard (port 8501)
uvicorn api.main:app --reload                    # API (port 8000)
```

All CLI scripts accept `--synthetic` and write to `reports/<name>-latest.md|json`.

---

## 4. Architecture

```
core/                    headless library — MUST NOT import streamlit or fastapi
  config.py              YAML settings + env-var LLM config
  ingest/
    prices.py            yfinance + parquet cache; chunked fetch, retry ladder, drop policy
    company_names.py     static map -> yfinance -> JSON cache
  universe/              *** universe resolution (static/dynamic/hybrid) ***
    __init__.py          resolve_universe() - the single entry point
    discover.py          Yahoo screener discovery + gates + discovery cache
    point_in_time.py     per-rebalance liquidity universe (look-ahead free)
  features/
    indicators.py        pure pandas RSI/MACD/ATR/SMA (no TA-Lib)
    metrics.py           *** point-in-time metric engine ***
    trending.py          percentile ranking -> 0-100 trend score
  agent/
    tools.py             read-only tools + LLM JSON schemas
    analyst.py           deterministic Thesis builder
    llm.py               optional narrative refinement
  risk/rules.py          ATR position sizing + validator vetoes
  backtest/
    engine.py            reference walk-forward backtester
    robustness.py        Panel + fast evaluator + bootstrap/plateau/cost tests
    attribution.py       Jensen's alpha (Newey-West), sector-neutral, turnover
    sector_neutral.py    within-sector scoring + quota targeting
    rotation.py          rotation-vs-static-tilt diagnostic
    sectors.py           GICS map
    *_report.py          markdown renderers
  report/render.py       screen markdown
  pipeline.py            orchestration: ingest -> features -> screen -> analyze -> validate
api/                     thin FastAPI shell
web/                     Streamlit: app.py + pages/{1_Backtest,2_Robustness,3_Attribution}.py
scripts/                 CLI entrypoints
config/settings.yaml     seed universe, discovery gates, weights, risk, agent settings
tests/                   pytest suite
```

Layer rule: `web/` and `api/` may import `core/`; `core/` must never import them.

---

## 5. Hard invariants — do not break these

1. **No look-ahead bias.** Every historical metric must come from
   `core/features/metrics.py::metrics_at(df, idx)`, which only sees rows up to
   `idx`. Never compute a feature using the full series and then slice it.
   `tests/test_backtest.py::test_metrics_at_ignores_future_data` enforces this by
   tampering with future bars and asserting past metrics are unchanged.

2. **Numbers are deterministic; the LLM only narrates.** `core/agent/llm.py` may
   modify `thesis`, `catalysts`, `risks`, `direction`, and a bounded
   `conviction_delta` (±0.15). It must **never** alter price, levels, stop,
   targets, or position size. Those always come from `core/`.

3. **The fast evaluator must match the reference engine.**
   `core/backtest/robustness.py::evaluate_panel` is the optimized path used by
   bootstrap/sweeps. `tests/test_robustness.py::test_panel_matches_engine` pins it
   to `run_backtest`. If you change one, change the other and keep that test green.
   (This test previously caught a real bug: the engine treating the benchmark as a
   tradeable candidate.)

4. **The benchmark is never a candidate.** The benchmark is fetched for relative
   strength and the regime read, but must never enter the tradeable universe.
   `core/universe/__init__.py::resolve_universe` strips it, and `run_backtest`,
   `build_panel` and `run_pipeline` all select from a benchmark-free view.
   `tests/test_pipeline.py::test_benchmark_is_never_a_candidate` enforces it.
   (In the live pipeline the benchmark must stay *inside* `ranked`/`ctx.features`
   so `_regime_summary` and `get_market_regime` still work — exclude it from the
   tradeable view, not from the feature table.)

5. **Universe resolution has exactly one entry point.** Always get tickers from
   `core/universe/__init__.py::resolve_universe`, never from
   `settings["universe"]` directly. It implements static/dynamic/hybrid and
   records provenance (`source`, `discovered_at`). A report must never present a
   discovered universe as if it were the configured one.

6. **The live universe may be discovered; the backtest may not.** The screener
   returns only today's list, so using it historically is look-ahead. Backtests
   rebuild their universe per rebalance date from price history via
   `core/universe/point_in_time.py` (`backtest.mode: point_in_time_liquidity`)
   and use only bars up to that date.
   `tests/test_universe.py::test_point_in_time_universe_ignores_future_volume`
   and `test_panel_point_in_time_matches_engine` enforce this.

7. **Missing data is dropped, never fabricated.** `fetch_prices` retries with
   wider windows (`ingest.retry_periods`) and then omits the ticker with an error
   recorded. Synthetic prices are produced only when `synthetic=True` or
   `ingest.on_missing: synthetic` is set explicitly. Never reintroduce a silent
   fallback that mixes generated prices into a real screen.

8. **Tests must run offline.** Use `synthetic=True`, `use_cache=False`, and
   `allow_network=False`. Never add a test that requires the network. Tests that
   touch the discovery cache must monkeypatch `CACHE_PATH` to a temp path so they
   are independent of whatever is on disk.

9. **Fail soft on data.** yfinance is flaky. Every external lookup must degrade
   gracefully: cache → static map → synthetic → the ticker itself. Never let a
   missing name, sector, or price raise. This applies to discovery too: if the
   screener is unavailable, fall back to the cache and then the static list,
   recording *why* in `errors`.

10. **Empty universes/filters must not crash.** `rank_trending` returning an empty
    frame is a valid state; callers handle it.

---

## 6. Conventions

- Type hints on public functions; `from __future__ import annotations` at the top.
- Dataclasses for structured results (`Thesis`, `BacktestConfig`, `Panel`).
- Private helpers are `_prefixed`; `_stats` and `_summarize` are shared inside
  `core/backtest/` despite the underscore.
- Round stored returns to 4 dp (`_r(value, 4)`); full precision internally. This
  is why the panel-vs-engine test uses `atol=1e-4`.
- Weights in `config/settings.yaml` must sum to 1.0; factors are percentile-ranked
  before weighting.
- Weighted factors map via `WEIGHT_TO_COLUMN` in `core/features/trending.py`. If
  you add a factor, add it there, to `settings.yaml`, and to `metrics_at`.
- Reports are markdown; escape `|` in table cells (`_esc` in `core/report/render.py`).
- No new heavy dependencies without justification. Notably `scipy` is deliberately
  absent — Spearman is done as Pearson-on-ranks, and OLS/Newey-West is hand-rolled
  in `core/backtest/attribution.py::ols_nw`.

---

## 7. Adding a new factor

1. Add the computation to `core/features/metrics.py::metrics_at` (point-in-time!).
2. Add the column name to `WEIGHT_TO_COLUMN` in `core/features/trending.py`.
3. Add the weight to `scoring_weights` in `config/settings.yaml` (rebalance so the
   sum stays 1.0).
4. Run the full suite — `test_metrics_at_matches_feature_table` guards consistency
   between the live screen and the point-in-time path.

## 8. Adding a new screen/report

Follow the existing split: compute in `core/`, expose via a `scripts/*.py` CLI,
render markdown in a `*_report.py`, add a `web/pages/N_Name.py` page, and add
tests. New pages are auto-discovered by Streamlit.

---

## 9. Data sources (quick reference)

| Data | Source | Cache |
|---|---|---|
| OHLCV | yfinance | `data/prices/*.parquet`, 6 h TTL |
| Company names | static map → yfinance `.info` (screener returns names free) | `data/company_names.json` |
| Sectors | static GICS map → yfinance `.info` | `data/sectors.json` |
| Discovered universe | Yahoo equity screener (`yf.EquityQuery`) | `data/universe/discovered.json` |
| Fundamentals/news | yfinance (LLM tools only) | none |
| Static universe | `config/settings.yaml` → `universe:` | — |

Caveats to repeat when relevant: Yahoo data revises adjusted prices, excludes
delisted tickers (survivorship bias), and rate-limits large universes. The
screener returns only *today's* membership, so it must never be used to build a
historical universe — see invariant 6.

---

## 10. Reporting results honestly

When you report backtest numbers, always:

- Compare against the **equal-weight universe**, not just SPY — universe selection
  is not skill.
- Report the **sector-neutral** excess, because most of this strategy's apparent
  alpha is sector tilting.
- State **t-statistics**, not just point estimates. Overlapping windows make naive
  t-stats too generous, so use the Newey-West path.
- Flag that the universe is **hindsight-selected** and inflates absolute returns.
- Avoid quoting large percentage-point differences of cumulative returns as if
  they were proportional (e.g. "+488% vs −155%" is a ~1.5× ratio, not a 643pp win).

If a change makes the strategy look better, first assume you introduced a bug or a
bias, and try to prove it.
