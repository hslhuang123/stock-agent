# Stock Trend Agent

An agentic pipeline that **detects trending US equities, analyzes them, and produces
structured, risk-bounded trade theses**.

> ⚠️ Research and educational output only. **Not financial advice.** Nothing here
> places orders. Paper-trade and validate before risking capital.

> 🤖 Working on this codebase (human or AI)? Read **[AGENT.md](AGENT.md)** first —
> it documents the hard invariants (no look-ahead, deterministic numbers, the
> engine/evaluator consistency pin) and the honest research status.
>
> 📋 **[FINDINGS.md](FINDINGS.md)** — what the research established and the open
> issues, including a known bug (benchmark treated as a tradeable candidate) with
> reproduction steps and a proposed fix.
>
> ⚠️ **Status update (2026-09-12, P0c):** the sample results in this README use
> the original **static, hindsight-selected 58-name universe**. The gate has since
> been re-run on a **screener-discovered 248-name universe** and the verdict held —
> and weakened: edge vs universe **+241% → +36%**, IC **+0.025 → −0.005**, and the
> market alpha lost significance (**t 2.59 → 1.41**). The signal is essentially
> dead on a discovered pool. Full comparison: **FINDINGS.md → F6**. Note that
> `reports/*-latest.*` now contain the discovered-universe runs; the static-58
> baselines are the `reports/*-20260911T*.md` files.

## Design principle

Numbers are **deterministic**; the LLM only **narrates**.

```
ingest (prices)  ->  features (indicators)  ->  screen (rank)  ->  agent (thesis)  ->  validator (veto)
    yfinance          pure pandas/numpy         percentile ranks     rules or LLM        risk rules
```

The LLM can call read-only tools but can never change prices, levels, stop
distances, or position sizes. Every candidate passes a deterministic validator
that can veto it on risk grounds.

## Quickstart

```bash
cd stock-agent
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Offline demo — no network, no API key needed:
python scripts/daily_run.py --synthetic

# Live market data, rule-based analyst:
python scripts/daily_run.py

# Add the LLM narrative layer:
cp .env.example .env   # then set OPENAI_API_KEY
python scripts/daily_run.py --provider llm
```

Reports land in `reports/<run_id>.md`, `reports/<run_id>.json`, plus
`reports/latest.md`.

## Web UI (Streamlit)

```bash
source .venv/bin/activate
streamlit run web/app.py            # http://localhost:8501
```

Useful flags:

```bash
streamlit run web/app.py --server.port 8502
streamlit run web/app.py --server.headless true    # remote / SSH box
```

The app resolves the project root itself, so it runs from any working directory
(as long as `.venv` is the active interpreter).

Four pages, auto-discovered from `web/pages/`:

| Page | URL | What it shows |
|---|---|---|
| **Dashboard** | `/` | Ranked shortlist (with company names), thesis cards, trade plans, markdown download |
| **Backtest** | `/Backtest` | Walk-forward run: equity curves, fold stability, IC, controls |
| **Robustness** | `/Robustness` | Bootstrap distribution, parameter plateau, cost survival |
| **Attribution** | `/Attribution` | Jensen's alpha, sector-neutral benchmark, rotation diagnostic |

Every page has an **Offline / synthetic data** toggle in the sidebar, so you can
explore the whole thing without network access or an API key. All four pages also
let you choose the **universe source** (`static` / `dynamic` / `hybrid`); the three
analysis pages additionally choose a **point-in-time vs static backtest universe**
and warn if a discovered pool is paired with a static (look-ahead) backtest
universe. Long runs (backtest, robustness, attribution) report progress through a
progress bar and write markdown to `reports/`.

If a page fails to load, check the terminal running Streamlit first — import or
runtime errors surface there, not in the browser.

## API (FastAPI)

```bash
uvicorn api.main:app --reload --port 8000
```

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/runs` | Queue a screen; returns `run_id` immediately |
| `GET`  | `/runs/{id}` | Poll status + results |
| `GET`  | `/reports/latest` | Latest markdown report |
| `POST` | `/analyze/{ticker}` | Single-name thesis |
| `POST` | `/chat` | Raw tool outputs for a ticker (debug) |

```bash
curl -X POST localhost:8000/runs -H 'content-type: application/json' \
     -d '{"top_n": 10, "synthetic": false}'
curl localhost:8000/runs/<run_id>
```

## Tests

The suite is **87 tests** and runs **fully offline** — no network, no API key, no
fixtures to download. Tests use the synthetic price generator with
`use_cache=False`, so they are deterministic and safe to run in CI or on a plane.

```bash
source .venv/bin/activate    # Python 3.11 — see Quickstart
pip install pytest
pytest -q                    # ~4.5 minutes
```

Run a subset while iterating:

```bash
pytest tests/test_company_names.py -q    # <1s  — smoke test after env changes
pytest tests/test_web_universe.py -q     # ~2s  — Streamlit pages render + universe selector
pytest tests/test_universe.py -q         # ~12s — discovery, point-in-time, ingest policy
pytest tests/test_rotation.py -q         # ~15s
pytest tests/test_backtest.py -q         # ~20s — look-ahead safety, engine mechanics
pytest tests/test_pipeline.py -q         # ~20s — indicators, scoring, risk, end-to-end
pytest tests/test_attribution.py -q      # ~60s — OLS/HAC, sector-neutral, hysteresis
pytest tests/test_sector_neutral.py -q   # ~75s
pytest tests/test_robustness.py -q       # ~100s — panel ⇔ engine consistency
```

| File | Tests | Covers |
|---|---|---|
| `test_universe.py` | 18 | Discovery filters/de-dupe, point-in-time reconstruction, retry ladder, drop policy |
| `test_attribution.py` | 15 | OLS/Newey-West correctness, Jensen's alpha, sector weights, hysteresis, universe resolution |
| `test_pipeline.py` | 13 | Indicators, trend scoring, liquidity filters, position sizing, validator vetoes, end-to-end run |
| `test_sector_neutral.py` | 12 | Largest-remainder quotas, within-sector ranking, selection |
| `test_backtest.py` | 8 | Point-in-time safety, stats, costs, date filtering |
| `test_robustness.py` | 8 | Panel ⇔ engine equivalence, bootstrap, sweeps, cost grids |
| `test_company_names.py` | 6 | Name lookup, offline fallback, cache behaviour |
| `test_rotation.py` | 4 | Rotation-vs-static-tilt diagnostic |
| `test_web_universe.py` | 3 | Streamlit universe selector + page render smoke tests |

### The tests that matter most

If you change anything in `core/features/`, `core/backtest/` or `core/universe/`,
these are the ones that protect you from silently producing wrong numbers:

| Test | What it protects |
|---|---|
| `test_metrics_at_ignores_future_data` | **No look-ahead in metrics.** Tampers with future bars and asserts past metrics are byte-identical. |
| `test_point_in_time_universe_ignores_future_volume` | **No look-ahead in the universe.** Same technique applied to universe membership. |
| `test_panel_point_in_time_matches_engine` | The fast evaluator agrees with the reference engine under a *dynamic* universe. |
| `test_panel_matches_engine` | The fast panel evaluator returns the *same* results as the reference backtester. This caught a real bug (benchmark treated as a tradeable candidate). |
| `test_metrics_at_matches_feature_table` | The live screen and the historical path compute identical values. |
| `test_backtest_costs_reduce_returns` | Costs reduce net returns, while gross stays cost-independent. |
| `test_price_at_never_returns_future_price` | Date lookup never returns a future price. |
| `test_benchmark_is_never_a_candidate` | The benchmark is fetched for regime but never offered as a tradeable pick. |
| `test_missing_drops_by_default` | Missing data is dropped and recorded — never silently replaced with synthetic prices. |
| `test_cached_discovery_respects_max_results` | A smaller `max_results` trims the result, not the on-disk cache. |
| `test_static_names_cover_default_universe` / `test_static_sector_map_covers_default_universe` | Offline data maps stay complete — add a ticker to `config/settings.yaml` and these fail loudly if it lacks a name or sector. |

Full CLI/UI verification (not part of `pytest`):

```bash
python scripts/daily_run.py --synthetic    # end-to-end screen, offline
python scripts/daily_run.py --universe-mode dynamic --max-names 100   # live discovery
python scripts/backtest.py --synthetic --lookback 3y
python scripts/backtest.py --backtest-universe point_in_time_liquidity
python scripts/robustness.py --synthetic --trials 10
python scripts/attribution.py --synthetic --lookback 3y
streamlit run web/app.py                   # then visit /, /Backtest, /Robustness, /Attribution
```

## Walk-forward backtest

The backtester re-uses the *same* metric code (`core/features/metrics.py`) at every
historical rebalance, so there is no look-ahead by construction. Delisted tickers
trade at their last print instead of vanishing (no survivorship shortcut), and
commission + slippage are charged on turnover at every rebalance.

```bash
python scripts/backtest.py --lookback 10y --rebalance 21 --hold 21 --top 10
python scripts/backtest.py --synthetic          # offline
python scripts/backtest.py --cost-bps 25 --folds 6
```

Output: `reports/backtest-<id>.md` and `backtest-latest.md`, plus a Streamlit page.

It reports four things that matter:

| Output | What it tells you |
|---|---|
| **Controls** | Top-N vs a *random* N-name basket vs the *equal-weight universe* |
| **Information coefficient** | Rank correlation between trend score and forward return |
| **Quintile spread** | Whether higher scores monotonically earn higher returns |
| **Fold stability** | Whether edge survives across out-of-sample sub-periods |

### Sample live result (10y, monthly, top 10)

| | Strategy (net) | Random basket | Universe EW | SPY |
|---|---|---|---|---|
| Total return | +1016% | +807% | +775% | +277% |
| CAGR | +29.2% | +26.4% | +25.9% | +15.1% |
| Sharpe | 1.16 | 1.22 | 1.22 | 0.90 |
| Max DD | −33.7% | −32.5% | −32.9% | −31.7% |

Read this honestly: the screen beat random and equal-weight by ~+210–240% over
9.5 years, so the *ranking* adds something. But the universe itself returned
+775% vs SPY's +277% — most of the headline "alpha" is the choice of universe
(today's mega-caps, selected with hindsight), not the signal. IC is a weak 0.025
and the quintile ordering is nearly flat (Q5 +2.5% vs Q1 +2.2%, with Q2 below Q1).
One of four folds produced **negative** alpha.

**Conclusion: promising, not proven.** Before trusting it, replace the universe
with a point-in-time index membership list and re-run.

## Robustness gate

`scripts/robustness.py` (or the 🛡️ Streamlit page) runs three cheap tests that
decide whether the edge is real before you invest in a universe rebuild:

1. **Universe bootstrap** — repeat on many random subsets (default 30 trials, drop 30%).
2. **Parameter plateau** — sweep `top_n` × `rebalance_days`.
3. **Cost sensitivity** — 0 → 50 bps per side.

```bash
python scripts/robustness.py --lookback 10y --trials 30 --drop 0.30
```

### Sample live result (10y, top 10, 10 bps)

**Verdict: ⚠️ PARTIAL (2/3)**

| Test | Result | Evidence |
|---|---|---|
| Universe bootstrap | ✅ PASS | **100%** of 30 subsets kept a positive edge (median **+375%**) |
| Parameter plateau | ✅ PASS\* | 92% of 12 configs positive, but spread **1352pp** |
| Cost survival | ❌ **FAIL** | edge +241% at 10 bps → +22% at 30 bps → **−155% at 50 bps** |

\* passes the count threshold but the spread is the real story — see below.

Cost is the binding constraint:

| Cost (bps/side) | Net total | Universe EW | **Edge** |
|---|---|---|---|
| 0 | +1145% | +775% | **+370%** |
| 10 | +1016% | +775% | **+241%** |
| 30 | +797% | +775% | **+22%** |
| 50 | +620% | +775% | **−155%** |

Edge vs universe by configuration (rows = rebalance, cols = positions):

| Rebalance \ Positions | 5 | 10 | 20 | 30 |
|---|---|---|---|---|
| 5d | +1058% | +507% | +9% | −83% |
| 21d | +1039% | +241% | +302% | +71% |
| 63d | +1269% | +562% | +246% | +125% |

**Interpretation.** The bootstrap is genuinely encouraging — the edge is broad,
not concentrated in a few tickers. But two things are damning: (1) the edge is
**monotonically destroyed by transaction costs**, surviving 10 bps and gone by 30;
and (2) the edge **decays as position count rises** (5 names ≫ 30 names), which is
the signature of concentration/beta rather than a stable cross-sectional factor.

**Verdict: not tradable as-is.** The problem is signal design (turnover, beta
exposure), not the universe. Fix those before spending effort on point-in-time
membership.

## Factor attribution

`scripts/attribution.py` (or the 🔬 Streamlit page) answers the go/no-go question:
**is the edge skill, or just beta and sector exposure?**

```bash
python scripts/attribution.py --lookback 10y --top 10
```

It runs three analyses:

1. **Jensen's alpha** — regress strategy returns on the benchmark with Newey-West
   (HAC) standard errors. Overlapping windows make naive t-stats too generous.
2. **Sector-neutral benchmark** — a portfolio holding the strategy's *sector
   weights* but picking names at random within each sector. Beating it requires
   stock selection, not sector allocation.
3. **Turnover variants** — `hysteresis` bands (`hold until rank < top_n × exit×`)
   and longer holds, each re-tested for cost survival *and* sector-neutral alpha.

### Sample live result (10y, top 10)

**Verdict: ⚠️ PARTIAL (2/3)**

| Test | Result | Evidence |
|---|---|---|
| Market alpha (Jensen) | ✅ PASS | **+12.2%** annualized, **t = 2.59**, beta 1.07, R² 0.56 |
| Sector-neutral excess | ❌ **FAIL** | **+3.3%** annualized, **t = 0.90** — not significant |
| Cost-viable variant @ 50bps | ✅ PASS | 5/6 variants keep a positive edge |

Turnover variants (the practical breakthrough — and its limit):

| Rebalance | Selection | Turnover | Edge @50bps | Alpha (ann.) | Beta | **Sector-neutral** |
|---|---|---|---|---|---|---|
| 21d | top-N | 49% | **−155%** | +12.2% | 1.07 | +3.3% (t=0.90) |
| 21d | hysteresis ×1.5 | 34% | +375% | +15.9% | 1.16 | +6.3% (t=1.60) |
| 21d | hysteresis ×2.5 | **22%** | +488% | +15.5% | 1.17 | +5.6% (t=1.38) |
| 63d | top-N | 73% | +303% | +14.2% | 1.19 | +7.0% (t=1.69) |
| 63d | hysteresis ×2.5 | 47% | +340% | +11.9% | 1.34 | +6.8% (t=1.64) |

Average sector exposure: **40.2% Information Technology**.

**Interpretation — the decisive finding.**

- There *is* significant alpha against market beta (+12.2%, t=2.59). It is not a
  levered index fund.
- But **once sector allocation is matched, the excess collapses to +3.3% with
t  = 0.90** — statistically indistinguishable from zero. The apparent edge is
  overwhelmingly **sector tilting**, not stock selection.
- Turnover reduction is a genuine win: hysteresis cuts turnover 49% → 22% and
  turns a −155% edge at 50 bps into +488%. But it does **not** create
  stock-selection alpha — no variant clears t = 2 on the sector-neutral test.

**Conclusion: this is a sector-rotation engine wearing a stock-picker's clothes.**

### Sector-neutral scoring (the direct test)

Instead of ranking all names together, `sector_neutral` scoring percentile-ranks each
factor **within its sector** and fixes sector quotas. That removes the sector bet
and forces the signal to compete on stock selection alone.

| Approach | Variants | Mean turnover | Mean sector-neutral alpha | Best t | Significant |
|---|---|---|---|---|---|
| Pooled | 6 | 47% | +6.0% | 1.69 | **0** |
| Sector-neutral | 6 | 50% | +5.8% | 1.74 | **0** |

Best of all 12 variants: **63d / sector-neutral / hysteresis ×1.5 → +8.2%/yr, t = 1.63**.

**Result: sector-neutral scoring does not rescue the signal.** All twelve variants
produce a *positive* within-sector excess (+3.3% to +8.2%) but **none clears t = 2**.
The t-stats cluster at 1.1–1.7 — weak, consistent evidence at best, and 113 monthly
observations over 9.5 years is not enough to resolve it. Given the universe is still
hindsight-selected, the realistic posterior is ~0.

**Bottom line: no demonstrated stock-selection edge. The signal's value is sector allocation.**

### Rotation or static tilt?

The remaining question: is that sector allocation *skill* (dynamic rotation) or just a
static regime tilt? `core/backtest/rotation.py` correlates each period's sector
weight with that same period's sector return.

| Measure | Value |
|---|---|
| Correlation (weight vs same-period sector return) | **+0.037** |
| Rotation turbulence (mean weight std) | 11.0% |
| Dynamic? | yes |
| **Predictive?** | **no** |

Information Technology averaged 40.2% but ranged from **0% to 100%** (std 21.1%) —
the book moves a lot. It just doesn't move *usefully*: the correlation with
subsequent sector returns is +0.037, i.e. zero.

**So the +12.2% market alpha decomposes into: a hindsight-selected universe, beta
1.07, and a static-ish growth/tech tilt that happened to pay off in a tech bull
market. None of it is repeatable skill.**

### Where that leaves things

| Source of return | Verdict |
|---|---|
| Stock selection (within sector) | +3.3%/yr, t = 0.90 → **no** |
| Sector rotation timing | corr +0.037 → **no** |
| Static tech/growth tilt | present, but a **regime bet** |
| Hindsight-selected universe | inflates everything |
| Turnover reduction (hysteresis) | genuine operational win (49% → 22%) |

## Dynamic universe discovery

By default the screened universe is the static `universe:` list in
`config/settings.yaml` — reproducible, but hand-picked (see
[FINDINGS.md](FINDINGS.md), ISSUE-002). Set `universe_config.mode` to discover
names automatically instead:

```yaml
universe_config:
  mode: dynamic          # static | dynamic | hybrid
  max_results: 250
  discovery:
    region: us
    exchanges: [NYQ, NMS, NGM, NCM, ASE]   # NYSE + Nasdaq tiers + AMEX
    min_price: 5.0
    min_market_cap: 2000000000             # $2B
    min_avg_volume_3m: 500000              # 500k shares/day
    dedupe_companies: true                 # collapse GOOG/GOOGL
```

| Mode | Behaviour |
|---|---|
| `static` | Screen the seed list exactly (default) |
| `dynamic` | Discover from the Yahoo screener; ignore the seed list |
| `hybrid` | Discovered names unioned with the seed list |

Or override per run without touching the file:

```bash
python scripts/daily_run.py --universe-mode dynamic --max-names 100
```

Discovery is only used for the **live screen**. Discovered names always carry a
`source` and timestamp so a report can never silently pass off a discovered
universe as the configured one.

### Discovery is not free of look-ahead — so backtests use a different mechanism

The screener returns **today's** list; it has no history. Pointing it at a 10-year
backtest would be look-ahead bias. So the backtest rebuilds its universe
point-in-time from price history instead:

```yaml
universe_config:
  backtest:
    mode: point_in_time_liquidity   # or static
    pool_size: 1000                 # names carried into each rebalance
    lookback_days: 20               # trailing dollar-volume window
```

At each rebalance date the engine ranks the price pool by trailing average
dollar volume and keeps the top `pool_size`. Dollar volume as of date D is
knowable on date D, so no future information leaks in. This mirrors the guard
already used for metrics.

> **Note:** with the default 58-name seed list and `pool_size: 1000`, nothing is
> filtered — the mode only bites once the pool is larger than `pool_size`.

```bash
python scripts/backtest.py --backtest-universe point_in_time_liquidity
```

**Honest caveat.** This removes *hindsight-ranking* bias (the universe was
chosen knowing which names won), not *delisting* bias — companies that went to
zero are absent from the price pool entirely. That is ISSUE-002 in
[FINDINGS.md](FINDINGS.md).

### Missing data is dropped, never fabricated

When a ticker yields nothing, ingestion retries with progressively wider windows
(`ingest.retry_periods`, default `[2y, 5y, max]`). If every retry fails the
ticker is **dropped** and recorded in the run's error notes — the screen never
mixes fabricated prices into real results. Setting `ingest.on_missing: synthetic`
restores the old fabricate-as-fallback behaviour, and is intended for smoke runs
only.

Prices for a large universe are downloaded in chunks (`ingest.batch_size`, with a
short pause between) because a single `yf.download` across hundreds of tickers
fails partially and silently.

## Configuration

Everything lives in `config/settings.yaml`:

- **universe** — seed tickers to screen
- **universe_config** — static/dynamic/hybrid mode, discovery gates, backtest universe
- **ingest** — retry ladder, missing-data policy, batch size
- **screening** — lookback, shortlist size, liquidity/price filters
- **scoring_weights** — factor weights (percentile-ranked, sum to 1.0)
- **risk** — account size, % risked per trade, ATR stop multiple, conviction floor
- **agent** — `rules` or `llm`, model, temperature, horizon

### Trend score factors

| Factor | Signal |
|---|---|
| `momentum_3m` / `momentum_6m` | price return |
| `relative_strength_3m` | excess return vs benchmark |
| `trend_quality` | close>SMA20, SMA20>SMA50, SMA50>SMA200, rising SMA50, positive MACD |
| `volume_surge` | 20d vs 60d volume |
| `proximity_52w_high` | distance from 52-week high |
| `risk_adjusted_momentum` | 3m return / annualized vol |

## Layout

```
core/            pure pipeline library (no web, no LLM required)
  ingest/        yfinance + cache + offline synthetic generator
  features/      indicators + point-in-time metrics + trend scoring
  agent/         tools, deterministic analyst, optional LLM refiner
  risk/          position sizing + validator
  backtest/      engine + panel evaluator + robustness + attribution + sectors
  report/        markdown rendering
  pipeline.py    orchestration
api/             FastAPI shell
web/             Streamlit: app.py + pages/{1_Backtest,2_Robustness,3_Attribution}.py
scripts/         CLI (daily_run.py, backtest.py, robustness.py, attribution.py)
config/          settings.yaml
tests/           pytest suite (42 tests)
```

## Roadmap / next steps

- [x] Walk-forward backtest of the *screen* with controls (`core/backtest/`)
- [x] Robustness gate: bootstrap, parameter plateau, cost sensitivity
- [x] Factor attribution: Jensen's alpha, sector-neutral benchmark, turnover variants
- [x] **Sector-neutral scoring** — ranks within sector, fixes sector quotas. **No variant clears t = 2 → no demonstrated within-sector edge**
- [x] **Rotation vs static tilt diagnostic** — sector weights are dynamic but not predictive (+0.037)
- [x] **P0c: verdict re-tested on a discovered 248-name universe** — edge vs universe +241% → +36%, IC +0.025 → −0.005, market alpha t 2.59 → 1.41. **No edge; robust to universe construction** (FINDINGS.md F6)
- [ ] **ISSUE-009: normalise the sector taxonomy** (GICS vs Yahoo labels) before trusting any sector-neutral result on a dynamic universe
- [ ] Decide: keep the harness as research infrastructure, or retire the strategy
- [ ] Sector-momentum ETF benchmark — only worth it if a predictive sector signal is ever found
- [ ] **Point-in-time index membership** — the remaining source of inflation, though
      the skill tests have already failed
- [ ] Fundamentals & earnings-calendar ingest (avoid holding through earnings by default)
- [ ] Persist runs to SQLite/Postgres for audit + agent-decision backtesting
- [ ] Alerts (email/Slack) on new top-ranked names
- [ ] Sector/industry neutralization so the shortlist isn't one crowded theme
- [ ] Slippage/commission model in sizing
