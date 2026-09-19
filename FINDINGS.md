# Findings & Open Issues

Handoff notes for the Stock Trend Agent. Captures **what the research established**
and **what is still broken**, so work can resume without re-deriving context.

- **Last updated:** 2026-09-12 (P0c: verdict re-tested on a discovered 248-name universe)
- **Repo:** `~/stock-agent` · **Env:** `.venv` (Python 3.11) · **Tests:** 87 passing
- **Status:** research-complete on the original hypothesis; strategy **not** cleared for
  capital. P0c re-confirmed the verdict on a discovered universe — see **F6**.

---

# Part 1 — Research findings

The strategy was tested four independent ways. Every one falsified the original
thesis ("the trend score picks winners"). The *harness* is the durable asset.

## Summary

| Question | Method | Result | Verdict |
|---|---|---|---|
| Does the screen beat buy-and-hold? | Walk-forward backtest vs SPY | +1016% vs +277% (10y) | Looks good — but see below |
| Is it survivorship-inflated? | Equal-weight universe control | Universe EW = +775%, **strategy edge vs universe only +241%** | Most "alpha" was the universe |
| Is it tradable after costs? | Cost sensitivity 0→50 bps | +241% @10bps → +22% @30bps → **−155% @50bps** | **Fragile** |
| Is the edge broad or a fluke? | Bootstrap, 30 random 70% subsets | **100%** kept positive edge, median +375% | **Broad — genuinely real** |
| Is it stock selection? | Sector-neutral benchmark | +3.3%/yr, **t = 0.90** | **No** |
| Is it sector rotation skill? | weight vs same-period sector return | corr **+0.037** | **No** |
| Can sector-neutral scoring rescue it? | 12 variants, rank within sector | best +8.2%/yr, **t = 1.63**, 0/12 clear t=2 | **No** |
| Does the verdict survive a discovered universe? | Re-run all gates on a 248-name screener pool (P0c) | Edge vs universe +241%→**+36%**; IC +0.025→**−0.005**; market alpha t 2.59→**1.41** | **No edge — robust to universe construction** |

## F1 — The screen has no demonstrable skill

Consistent across all tests: **no source of repeatable edge.** Every candidate
explanation was tested and eliminated.

- Stock selection (within sector): +3.3%/yr, t = 0.90 → not significant
- Sector rotation timing: corr +0.037 → no predictive power
- The apparent +12.2%/yr Jensen's alpha vs SPY decomposes into: beta 1.07 + a
  static growth/tech tilt + a hindsight-selected universe

## F2 — The universe is the largest source of return *and* the largest source of bias

The universe was originally a **hardcoded list of 58 tickers** in
`config/settings.yaml`, hand-picked from today's mega-caps. Equal-weighting it
returned +775% vs SPY's +277%. Most of the headline alpha is universe selection,
chosen with knowledge of which names won — a form of survivorship/hindsight bias.

Dynamic discovery now exists (`universe_config.mode: dynamic`, see ISSUE-002). It
addresses the hindsight-*ranking* half of this bias; the delisting
survivorship half remains open. The numbers above were produced with the static
universe and have **not** been re-measured on a discovered pool.

## F3 — Turnover was the binding constraint, and hysteresis fixes it

Momentum is turnover-heavy. `hysteresis` selection (hold until rank falls below
`top_n × exit_multiple`) is a genuine operational win:

| Variant | Turnover | Edge @50bps | Alpha (ann.) |
|---|---|---|---|
| 21d top-N | 49% | **−155%** | +12.2% |
| 21d hysteresis ×1.5 | 34% | +375% | +15.9% |
| 21d hysteresis ×2.5 | **22%** | +488% | +15.5% |

Turnover halves and the strategy becomes cost-viable at 50 bps. This is real and
worth keeping in any future iteration.

## F4 — The sector exposure is dynamic but not predictive

IT weight: mean 40.2%, **range 0%–100%**, std 21.1%. The book moves a lot — it
just doesn't move *usefully*. Correlation with subsequent sector returns: **+0.037**.

So it is a **static-ish regime tilt**, not rotation skill. Not repeatable.

## F5 — Reporting convention (learned the hard way)

Cumulative percentage-point gaps between large returns are misleading. "+488% vs
−155%" is a ~1.5× ratio, not a 643pp win. Always compare against the
**equal-weight universe** (not just SPY), report the **sector-neutral** excess, and
quote **t-statistics** rather than point estimates.

## F6 — The verdict survives a discovered universe (P0c, 2026-09-12)

F1 was measured on the hand-picked 58-name universe. P0c re-ran all three gates on
a **screener-discovered 248-name pool** (Yahoo equity screener: 1721 names matched
the gates, 250 returned — no pagination), same 10y window, top 10, 21d rebalance.

| Metric | Static 58 (hindsight) | **Discovered 248** |
|---|---|---|
| Strategy net (10y) | +1016% | **+563%** |
| Universe equal-weight | +775% | +526% |
| **Edge vs universe** | +241% | **+36%** |
| Random N-name basket | +807% | +495% |
| Strategy Sharpe | 1.16 | **0.90** |
| Random / universe Sharpe | 1.22 / 1.22 | **1.13 / 1.17** |
| IC mean | +0.025 | **−0.005** |
| Q5−Q1 spread | +0.3% | **−0.1%** (Q1 highest) |
| Folds with positive alpha | 3/4 | 3/4 (one −22%) |
| Bootstrap positive share / median edge | 100% / +375% | **83% / +146%** |
| Plateau positive / spread | 92% / 1352pp | 100% / 610pp |
| Edge @ 50 bps | −155% | **−286%** |
| Market alpha (Jensen) | +12.2%, **t=2.59** | +9.5%, **t=1.41** |
| Sector-neutral excess | +3.3%, t=0.90 | +5.8%, t=1.18 |
| Rotation corr (weight vs return) | +0.037 | −0.012 |
| IT weight (mean) | 40.2% | 17.4% (+14.4% “Technology”) |

**Every headline metric weakens or loses significance.** Specifically:

- The market alpha is **no longer significant** (t 2.59 → 1.41), confirming much of
  the static-universe alpha was the universe, not the signal.
- The information coefficient goes **negative** and the quintile ordering is
  flat/inverted — the score does not rank forward returns.
- The strategy's Sharpe (0.90) is **below both the random basket (1.13) and the
  equal-weight universe (1.17)**: it adds volatility, not return.
- The edge survives costs only in the same fragile way (gone by 30–50 bps).

One variant cleared t=2 (21d / pooled / hysteresis ×2.5 → +11.2%, t=2.05), but it
is **1 of 12** tested variants (multiple testing), was not significant on the base
configuration (t=1.18), and was computed with a **fractured sector taxonomy** (see
ISSUE-009). It is not credible evidence of skill.

**Conclusion: F1 is robust to universe construction.** Removing the hindsight
*ranking* bias does not rescue the signal — it removes most of what looked like
edge. The remaining bias (delisting survivorship, ISSUE-002.1) could only make the
absolute numbers worse.

## Reproducing the findings

```bash
cd ~/stock-agent && source .venv/bin/activate

# Static (hindsight-selected) universe
python scripts/backtest.py    --lookback 10y --quiet   # headline + controls
python scripts/robustness.py  --lookback 10y --trials 30 --quiet   # bootstrap/plateau/cost
python scripts/attribution.py --lookback 10y --quiet   # alpha, sector-neutral, rotation

# P0c: discovered 248-name universe (10y price cache makes reruns fast)
python scripts/backtest.py    --universe-mode dynamic --max-names 250 \
    --backtest-universe point_in_time_liquidity --lookback 10y --quiet
python scripts/robustness.py  --universe-mode dynamic --max-names 250 \
    --backtest-universe point_in_time_liquidity --lookback 10y --trials 30 --quiet
python scripts/attribution.py --universe-mode dynamic --max-names 250 --lookback 10y --quiet

# reports land in reports/*-latest.md
```

Committed outputs: `reports/backtest-latest.md`, `robustness-latest.md`,
`attribution-latest.md` (currently the **discovered-248** P0c runs). The static-58
baselines are the timestamped files under `reports/backtest-20260911T050422Z.*`,
`robustness-20260911T052428Z.*`, `attribution-20260911T062220Z.*`.

---

# Part 2 — Open issues

## ISSUE-001 — Benchmark is treated as a tradeable candidate ✅ FIXED

**Severity:** high (correctness / invariant violation) · **Effort:** ~15 min

### Resolution (2026-09-11)

`core/pipeline.py` now builds a separate tradeable view so the benchmark stays
available for regime lookups but can never be selected:

```python
tradeable = ranked.drop(index=benchmark, errors="ignore")   # selection + reporting
shortlist = tradeable.head(n)
ctx = MarketContext(prices=prices, features=ranked, benchmark=benchmark)  # regime intact
```

`universe_size` now reports `len(tradeable)`, so the live screen shows **58**
rather than 59. Guarded by
`tests/test_pipeline.py::test_benchmark_is_never_a_candidate`, which asserts the
benchmark is absent from the shortlist and candidates, that `universe_size`
equals the configured size, and that `regime.available` is still `True`.

`resolve_universe` additionally strips the benchmark centrally, so the live
pipeline, the engine and the panel cannot drift apart again.

### Original report

**Severity:** high (correctness / invariant violation) · **Effort:** ~15 min

### Symptom

`config/settings.yaml` defines **58** tickers, but reports say `universe_size: 59`:

```bash
python -c "
from core.config import load_settings
from core.ingest.prices import fetch_prices
from core.features.indicators import add_indicators
from core.features.trending import compute_feature_table, rank_trending
s = load_settings(); bench = s['benchmark']
prices, _ = fetch_prices(list(dict.fromkeys(s['universe'] + [bench])), period='1y', synthetic=True)
ranked = rank_trending(compute_feature_table(prices, benchmark=add_indicators(prices[bench])),
                       s['scoring_weights'], s['screening'])
print(len(s['universe']), len(ranked), bench in ranked.index)
"
# -> 58 59 True
```

### Cause

| Location | Code |
|---|---|
| `core/pipeline.py:45` | `universe = list(dict.fromkeys(settings.get("universe", [])))` — **no benchmark filter** |
| `core/pipeline.py:54` | `all_tickers = list(dict.fromkeys(universe + [benchmark]))` — benchmark added for fetch |
| `core/pipeline.py:83` | `ranked = rank_trending(...)` — ranks the benchmark along with candidates |

The benchmark is fetched (correct — needed for relative strength and regime) but
**never removed from the ranked table**, so it is eligible for the shortlist,
candidates, and a full trade thesis.

### Impact

- Low probability of actually firing: SPY sat at rank **51/59** in the sample.
- Real when it does: in a broad rally the index itself can enter the top 15 and
  receive a nonsense thesis ("BUY SPY, stop at 2×ATR").
- Inflates `universe_size` by 1.
- **Contradicts the documented invariant** `AGENT.md` §5.4 ("The benchmark is never
  a candidate"). The backtest enforces it; the live pipeline does not.

### The correct pattern already exists

```python
core/backtest/engine.py:166      universe = [t for t in dict.fromkeys(...) if t != benchmark]
core/backtest/robustness.py:69   universe = [t for t in universe if t != benchmark]
core/backtest/robustness.py:451  universe = [t for t in dict.fromkeys(...) if t != benchmark]
```

Same bug class was fixed in the backtester earlier; the identical line in
`pipeline.py` was missed.

### Proposed fix — **do not just filter `ranked`**

The benchmark must stay in `ranked` for regime reporting. Two call sites depend on it:

- `core/pipeline.py:140` — `_regime_summary`: `if benchmark not in ranked.index: return {"available": False}`
- `core/agent/tools.py:140` — `get_market_regime`: `if ctx.benchmark not in ctx.features.index: ...`

Filtering `ranked` directly would silently disable the market-regime section and
the analyst's regime sentence. The minimal correct change keeps a separate
tradeable view:

```python
# core/pipeline.py, after line 83
ranked = rank_trending(table, settings["scoring_weights"], screening)
if ranked.empty:
    raise RuntimeError("no tickers passed the liquidity filters")

# Benchmark stays in `ranked` for regime lookups, but is not tradeable.
tradeable = ranked.drop(index=benchmark, errors="ignore")

...
shortlist = tradeable.head(n)                      # was ranked.head(n)
ctx = MarketContext(prices=prices, features=ranked, benchmark=benchmark)  # unchanged
...
"universe_size": len(tradeable),                   # was len(ranked)
```

### Test to add (regression guard)

In `tests/test_pipeline.py`:

```python
def test_benchmark_is_never_a_candidate(settings):
    settings = dict(settings)
    settings["universe"] = [f"T{i:02d}" for i in range(25)] + ["SPY"]
    settings["screening"] = dict(settings["screening"], top_n=15, min_avg_dollar_volume=0)
    settings["agent"] = dict(settings["agent"], top_candidates=10)

    result = run_pipeline(settings=settings, synthetic=True, provider="rules", use_cache=False)

    assert all(c["ticker"] != "SPY" for c in result["candidates"])
    assert all(s["ticker"] != "SPY" for s in result["shortlist"])
    assert result["universe_size"] == 25          # benchmark excluded
    assert result["regime"]["available"] is True   # regime still works
```

---

## ISSUE-002 — Universe is hardcoded and hindsight-selected 🔶 PARTIALLY ADDRESSED

**Severity:** high (invalidates absolute returns) · **Effort:** days

The universe remains a hand-picked seed list by default, but dynamic discovery is
now implemented and opt-in:

```yaml
universe_config:
  mode: dynamic          # static | dynamic | hybrid (default: static)
  discovery:
    region: us
    exchanges: [NYQ, NMS, NGM, NCM, ASE]
    min_price: 5.0
    min_market_cap: 2000000000
    min_avg_volume_3m: 500000
```

or per run: `python scripts/daily_run.py --universe-mode dynamic --max-names 100`.

**What this fixes:** the universe is no longer necessarily today's hand-picked
mega-caps. Discovery is reproducible (gates are configuration), de-duplicated
across share classes, and carries provenance (`source`, `discovered_at`).

**What it does not fix — still open:**

1. **Delisting survivorship.** Companies that went to zero (SIVB, FRC, TWTR,
   ATVI…) are absent from Yahoo entirely, so any historical universe built from
   the price pool silently excludes them. This is the dominant remaining bias and
   needs an explicit delisted-ticker list.
2. **`mode: static` is still the default**, so existing results keep their
   hindsight bias until someone opts in.
3. **The seed list is unchanged** — `universe:` still holds the original 58.

Point-in-time liquidity reconstruction for backtests *is* implemented (see
`core/universe/point_in_time.py`), which removes hindsight-*ranking* bias. With
the default 58-name seed and `pool_size: 1000` nothing is filtered; it only bites
once the pool is larger than `pool_size`.

**Deprioritised:** since every skill test already failed (F1), broadening the
universe lowers the absolute numbers without changing the decision. Do the
delisted-ticker work only if a predictive signal is ever found. The recommended
next experiment — run the gate with a discovered pool (P0c) — **has now been
done** and confirmed the verdict (see F6); it lowered the numbers without changing
the decision, exactly as predicted.

---

## ISSUE-003 — Sector labels are static, not point-in-time ⚠️ OPEN — LOW

`core/backtest/sectors.py` uses a static GICS map with a yfinance fallback. GICS
reclassifications over a 10-year sample are not modelled. Minor; note it as a
caveat in attribution output (already done in `attribution_report.py`).

---

## ISSUE-004 — No earnings-calendar guard ⚠️ OPEN — MEDIUM

The screen can hold a name straight through an earnings gap, which is a large
unmodelled risk for a 21-day horizon. `config/settings.yaml` has no earnings
setting. Worth adding an "avoid new entries within N days of earnings" filter.

---

## ISSUE-005 — No earnings guard in risk sizing ⚠️ OPEN — LOW

Related to ISSUE-004: `core/risk/rules.py` sizes off 2×ATR with no event-risk
adjustment. ATR does not capture gap risk.

---

## ISSUE-006 — `--synthetic` benchmark drift is unrealistic ⚠️ OPEN — LOW

`core/ingest/prices.py::synthetic_prices` generates independent GBM per ticker, so
the synthetic "SPY" can drift −81% over the sample. This makes synthetic backtests
misleading if read literally (they are only smoke tests). Consider generating the
benchmark as a common factor and individual names as benchmark + idiosyncratic.

---

## ISSUE-007 — Cache has no eviction ⚠️ OPEN — LOW

`data/prices/` holds **121 parquet files** keyed by hash of `ticker|period|interval`
with a 6-hour TTL. Stale-but-unexpired files are never pruned, and there is no size
cap. Fine at this scale; will grow with many universes/periods.

---

## ISSUE-008 — Discovery depends on an undocumented Yahoo endpoint ⚠️ OPEN — MEDIUM

`yf.screen` / `yf.EquityQuery` are unofficial: the endpoint can change or start
rate-limiting without notice, and predefined screener names are not contractually
stable. Mitigations already in place:

- Discovery fails soft → discovery cache → static seed list, with the reason
  recorded in the run's `errors`.
- Every run reports `universe.source`, so a fallback is visible, never silent.
- `discovery.cache_minutes` (default 30) reuses a recent result instead of
  re-querying.

Also note the screener **hard-caps a single query at 250 rows** (`yf.screen` raises
`ValueError: Yahoo limits query size to 250`) while reporting a much larger
`total` — 1721 names match the default gates. Offset pagination is **not**
implemented, so a 1000-name pool cannot be discovered in one pass. A
`max_results` above 250 is now **clamped** with an explanatory note in the run's
`errors` rather than failing the whole pass, and `total_available` is reported so
the gap is visible. Add offset pagination before claiming a large pool.

---

## ISSUE-009 — Sector taxonomy is mixed GICS + Yahoo labels ⚠️ OPEN — MEDIUM

`core/backtest/sectors.py` maps the static seed universe to **GICS-style** labels
(`Information Technology`, `Financials`, `Health Care`, `Consumer Discretionary`,
`Consumer Staples`, `Materials`) but falls back to the **Yahoo `info.sector`** name
for anything else (`Technology`, `Financial Services`, `Healthcare`,
`Consumer Cyclical`, `Consumer Defensive`, `Basic Materials`). A dynamic universe
contains mostly non-seed names, so a single run ends up with both taxonomies.

**Observed** in the P0c attribution run: `Information Technology` 17.4%
**and** `Technology` 14.4%; `Financials` 1.7% **and** `Financial Services` 11.2%;
`Health Care` 3.4% **and** `Healthcare` 7.2%. The same company can therefore be
split across two buckets.

**Impact.** The sector-neutral benchmark draws each sector's random cohort from a
fragmented bucket, so the within-sector test is noisier and can be biased. The
P0c “stock selection” pass (t=2.05) should not be trusted until this is fixed.
(Same root cause as ISSUE-003 — no point-in-time, no taxonomy normalisation.)

**Fix.** Normalise both sources to one canonical sector enum (map Yahoo aliases
onto the GICS names) before attributing. Small, self-contained, and testable.

## Known limitations (not bugs — by design, document and don't "fix" casually)

- Prices from **Yahoo** via yfinance: adjusted prices get revised, delisted tickers
  are absent, large universes get rate-limited. Downloads are chunked and retried,
  but a ticker that never returns data is **dropped**, not fabricated.
- The screener returns **today's** membership only — fine for the live screen,
  unusable for a historical universe (hence point-in-time reconstruction).
- No borrow/short costs (long-only anyway).
- No liquidity/impact model beyond a flat bps assumption.
- Tests run **offline only** by design (`synthetic=True`, `allow_network=False`).
- LLM layer is optional and **narrative-only** — it must never own numbers
  (`AGENT.md` §5.2).

---

# Part 3 — Plan

## P0 — Fix ISSUE-001 ✅ DONE (2026-09-11)

Shipped as proposed: `tradeable` view in `core/pipeline.py`, benchmark excluded
centrally in `resolve_universe`, `universe_size` now 58, and
`test_benchmark_is_never_a_candidate` added. All acceptance criteria met.

## P0b — Dynamic universe discovery ✅ DONE (2026-09-11)

Implemented in `core/universe/`:

- `mode: static | dynamic | hybrid`, configurable gates (`yf.EquityQuery`)
- Screener discovery with share-class de-duplication and provenance
- On-demand **Refresh universe** button in the Streamlit sidebar
- `--universe-mode` / `--max-names` / `--backtest-universe` CLI flags
- Point-in-time liquidity universe for backtests (`core/universe/point_in_time.py`)
- Chunked price downloads, wider-window retry ladder, **drop-don't-fabricate**
  missing-data policy; `ingest.on_missing: synthetic` restores the old behaviour
- 18 new tests (82 total)

Bugs found and fixed during implementation (both were real, both now guarded):

1. A cached discovery trimmed to a smaller `max_results` was written back to disk,
   permanently shrinking the cache. Discovered while testing; guarded by
   `test_cached_discovery_respects_max_results` and
   `test_resolve_does_not_write_back_a_trimmed_cache`.
2. `use_cache=False` did not actually bypass the discovery cache, so a
   "no-cache" run still used stale membership. Now threading `use_cache` through
   `discover_universe`.

### What dynamic discovery does *not* solve

It makes the pipeline broader and more honest, **not more profitable**. Given F1,
expect it to *reduce* apparent edge. Delisting survivorship (ISSUE-002.1) remains
the dominant unresolved bias.

## P0c — Verify the verdict survives a discovered universe ✅ DONE (2026-09-12)

Re-ran all three gates on a **screener-discovered 248-name pool** (10y, top 10,
21d). **The verdict is robust**: edge vs universe +241% → +36%, IC +0.025 → −0.005,
market alpha t 2.59 → 1.41, strategy Sharpe below both random and universe. Full
table and interpretation in **F6**.

```bash
python scripts/backtest.py    --universe-mode dynamic --max-names 250 \
    --backtest-universe point_in_time_liquidity --lookback 10y --quiet
python scripts/robustness.py  --universe-mode dynamic --max-names 250 \
    --backtest-universe point_in_time_liquidity --lookback 10y --trials 30 --quiet
python scripts/attribution.py --universe-mode dynamic --max-names 250 --lookback 10y --quiet
```

Enabling P0c also exposed and fixed a harness bug: `core/backtest/attribution.py`
resolved its universe by reading `settings["universe"]` directly, silently
ignoring `universe_config` (violating AGENT.md §5.5). It now goes through
`resolve_universe`, and `scripts/attribution.py` gained
`--universe-mode` / `--max-names`. Guarded by
`tests/test_attribution.py::test_attribution_resolves_universe_not_settings_universe`.

**New caveat (ISSUE-009):** the discovered run surfaced a mixed GICS/Yahoo sector
taxonomy that fragments the sector-neutral test. Fix before trusting any
within-sector result on a dynamic universe.

**Cost note:** the 10y download for ~249 tickers took a few minutes and is now
cached under `data/prices/`, so reruns are fast.

**Reproducibility note:** P0c pinned one screener result across all three gates by
raising `universe_config.cache_minutes`, so backtest/robustness/attribution all saw
the identical 248-name list. Re-running the commands above without that pin will
re-query the screener and may return a slightly different (but equivalent) pool.

## P1 — Decide the project's future

Pick one, explicitly:

- **(a) Retire the strategy, keep the harness** as research infrastructure. Lowest
  effort, honest. The harness can kill a bad idea in an afternoon.
- **(b) Test a different factor** (mean reversion, post-earnings drift,
  quality-value). The panel/attribution/robustness code is factor-agnostic — add the
  factor to `metrics_at`, wire `WEIGHT_TO_COLUMN`, re-run the gate. ~1 day.
- **(c) Reduce to a research/dashboard tool** with the honest caveats printed in
  every report, and stop presenting it as a trading strategy.

**Recommendation: (a) or (b).**

## P2 — Small quality items (optional, low risk)

- ISSUE-004: earnings guard (meaningful if the strategy continues). The screener
  exposes `earningsTimestamp`, so discovery could feed this for free.
- ISSUE-002.1: add a delisted-ticker list to the pool (true survivorship fix).
- ISSUE-007: the discovery cache (`data/universe/discovered.json`) has no eviction
  either.
- CLI: consider a `--pool-size` flag so `pool_size` can be tuned without editing YAML.

---

# Appendix — Key locations

| Concern | File |
|---|---|
| Static universe seed | `config/settings.yaml` → `universe:` |
| Universe resolution (single entry point) | `core/universe/__init__.py::resolve_universe` |
| Screener discovery | `core/universe/discover.py` |
| Point-in-time universe | `core/universe/point_in_time.py` |
| Ingest retry/drop policy | `core/ingest/prices.py::fetch_prices` |
| Point-in-time metrics | `core/features/metrics.py` |
| Reference backtester | `core/backtest/engine.py` |
| Fast evaluator (pinned to engine) | `core/backtest/robustness.py::evaluate_panel` |
| Attribution / OLS / Newey-West | `core/backtest/attribution.py` |
| Sector-neutral scoring | `core/backtest/sector_neutral.py` |
| Rotation diagnostic | `core/backtest/rotation.py` |
| Invariants & conventions | `AGENT.md` |
| Test map | `README.md` → `## Tests` |
