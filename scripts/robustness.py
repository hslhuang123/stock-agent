#!/usr/bin/env python3
"""Robustness gate: bootstrap, parameter plateau, cost sensitivity.

Usage:
    python scripts/robustness.py --synthetic
    python scripts/robustness.py --trials 50 --drop 0.3
    python scripts/robustness.py --lookback 10y --top 10 --json rb.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.backtest.engine import BacktestConfig  # noqa: E402
from core.backtest.robustness import run_robustness_suite  # noqa: E402
from core.config import REPORT_DIR, load_settings  # noqa: E402
from core.universe import apply_universe_overrides  # noqa: E402


def _cli_progress(message: str, pct: float) -> None:
    bar = int(pct * 30)
    sys.stdout.write(f"\r[{'#' * bar}{'.' * (30 - bar)}] {pct * 100:5.1f}%  {message[:56]:<56}")
    sys.stdout.flush()
    if pct >= 1.0:
        sys.stdout.write("\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Robustness gate for the trend screen.")
    ap.add_argument("--settings", default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--lookback", default="10y")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--trials", type=int, default=30, help="bootstrap trials")
    ap.add_argument("--drop", type=float, default=0.30, help="fraction of universe dropped per trial")
    ap.add_argument("--rebalances", type=int, nargs="*", default=[5, 21, 63])
    ap.add_argument("--positions", type=int, nargs="*", default=[5, 10, 20, 30])
    ap.add_argument("--costs", type=float, nargs="*", default=[0.0, 10.0, 30.0, 50.0])
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument(
        "--universe-mode",
        choices=["static", "dynamic", "hybrid"],
        default=None,
        help="override universe_config.mode",
    )
    ap.add_argument("--max-names", type=int, default=None, help="cap on discovered names")
    ap.add_argument(
        "--backtest-universe",
        choices=["static", "point_in_time_liquidity"],
        default=None,
        help="how the backtest rebuilds its universe (point_in_time_liquidity = no look-ahead)",
    )
    ap.add_argument("--json", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING)
    settings = load_settings(args.settings)
    settings = apply_universe_overrides(
        settings,
        mode=args.universe_mode,
        max_results=args.max_names,
        backtest_mode=args.backtest_universe,
    )
    config = BacktestConfig(
        start=args.start,
        end=args.end,
        lookback=args.lookback,
        top_n=args.top,
        cost_bps=args.cost_bps,
    )

    result = run_robustness_suite(
        settings=settings,
        config=config,
        synthetic=args.synthetic,
        use_cache=not args.no_cache,
        bootstrap_trials=args.trials,
        drop_frac=args.drop,
        top_n_grid=args.positions,
        rebalance_grid=args.rebalances,
        cost_grid=args.costs,
        progress=None if args.quiet else _cli_progress,
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    md_path = REPORT_DIR / f"robustness-{result['run_id']}.md"
    json_path = REPORT_DIR / f"robustness-{result['run_id']}.json"
    md_path.write_text(result["markdown"], encoding="utf-8")
    payload = {k: v for k, v in result.items() if k != "markdown"}
    json_path.write_text(json.dumps(payload, default=str, indent=2), encoding="utf-8")
    (REPORT_DIR / "robustness-latest.md").write_text(result["markdown"], encoding="utf-8")
    (REPORT_DIR / "robustness-latest.json").write_text(json.dumps(payload, default=str, indent=2), encoding="utf-8")
    if args.json:
        Path(args.json).write_text(json.dumps(payload, default=str, indent=2), encoding="utf-8")

    verdict = result["verdict"]
    bs = result["bootstrap"]
    sweep = result["sweep"]
    costs = result["costs"]
    base = result["baseline"]

    print(f"\nReport: {md_path}")
    print(f"\n  Baseline edge vs universe: {(base.get('edge_vs_universe') or 0) * 100:+.1f}%")
    print(f"  Bootstrap: {(bs.get('edge_positive_share') or 0) * 100:.0f}% of runs kept positive edge "
          f"(median {(bs.get('edge_median') or 0) * 100:+.1f}%)")
    print(f"  Plateau:   {(sweep.get('edge_positive_share') or 0) * 100:.0f}% of configs positive, "
          f"spread {(sweep.get('edge_spread') or 0) * 100:.0f}pp")
    print(f"  Cost:      edge at {(costs.get('max_cost_bps') or 0):.0f} bps = "
          f"{(costs.get('edge_at_max_cost') or 0) * 100:+.1f}%")
    print(f"\n  VERDICT: {verdict['overall']} ({verdict['n_passed']}/3)")
    for c in verdict["checks"]:
        print(f"    [{'PASS' if c['passed'] else 'FAIL'}] {c['name']}: {c['detail']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
