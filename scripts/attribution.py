#!/usr/bin/env python3
"""Factor attribution: is the edge skill, or beta and sector exposure?

Usage:
    python scripts/attribution.py --synthetic
    python scripts/attribution.py --lookback 10y --top 10
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.backtest.attribution import run_attribution_suite  # noqa: E402
from core.backtest.engine import BacktestConfig  # noqa: E402
from core.config import REPORT_DIR, load_settings  # noqa: E402
from core.universe import apply_universe_overrides  # noqa: E402


def _cli_progress(message: str, pct: float) -> None:
    bar = int(pct * 30)
    sys.stdout.write(f"\r[{'#' * bar}{'.' * (30 - bar)}] {pct * 100:5.1f}%  {message[:56]:<56}")
    sys.stdout.flush()
    if pct >= 1.0:
        sys.stdout.write("\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Factor attribution for the trend screen.")
    ap.add_argument("--settings", default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--lookback", default="10y")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--costs", type=float, nargs="*", default=[10.0, 30.0, 50.0])
    ap.add_argument("--rebalances", type=int, nargs="*", default=[21, 63])
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument(
        "--universe-mode",
        choices=["static", "dynamic", "hybrid"],
        default=None,
        help="override universe_config.mode",
    )
    ap.add_argument("--max-names", type=int, default=None, help="cap on discovered names")
    ap.add_argument("--json", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING)
    settings = load_settings(args.settings)
    settings = apply_universe_overrides(
        settings, mode=args.universe_mode, max_results=args.max_names
    )
    config = BacktestConfig(
        start=args.start, end=args.end, lookback=args.lookback,
        top_n=args.top, cost_bps=args.cost_bps,
    )

    result = run_attribution_suite(
        settings=settings, config=config, synthetic=args.synthetic,
        use_cache=not args.no_cache, cost_grid=args.costs,
        rebalance_list=args.rebalances,
        progress=None if args.quiet else _cli_progress,
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    md_path = REPORT_DIR / f"attribution-{result['run_id']}.md"
    json_path = REPORT_DIR / f"attribution-{result['run_id']}.json"
    md_path.write_text(result["markdown"], encoding="utf-8")
    payload = {k: v for k, v in result.items() if k != "markdown"}
    json_path.write_text(json.dumps(payload, default=str, indent=2), encoding="utf-8")
    (REPORT_DIR / "attribution-latest.md").write_text(result["markdown"], encoding="utf-8")
    (REPORT_DIR / "attribution-latest.json").write_text(json.dumps(payload, default=str, indent=2), encoding="utf-8")
    if args.json:
        Path(args.json).write_text(json.dumps(payload, default=str, indent=2), encoding="utf-8")

    m = result["market"]
    sec = result["sector_neutral"]
    v = result["verdict"]

    print(f"\nReport: {md_path}")
    print(
        f"\n  Market alpha: {((m.get('alpha_annualized') or 0) * 100):+.1f}% ann.  "
        f"t={m.get('alpha_t')}  beta={m.get('beta')}  R²={m.get('r_squared')}"
    )
    print(
        f"  Sector-neutral excess: {((sec.get('mean_annualized') or 0) * 100):+.1f}% ann.  "
        f"t={sec.get('t_stat')}"
    )
    print(f"\n  VERDICT: {v['overall']} ({v['n_passed']}/3)")
    for c in v["checks"]:
        print(f"    [{'PASS' if c['passed'] else 'FAIL'}] {c['name']}: {c['detail']}")

    variants = result.get("turnover_variants", [])
    if variants:
        print("\n  Variants (edge vs universe by cost; sector-neutral = stock-selection alpha):")
        print(f"    {'reb':>4} {'scoring':<15} {'selection':<12} {'turn':>5}  "
              f"{'sec-neutral':>12}  edges")
        for row in variants:
            ego = row.get("edge_by_cost") or {}
            cells = "  ".join(f"{k}bps {((ego.get(k) or 0) * 100):+.0f}%" for k in sorted(ego, key=int))
            scoring = row.get("scoring", "pooled")
            sec = row.get("sector_neutral_annualized")
            sec_t = row.get("sector_neutral_t")
            sec_str = f"{(sec or 0) * 100:+.1f}% (t={sec_t:.2f})" if sec_t is not None else "n/a"
            print(
                f"    {row['rebalance_days']:>3}d {scoring:<15} {row['selection']:<12} "
                f"{(row.get('avg_turnover') or 0) * 100:>4.0f}%  {sec_str:>18}  {cells}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
