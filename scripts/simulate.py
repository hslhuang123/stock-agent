#!/usr/bin/env python3
"""Same-day round-trip simulator: buy the top picks at the open, sell at the close.

Usage:
    python scripts/simulate.py --synthetic
    python scripts/simulate.py --lookback 1y --picks 2 --shares 100 --cost-bps 10
    python scripts/simulate.py --start 2025-01-01 --end 2025-12-31
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.backtest.daily_sim import DailySimConfig, run_daily_sim  # noqa: E402
from core.config import REPORT_DIR, load_settings  # noqa: E402
from core.universe import apply_universe_overrides  # noqa: E402


def _cli_progress(message: str, pct: float) -> None:
    bar = int(pct * 30)
    sys.stdout.write(f"\r[{'#' * bar}{'.' * (30 - bar)}] {pct * 100:5.1f}%  {message[:58]:<58}")
    sys.stdout.flush()
    if pct >= 1.0:
        sys.stdout.write("\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Same-day round-trip simulator.")
    ap.add_argument("--settings", default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--lookback", default="3y")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--picks", type=int, default=2, help="names bought each day")
    ap.add_argument("--shares", type=int, default=100, help="shares bought per pick")
    ap.add_argument("--cost-bps", type=float, default=10.0, help="one-way commission+slippage")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--universe-mode", choices=["static", "dynamic", "hybrid"], default=None)
    ap.add_argument("--max-names", type=int, default=None, help="cap on discovered names")
    ap.add_argument("--json", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING)

    settings = load_settings(args.settings)
    settings = apply_universe_overrides(
        settings, mode=args.universe_mode, max_results=args.max_names
    )
    config = DailySimConfig(
        start=args.start,
        end=args.end,
        lookback=args.lookback,
        picks=args.picks,
        shares=args.shares,
        cost_bps=args.cost_bps,
    )

    result = run_daily_sim(
        settings=settings,
        config=config,
        synthetic=args.synthetic,
        use_cache=not args.no_cache,
        progress=None if args.quiet else _cli_progress,
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    md_path = REPORT_DIR / f"daily-sim-{result['run_id']}.md"
    json_path = REPORT_DIR / f"daily-sim-{result['run_id']}.json"
    md_path.write_text(result["markdown"], encoding="utf-8")
    payload = {k: v for k, v in result.items() if k != "markdown"}
    json_path.write_text(json.dumps(payload, default=str, indent=2), encoding="utf-8")
    (REPORT_DIR / "daily-sim-latest.md").write_text(result["markdown"], encoding="utf-8")
    (REPORT_DIR / "daily-sim-latest.json").write_text(
        json.dumps(payload, default=str, indent=2), encoding="utf-8"
    )
    if args.json:
        Path(args.json).write_text(json.dumps(payload, default=str, indent=2), encoding="utf-8")

    s = result["summary"]
    print(f"\nReport: {md_path}")
    if not s.get("traded_days"):
        print("  no tradable days in range")
        return 0
    print(
        f"\n  net P&L {s['total_net']:+,.2f} USD over {s['traded_days']} days  "
        f"({s['win_days']} win / {s['loss_days']} loss, win rate "
        f"{(s.get('win_rate') or 0) * 100:.1f}%)"
    )
    print(
        f"  gross {s['total_gross']:+,.2f}  costs -{s['total_cost']:,.2f}  "
        f"avg/day {s['avg_net_per_day']:+,.2f}  "
        f"best {s['best_day']['net']:+,.2f}  worst {s['worst_day']['net']:+,.2f}"
    )
    print(f"  avg capital deployed ${s['avg_capital']:,.0f}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
