#!/usr/bin/env python3
"""Walk-forward backtest of the trend screen.

Usage:
    python scripts/backtest.py --synthetic
    python scripts/backtest.py --lookback 10y --rebalance 21 --hold 21 --top 10
    python scripts/backtest.py --start 2018-01-01 --end 2024-12-31 --json bt.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.backtest.engine import BacktestConfig, run_backtest  # noqa: E402
from core.config import REPORT_DIR, load_settings  # noqa: E402
from core.universe import apply_universe_overrides  # noqa: E402


def _cli_progress(message: str, pct: float) -> None:
    bar = int(pct * 30)
    sys.stdout.write(f"\r[{'#' * bar}{'.' * (30 - bar)}] {pct * 100:5.1f}%  {message[:58]:<58}")
    sys.stdout.flush()
    if pct >= 1.0:
        sys.stdout.write("\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Walk-forward backtest of the trend screen.")
    ap.add_argument("--settings", default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--lookback", default="10y")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--rebalance", type=int, default=21, help="trading days between rebalances")
    ap.add_argument("--hold", type=int, default=21, help="holding period in trading days")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--cost-bps", type=float, default=10.0, help="one-way commission+slippage")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--weight-mode", choices=["equal", "score"], default="equal")
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
        rebalance_days=args.rebalance,
        holding_days=args.hold,
        top_n=args.top,
        cost_bps=args.cost_bps,
        folds=args.folds,
        weight_mode=args.weight_mode,
    )

    result = run_backtest(
        settings=settings,
        config=config,
        synthetic=args.synthetic,
        use_cache=not args.no_cache,
        progress=None if args.quiet else _cli_progress,
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    md_path = REPORT_DIR / f"backtest-{result['run_id']}.md"
    json_path = REPORT_DIR / f"backtest-{result['run_id']}.json"
    md_path.write_text(result["markdown"], encoding="utf-8")
    payload = {k: v for k, v in result.items() if k != "markdown"}
    json_path.write_text(json.dumps(payload, default=str, indent=2), encoding="utf-8")
    (REPORT_DIR / "backtest-latest.md").write_text(result["markdown"], encoding="utf-8")
    (REPORT_DIR / "backtest-latest.json").write_text(json.dumps(payload, default=str, indent=2), encoding="utf-8")
    if args.json:
        Path(args.json).write_text(json.dumps(payload, default=str, indent=2), encoding="utf-8")

    net = result["summary"]["net"]
    bench = result["summary"]["benchmark"]
    sig = result["signal_quality"]
    print(f"\nReport: {md_path}")
    print(
        f"\n  net total {net.get('total_return', 0) * 100:+.1f}%  "
        f"benchmark {bench.get('total_return', 0) * 100:+.1f}%  "
        f"alpha {(net.get('total_return', 0) - bench.get('total_return', 0)) * 100:+.1f}%"
    )
    print(
        f"  CAGR {net.get('cagr', 0) * 100:+.1f}%  Sharpe {net.get('sharpe')}  "
        f"maxDD {net.get('max_drawdown', 0) * 100:.1f}%  win-vs-bench "
        f"{(net.get('win_rate_vs_bench') or 0) * 100:.0f}%"
    )
    print(
        f"  IC mean {sig.get('ic_mean')}  top-minus-bottom {sig.get('top_minus_bottom_spread')}"
    )
    rnd = result["summary"].get("random", {})
    uni = result["summary"].get("universe", {})
    print(
        f"  controls: random {rnd.get('total_return', 0) * 100:+.1f}% | "
        f"universe EW {uni.get('total_return', 0) * 100:+.1f}% | "
        f"edge vs universe {(net.get('total_return', 0) - uni.get('total_return', 0)) * 100:+.1f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
