#!/usr/bin/env python3
"""Daily screening run.

Usage:
    python scripts/daily_run.py                     # live data, rules analyst
    python scripts/daily_run.py --synthetic         # offline demo data
    python scripts/daily_run.py --provider llm      # requires OPENAI_API_KEY
    python scripts/daily_run.py --top 20 --json out.json
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import load_settings  # noqa: E402
from core.pipeline import run_pipeline, save_result  # noqa: E402
from core.universe import apply_universe_overrides  # noqa: E402


def _cli_progress(message: str, pct: float) -> None:
    bar = int(pct * 30)
    sys.stdout.write(f"\r[{'#' * bar}{'.' * (30 - bar)}] {pct * 100:5.1f}%  {message[:60]:<60}")
    sys.stdout.flush()
    if pct >= 1.0:
        sys.stdout.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the stock screening pipeline.")
    parser.add_argument("--settings", default=None, help="path to settings.yaml")
    parser.add_argument("--synthetic", action="store_true", help="use offline generated data")
    parser.add_argument("--provider", choices=["rules", "llm"], default=None)
    parser.add_argument("--top", type=int, default=None, help="number of candidates")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--universe-mode",
        choices=["static", "dynamic", "hybrid"],
        default=None,
        help="override universe_config.mode",
    )
    parser.add_argument("--max-names", type=int, default=None, help="cap on discovered names")
    parser.add_argument("--json", default=None, help="also write JSON to this path")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    settings = load_settings(args.settings)
    settings = apply_universe_overrides(
        settings, mode=args.universe_mode, max_results=args.max_names
    )
    progress = None if args.quiet else _cli_progress

    result = run_pipeline(
        settings=settings,
        synthetic=args.synthetic,
        provider=args.provider,
        top_n=args.top,
        use_cache=not args.no_cache,
        progress=progress,
    )
    json_path, md_path = save_result(result)

    if args.json:
        Path(args.json).write_text(__import__("json").dumps(result, default=str, indent=2), encoding="utf-8")

    if not args.quiet:
        print(f"\nReport: {md_path}")
        print(f"JSON:   {json_path}")

    passed = [c for c in result["candidates"] if c["validation"]["passed"]]
    print(f"\n{len(passed)}/{len(result['candidates'])} candidates passed validation")
    for c in result["candidates"][:10]:
        flag = "OK " if c["validation"]["passed"] else "XX "
        name = (c.get("name") or "")[:28]
        print(
            f"  {flag} {c['ticker']:<6} {name:<28} score {c['trend_score']:>5.1f}  "
            f"conv {c['conviction']:.2f}  {c['direction']:<5}  {c['position'].get('shares', 0):>5} sh"
        )
    if result["errors"]:
        print(f"\n{len(result['errors'])} data note(s); first few:")
        for err in result["errors"][:5]:
            print(f"  - {err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
