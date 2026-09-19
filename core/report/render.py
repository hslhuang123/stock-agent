"""Report rendering (markdown)."""
from __future__ import annotations

from typing import Any


def _fmt_pct(value: Any, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value) * 100:+.{digits}f}%"
    except (TypeError, ValueError):
        return "n/a"


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _esc(text: Any) -> str:
    """Markdown-safe table cell text."""
    return str(text).replace("|", "\\|").strip()


def render_markdown(result: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# Stock Agent Report — {result.get('as_of', '')}")
    lines.append("")
    lines.append(
        f"*Run `{result.get('run_id', '')}` · universe {result.get('universe_size', 0)} · "
        f"benchmark {result.get('benchmark', '')} · analyst `{result.get('provider', '')}`*"
    )
    lines.append("")

    regime = result.get("regime") or {}
    if regime.get("available"):
        lines.append("## Market regime")
        lines.append("")
        lines.append(
            f"- {regime['benchmark']}: {_fmt(regime.get('close'))} · "
            f"1m {_fmt_pct(regime.get('ret_1m'))} · 3m {_fmt_pct(regime.get('ret_3m'))} · "
            f"above SMA50: {regime.get('above_sma50')} · above SMA200: {regime.get('above_sma200')}"
        )
        lines.append("")

    lines.append("## Ranked candidates")
    lines.append("")
    lines.append("| # | Ticker | Company | Price | Trend | Conv. | Dir | 3m | vs Bench | RSI | Stop | Shares |")
    lines.append("|---|--------|---------|-------|-------|-------|-----|----|----------|-----|------|--------|")
    for i, t in enumerate(result.get("candidates", []), 1):
        pos = t.get("position", {})
        tech = t.get("technicals", {})
        rel = t.get("relative_strength", {})
        name = _esc(t.get("name") or "")
        lines.append(
            f"| {i} | **{t['ticker']}** | {name} | {_fmt(t.get('price'))} | {t.get('trend_score')} | "
            f"{t.get('conviction')} | {t.get('direction')} | {_fmt_pct(tech.get('ret_3m') if tech.get('ret_3m') is not None else rel.get('ret_3m'))} | "
            f"{_fmt_pct(rel.get('rel_strength_3m'))} | {_fmt(tech.get('rsi14'), 0)} | "
            f"{_fmt(pos.get('stop'))} | {pos.get('shares', 0)} |"
        )
    lines.append("")

    for t in result.get("candidates", []):
        name = t.get("name") or ""
        heading = f"### {t['ticker']}"
        if name and name != t["ticker"]:
            heading += f" — {name}"
        heading += f" — {t.get('direction', '').upper()} (conviction {t.get('conviction')})"
        lines.append(heading)
        lines.append("")
        lines.append(t.get("thesis", ""))
        lines.append("")

        if t.get("catalysts"):
            lines.append("**Catalysts / bull case**")
            lines.extend(f"- {c}" for c in t["catalysts"])
            lines.append("")
        if t.get("risks"):
            lines.append("**Risks / bear case**")
            lines.extend(f"- {r}" for r in t["risks"])
            lines.append("")

        lv = t.get("levels", {})
        tech = t.get("technicals", {})
        pos = t.get("position", {})
        lines.append(
            f"**Levels:** support {_fmt(lv.get('support_20d'))} · "
            f"resistance {_fmt(lv.get('resistance_20d'))} · 52w high {_fmt(lv.get('high_52w'))}"
        )
        lines.append("")
        lines.append(
            f"**Plan:** entry ~{_fmt(t.get('price'))} · stop {_fmt(pos.get('stop'))} · "
            f"target1 {_fmt(pos.get('target_1'))} · target2 {_fmt(pos.get('target_2'))} · "
            f"R:R {pos.get('risk_reward_1')} · size {pos.get('shares', 0)} sh "
            f"({_fmt(pos.get('position_value'))} = {pos.get('position_pct', 0)}% of account)"
        )
        lines.append("")
        lines.append(f"**Invalidation:** {t.get('invalidation', 'n/a')}")
        lines.append("")
        validation = t.get("validation") or {}
        if validation:
            status = "PASS" if validation.get("passed") else "VETOED"
            note = "" if validation.get("passed") else f" — {'; '.join(validation.get('reasons', []))}"
            lines.append(f"**Validator:** {status}{note}")
            lines.append("")
        lines.append(f"*Source: {t.get('source')} · citations: {', '.join(t.get('citations', [])) or 'n/a'}*")
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.append("> Not financial advice. Research and educational output only.")
    return "\n".join(lines)
