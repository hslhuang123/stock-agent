"""Deterministic risk rules: position sizing and thesis validation."""
from __future__ import annotations

from typing import Any


def position_size(
    account_size: float,
    risk_per_trade_pct: float,
    entry: float,
    stop: float,
    max_position_pct: float = 100.0,
) -> dict[str, Any]:
    """Fixed-fractional sizing: risk a fixed % of the account per trade."""
    if not entry or not stop or entry <= stop:
        return {"shares": 0, "dollar_risk": 0.0, "position_value": 0.0, "note": "invalid entry/stop"}

    risk_dollars = account_size * (risk_per_trade_pct / 100.0)
    per_share_risk = entry - stop
    shares = int(risk_dollars // per_share_risk)
    position_value = shares * entry
    cap = account_size * (max_position_pct / 100.0)
    capped = False
    if position_value > cap:
        shares = int(cap // entry)
        position_value = shares * entry
        capped = True

    return {
        "shares": shares,
        "dollar_risk": round(shares * per_share_risk, 2),
        "position_value": round(position_value, 2),
        "position_pct": round(100 * position_value / account_size, 2) if account_size else 0.0,
        "stop": round(stop, 2),
        "capped_by_max_position": capped,
    }


def build_position(row: Any, risk: dict[str, Any]) -> dict[str, Any]:
    """Derive entry/stop from ATR and size the position."""
    entry = float(row["close"])
    atr = float(row.get("atr14") or 0.0)
    stop = entry - atr * float(risk.get("atr_stop_multiple", 2.0))
    stop = max(stop, 0.01)
    plan = position_size(
        account_size=float(risk.get("account_size", 100_000)),
        risk_per_trade_pct=float(risk.get("risk_per_trade_pct", 1.0)),
        entry=entry,
        stop=stop,
        max_position_pct=float(risk.get("max_position_pct", 100.0)),
    )
    plan["atr14"] = round(atr, 2)
    plan["target_1"] = round(entry + atr * 3.0, 2)
    plan["target_2"] = round(entry + atr * 5.0, 2)
    plan["risk_reward_1"] = round((plan["target_1"] - entry) / (entry - stop), 2) if entry > stop else None
    return plan


def validate_thesis(thesis: dict[str, Any], risk: dict[str, Any]) -> tuple[bool, list[str]]:
    """Apply hard vetoes. Returns (passed, reasons)."""
    reasons: list[str] = []
    conviction = float(thesis.get("conviction", 0.0))
    min_conviction = float(risk.get("min_conviction", 0.0))

    if conviction < min_conviction:
        reasons.append(f"conviction {conviction:.2f} below minimum {min_conviction:.2f}")

    rsi = thesis.get("technicals", {}).get("rsi14")
    if rsi is not None and float(rsi) > 82:
        reasons.append(f"RSI {rsi:.0f} severely overbought")

    ext = thesis.get("technicals", {}).get("ext_from_sma20")
    if ext is not None and float(ext) > 0.25:
        reasons.append(f"extended {float(ext) * 100:.0f}% above SMA20 — poor entry")

    position = thesis.get("position", {})
    if position.get("shares", 0) <= 0:
        reasons.append("position size resolves to zero shares")

    return (len(reasons) == 0), reasons
