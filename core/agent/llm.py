"""Optional LLM refinement layer.

The model receives a deterministic base thesis and may call read-only tools to
gather more context. It returns narrative plus a conviction adjustment — it can
never change prices, levels, or position sizing.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from core.agent.analyst import Thesis
from core.agent.tools import TOOL_SCHEMAS, MarketContext, dispatch

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a disciplined equity research analyst.

You are given a deterministic, pre-computed technical thesis for one stock.
Your job is NOT to invent signals or price targets. Your job is to:
  1. Use the provided tools to gather any extra context you need (levels,
     fundamentals, news, market regime).
  2. Write a concise, balanced bull case and bear case grounded ONLY in the
     data returned by the tools.
  3. Adjust conviction modestly (max +/- 0.15) based on what you find.

Hard rules:
  - Never state a price target that is not already in the provided data.
  - Never reference a fact that is not in a tool result.
  - If data is missing, say so rather than guessing.
  - Return ONLY a JSON object, no prose outside it.

JSON schema:
{
  "thesis": "2-4 sentence summary",
  "catalysts": ["..."],
  "risks": ["..."],
  "conviction_delta": -0.15..0.15,
  "direction": "long" | "watch" | "avoid",
  "evidence": ["short note of which tool result supports each key claim"]
}"""


def _tool_loop(client: Any, model: str, temperature: float, ctx: MarketContext, ticker: str, base: Thesis, max_rounds: int = 3) -> list[dict[str, Any]]:
    """Let the model call tools before producing its answer."""
    used: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Ticker: {ticker}\n"
                f"Deterministic base thesis (authoritative for all numbers):\n"
                f"{json.dumps(base.to_dict(), default=str, indent=2)}\n\n"
                "Investigate with tools as needed, then return the JSON object."
            ),
        },
    ]

    for _ in range(max_rounds):
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
        )
        msg = response.choices[0].message
        messages.append(
            {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in (msg.tool_calls or [])
                ]
                or None,
            }
        )
        if not msg.tool_calls:
            return messages

        for call in msg.tool_calls:
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = dispatch(ctx, call.function.name, args)
            used.append({"tool": call.function.name, "args": args, "result": result})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result, default=str),
                }
            )

    messages.append(
        {"role": "user", "content": "Stop calling tools. Return the JSON object now."}
    )
    return messages


def _parse_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in model output")
    return json.loads(text[start : end + 1])


def refine_with_llm(
    ctx: MarketContext,
    base: Thesis,
    settings: dict[str, Any],
    llm_cfg: dict[str, Any],
) -> Thesis:
    from openai import OpenAI

    client = OpenAI(api_key=llm_cfg["api_key"], base_url=llm_cfg.get("base_url"))
    model = llm_cfg["model"]
    temperature = float(settings.get("agent", {}).get("temperature", 0.2))

    messages = _tool_loop(client, model, temperature, ctx, base.ticker, base)

    final = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=messages + [{"role": "user", "content": "Return the JSON object now."}],
        response_format={"type": "json_object"},
    )
    payload = _parse_json(final.choices[0].message.content or "")

    # Merge: narrative from the model, numbers from the deterministic base.
    base.source = "llm"
    base.thesis = str(payload.get("thesis") or base.thesis)
    if isinstance(payload.get("catalysts"), list):
        base.catalysts = [str(x) for x in payload["catalysts"]][:8]
    if isinstance(payload.get("risks"), list):
        base.risks = [str(x) for x in payload["risks"]][:8]

    delta = payload.get("conviction_delta")
    if isinstance(delta, (int, float)):
        delta = max(-0.15, min(0.15, float(delta)))
        base.conviction = round(min(0.95, max(0.0, base.conviction + delta)), 2)

    direction = payload.get("direction")
    if direction in {"long", "watch", "avoid"}:
        base.direction = direction

    if isinstance(payload.get("evidence"), list):
        base.citations = [str(x) for x in payload["evidence"]][:10]

    return base
