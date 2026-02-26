import json
from typing import Any, Dict, List, Optional

from anthropic import Anthropic

_HISTORY_WINDOW = 4


_SYSTEM_PROMPT = """
You are the execution planner for an OLAP retail analytics assistant (Jan 2022 – Dec 2024).

Your ONLY responsibility:
Return a valid `create_execution_plan` tool call.
Never respond in plain text.

━━━━━━━━ DATA MODEL ━━━━━━━━

Dimensions:
  year, quarter, month, month_name
  region, country
  category, subcategory
  customer_segment

Measures:
  revenue, profit, cost, quantity, order_count, profit_margin

━━━━━━━━ AGENTS ━━━━━━━━

navigator:
  drill_down(dimension, current_level, current_value)
  roll_up(dimension, current_level, current_value)

cube:
  slice(dimension, value)
  dice(filters)
  pivot(row_dim, col_dim, measure)

kpi:
  yoy_growth(metric, dimension)
  mom_change(metric, year)
  top_n(dimension, metric, n, filters)
  profit_margins(dimension)

anomaly:
  run(query)

visualization:
  run(query)
  IMPORTANT: Must be the LAST step.
  Use ONLY if user explicitly asks for chart, graph, plot, or visualization.

executive_summary:
  run(query)
  IMPORTANT: Must be the LAST step.
  Use ONLY if user explicitly requests executive summary or briefing.

━━━━━━━━ ROUTING RULES ━━━━━━━━

Use the MINIMUM number of steps required.

Single metric question:
  → 1 step

Comparison / growth question:
  → 1 step (yoy_growth or mom_change)

Ranking:
  → 1 step (top_n)

Single filter:
  → slice

Multiple filters:
  → dice

Drill-down:
  → prior context step + drill_down

Complex multi-metric overview:
  → 2–3 steps maximum

Never exceed 4 steps.
Prefer kpi methods over cube for aggregate metrics.
Only add visualization or executive_summary if explicitly requested.

Return ONLY the tool call.
"""


_PLAN_TOOL = {
    "name": "create_execution_plan",
    "description": "Return ordered agent steps to answer the user query.",
    "input_schema": {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "minItems": 1,
                "maxItems": 4,
                "items": {
                    "type": "object",
                    "properties": {
                        "agent": {
                            "type": "string",
                            "enum": [
                                "navigator",
                                "cube",
                                "kpi",
                                "anomaly",
                                "visualization",
                                "executive_summary",
                            ],
                        },
                        "method": {"type": "string"},
                        "params": {"type": "object"},
                        "title": {"type": "string"},
                    },
                    "required": ["agent", "method", "params", "title"],
                },
            }
        },
        "required": ["steps"],
    },
}


class Planner:
    """
    Deterministic execution planner.
    Responsible ONLY for generating execution steps.
    """

    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)

    def _build_context(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, str]]:
        messages = []

        if conversation_history:
            history = conversation_history[-_HISTORY_WINDOW:]
            for turn in history:
                messages.append(
                    {
                        "role": "user",
                        "content": f"Previous question: {turn.get('query', '')}\nFinding: {turn.get('finding', '')}",
                    }
                )

        messages.append({"role": "user", "content": query})

        return messages

    def create_plan(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        messages = self._build_context(query, conversation_history)

        response = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=600,
            temperature=0,
            system=_SYSTEM_PROMPT,
            tools=[_PLAN_TOOL],
            tool_choice={"type": "tool", "name": "create_execution_plan"},
            messages=messages,
        )

        if not response.content:
            raise RuntimeError("Planner returned empty response.")

        tool_block = None
        for block in response.content:
            if block.type == "tool_use" and block.name == "create_execution_plan":
                tool_block = block
                break

        if tool_block is None:
            raise RuntimeError("Planner did not return a valid execution plan.")

        try:
            plan = tool_block.input
            steps = plan.get("steps", [])
        except Exception as e:
            raise RuntimeError(f"Invalid plan format: {e}")

        if not steps:
            raise RuntimeError("Planner returned an empty step list.")

        return steps
