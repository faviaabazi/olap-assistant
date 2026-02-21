"""
Planner — master orchestrator for the OLAP assistant.

Loads credentials, initialises all four agents, uses Claude to classify every
user query into a structured execution plan, dispatches to the correct agent
methods (chaining where needed), and always returns results formatted by the
ReportGeneratorAgent.

Usage
─────
    from orchestrator.planner import Planner

    planner = Planner()
    planner.chat("Show me year-over-year revenue growth by region")
    planner.chat("Now break that down to country level in Europe")

    # Or get the raw dict back:
    result = planner.run("Top 5 countries by profit in 2024")
"""

from __future__ import annotations

import os
import sys
import pathlib
from typing import Any

import anthropic
import duckdb
from dotenv import load_dotenv

# ── Agent imports ──────────────────────────────────────────────────────────────
# Add project root to path so imports work regardless of working directory
_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents.dimension_navigator          import DimensionNavigatorAgent
from agents.cube_operations             import CubeOperationsAgent
from agents.kpi_calculator              import KPICalculatorAgent
from agents.report_generator            import ReportGeneratorAgent
from agents.optional.anomaly_detection  import AnomalyDetectionAgent
from agents.optional.visualization      import VisualizationAgent


# ── System prompt for query classification ─────────────────────────────────────

_SYSTEM_PROMPT = """
You are the query planner for an OLAP retail sales assistant (Jan 2022 – Dec 2024).
Your only job is to call the `create_execution_plan` tool with an ordered list of
agent steps that answers the user's question.  Never reply in plain text.

━━━ DATA MODEL ━━━
Dimensions / field names (use these exact strings):
  Time      : year, quarter, month, month_name
  Geography : region, country
  Product   : category, subcategory
  Customer  : customer_segment

Sample values:
  region   : North America | Europe | Asia Pacific | Latin America
  country  : United States | Canada | Mexico | Germany | United Kingdom |
             France | Italy | Spain | Netherlands | China | Japan |
             Australia | India | South Korea | Singapore |
             Brazil | Argentina | Colombia | Chile | Peru
  category : Electronics | Furniture | Office Supplies | Clothing

━━━ AVAILABLE AGENTS & METHODS ━━━

agent: "navigator"
  drill_down(dimension, current_level, current_value)
    dimension      : "time" | "geography" | "product"
    current_level  : year|quarter|month  /  region|country  /  category|subcategory
    current_value  : the value to filter at current_level before going deeper
  roll_up(dimension, current_level, current_value)
    same signature, navigates one level up instead

agent: "cube"
  slice(dimension, value)
    dimension : any field name
    value     : the value to filter by
  dice(filters)
    filters   : dict of field->value, e.g. {"year": 2024, "region": "Europe"}
  pivot(row_dim, col_dim, measure)
    row_dim / col_dim : any field name
    measure : revenue | profit | cost | quantity | profit_margin | order_count

agent: "kpi"
  yoy_growth(metric, dimension)
    metric    : revenue | profit | cost | quantity
    dimension : overall | quarter | region | country | category | subcategory | customer_segment
  mom_change(metric, year)
    metric : revenue | profit | cost | quantity
    year   : 2022 | 2023 | 2024
  top_n(dimension, metric, n, filters)
    dimension : any field name
    metric    : revenue | profit | cost | quantity | order_count | profit_margin
    n         : integer
    filters   : optional dict of field->value
  profit_margins(dimension)
    dimension : any field name

agent: "anomaly"
  run(query)
    query : {
      "metric"     : revenue | profit | cost | quantity,
      "dimension"  : year | quarter | month | region | country |
                     category | subcategory | customer_segment,
      "sensitivity": float (optional, default 2.0 — z-score threshold)
    }
    Returns anomaly rows, normal rows, z-scores, and a Claude business interpretation.

agent: "visualization"
  run(query)
    query : {
      "context": str,  # what the data represents, e.g. "Revenue by region 2022-2024"
      "title":   str,  # chart title (optional, falls back to context)
    }
    IMPORTANT: data is injected automatically from the IMMEDIATELY PRECEDING step.
    Always place visualization as the LAST step, after a data-producing step.
    Use ONLY when the user explicitly asks for a chart, graph, plot, or visualization.

━━━ ROUTING RULES ━━━
- "drill down / go deeper / more detail"  → navigator.drill_down
- "roll up / less detail / summarise"     → navigator.roll_up
- "filter by one value"                   → cube.slice
- "filter by multiple values"             → cube.dice
- "pivot / cross-tab / matrix"            → cube.pivot
- "year over year / YoY / annual growth"  → kpi.yoy_growth
- "month over month / MoM / monthly"      → kpi.mom_change
- "top N / best / highest / ranking"      → kpi.top_n
- "margin / profitability / profit %"     → kpi.profit_margins
- "anomaly / outlier / unusual / spike / dip / abnormal" → anomaly.run
- "chart / plot / graph / visualize / show me visually" → prior data step + visualization.run
- Complex queries: use 2-3 steps combined into one report.

━━━ CHAINING EXAMPLES ━━━
"Revenue in Europe in 2024, then break it down to country"
  → cube.dice({"region":"Europe","year":2024}) + navigator.drill_down(geography, region, Europe)

"Top products in Asia Pacific with quarterly breakdown"
  → kpi.top_n(subcategory, revenue, 5, {"region":"Asia Pacific"})
    + kpi.mom_change(revenue, 2024)  [or yoy_growth]

"Full performance overview"
  → kpi.yoy_growth + kpi.profit_margins + cube.pivot
""".strip()


# ── Classification tool ────────────────────────────────────────────────────────

_PLAN_TOOL = {
    "name": "create_execution_plan",
    "description": (
        "Produce an ordered list of agent steps that together answer the user query. "
        "Use 1 step for simple queries, 2-3 for compound or follow-up queries."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "Brief (1-2 sentence) explanation of why you chose these steps.",
            },
            "steps": {
                "type": "array",
                "minItems": 1,
                "maxItems": 4,
                "items": {
                    "type": "object",
                    "properties": {
                        "agent": {
                            "type": "string",
                            "enum": ["navigator", "cube", "kpi", "anomaly", "visualization"],
                            "description": "Which agent to call.",
                        },
                        "method": {
                            "type": "string",
                            "description": "Method name on that agent.",
                        },
                        "params": {
                            "type": "object",
                            "description": "Keyword arguments for the method.",
                        },
                        "title": {
                            "type": "string",
                            "description": "Human-readable section title for this result.",
                        },
                    },
                    "required": ["agent", "method", "params", "title"],
                },
            },
        },
        "required": ["reasoning", "steps"],
    },
}


# ── Planner ────────────────────────────────────────────────────────────────────

class Planner:
    """
    Master orchestrator.  One instance per session; conversation_history
    persists across multiple run() / chat() calls.
    """

    # How many past messages to include as context for follow-up classification
    _HISTORY_WINDOW = 6     # 3 turns (user + assistant per turn)

    def __init__(self, db_path: str | None = None) -> None:
        # Load API key from .env
        load_dotenv(_ROOT / ".env")
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY not found.  "
                "Copy .env.example to .env and set your key."
            )

        self.client = anthropic.Anthropic(api_key=api_key)

        # Connect to DuckDB
        resolved_db = db_path or str(_ROOT / "olap.duckdb")
        self.con = duckdb.connect(resolved_db, read_only=True)

        # Initialise all agents with the shared client and connection
        self.navigator = DimensionNavigatorAgent(self.client, self.con)
        self.cube      = CubeOperationsAgent(self.client, self.con)
        self.kpi       = KPICalculatorAgent(self.client, self.con)
        self.reporter  = ReportGeneratorAgent(self.client, self.con)
        self.anomaly        = AnomalyDetectionAgent(self.client, self.con)
        self.visualization  = VisualizationAgent(self.client, self.con)

        # Conversation history: list of {"role": "user"|"assistant", "content": str}
        self.conversation_history: list[dict] = []

    # ── public interface ───────────────────────────────────────────────────────

    def run(self, user_query: str) -> dict:
        """
        Process *user_query*, route to the correct agent(s), chain if needed,
        and return the full formatted report as a dict.

        The conversation history is updated so subsequent calls can reference
        previous results ("now show that for Europe", "break it down by month").

        Returns
        -------
        dict with at minimum: status, operation, report (full text), section_count
        """
        # 1. Record the new user turn
        self.conversation_history.append({"role": "user", "content": user_query})

        # 2. Classify intent → execution plan
        try:
            steps, reasoning = self._classify(user_query)
        except Exception as exc:
            error = {
                "status":  "error",
                "message": f"Classification failed: {exc}",
            }
            self.conversation_history.append(
                {"role": "assistant", "content": str(error)}
            )
            return error

        # 3. Execute each step, attach title for report section headings
        agent_results: list[dict] = []
        for step in steps:
            # Inject OLAP data from the immediately preceding result into
            # visualization steps so the agent doesn't have to query the DB itself.
            if step.get("agent") == "visualization" and agent_results:
                prev  = agent_results[-1]
                data  = prev.get("rows") or prev.get("all_rows") or []
                params = step.setdefault("params", {})
                query  = params.setdefault("query", {})
                query["data"] = data

            result = self._dispatch(step)
            result["title"] = step.get("title", "")
            agent_results.append(result)

        # 4. Pass all results through ReportGenerator
        report = self.reporter.full_report(agent_results)

        # 4b. Bubble up figure_json and anomaly data to the top-level response
        #     so the frontend can render charts and highlighted tables directly.
        for ar in agent_results:
            if "figure_json" in ar and "figure_json" not in report:
                report["figure_json"]    = ar["figure_json"]
                report["chart_type"]     = ar.get("chart_type", "")
                report["chart_reasoning"] = ar.get("reasoning", "")
            if ar.get("anomaly_count", 0) > 0 and "anomalies" not in report:
                report["anomalies"]      = ar["anomalies"]
                report["anomaly_count"]  = ar["anomaly_count"]
                report["interpretation"] = ar.get("interpretation", "")

        # 5. Record the assistant turn (store just the text summary)
        self.conversation_history.append({
            "role":    "assistant",
            "content": report.get("report", report.get("message", "")),
        })

        # Attach routing metadata for transparency
        report["_routing"] = {"reasoning": reasoning, "steps": steps}

        return report

    def chat(self, user_query: str) -> None:
        """
        Interactive helper: calls run() and prints the formatted report to stdout.
        Uses UTF-8 to handle em-dashes and other Unicode safely on all platforms.
        """
        result = self.run(user_query)
        text   = result.get("report") or result.get("message") or repr(result)
        # Write UTF-8 bytes directly so Windows cp1252 console doesn't choke
        sys.stdout.buffer.write((text + "\n").encode("utf-8"))
        sys.stdout.buffer.flush()

    # ── private helpers ───────────────────────────────────────────────────────

    def _classify(self, user_query: str) -> tuple[list[dict], str]:
        """
        Send the query (with recent history for context) to Claude and extract
        the execution plan returned by the create_execution_plan tool.

        Returns
        -------
        (steps, reasoning)
        """
        # Build message list: recent history + current query already appended
        window   = self.conversation_history[-self._HISTORY_WINDOW:]
        messages = window if window else [{"role": "user", "content": user_query}]

        response = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            tools=[_PLAN_TOOL],
            tool_choice={"type": "any"},   # force a tool call
            messages=messages,
        )

        tool_use = next(
            (block for block in response.content if block.type == "tool_use"),
            None,
        )
        if tool_use is None:
            raise RuntimeError("Claude did not return a tool call.")

        plan      = tool_use.input
        steps     = plan.get("steps", [])
        reasoning = plan.get("reasoning", "")
        return steps, reasoning

    def _dispatch(self, step: dict) -> dict:
        """
        Resolve a single plan step to the right agent method and call it.

        Unrecognised agent/method names surface as error dicts rather than
        raising, so a bad step doesn't abort the whole report.
        """
        agent_name  = step.get("agent", "")
        method_name = step.get("method", "")
        params      = step.get("params") or {}

        agent_map: dict[str, Any] = {
            "navigator":     self.navigator,
            "cube":          self.cube,
            "kpi":           self.kpi,
            "anomaly":       self.anomaly,
            "visualization": self.visualization,
        }

        agent = agent_map.get(agent_name)
        if agent is None:
            return {
                "status":  "error",
                "message": f"Unknown agent '{agent_name}'.",
            }

        method = getattr(agent, method_name, None)
        if method is None:
            return {
                "status":  "error",
                "message": f"Agent '{agent_name}' has no method '{method_name}'.",
            }

        try:
            return method(**params)
        except Exception as exc:
            return {
                "status":    "error",
                "operation": method_name,
                "message":   f"{type(exc).__name__}: {exc}",
            }

    def reset(self) -> None:
        """Clear conversation history to start a fresh session."""
        self.conversation_history.clear()
