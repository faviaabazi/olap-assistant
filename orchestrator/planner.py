"""
Planner — master orchestrator for the OLAP assistant.

Loads credentials, initialises all agents, uses Claude to classify every
user query into a structured execution plan, dispatches to the correct agent
methods (chaining where needed), and returns structured responses with
mode-specific fields, a finding sentence, and follow-up questions.

Response modes
──────────────
  direct, chart, comparison, list, summary, report, anomaly, default

Every response includes: finding, follow_up_questions, response_mode, plus
mode-specific fields (see _build_* methods).

Usage
─────
    from orchestrator.planner import Planner

    planner = Planner()
    result = planner.run("Show me year-over-year revenue growth by region")
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import sys
from typing import Any

import anthropic
import duckdb
from dotenv import load_dotenv

from agents.cube_operations import CubeOperationsAgent
from agents.dimension_navigator import DimensionNavigatorAgent
from agents.kpi_calculator import KPICalculatorAgent
from agents.optional.anomaly_detection import AnomalyDetectionAgent
from agents.optional.executive_summary import ExecutiveSummaryAgent
from agents.optional.visualization import VisualizationAgent
from agents.report_generator import ReportGeneratorAgent

# ── Agent imports ──────────────────────────────────────────────────────────────
# Add project root to path so imports work regardless of working directory
_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ── Response-mode detection ───────────────────────────────────────────────────

_COMPARISON_KEYWORDS = frozenset(
    {
        "vs",
        "versus",
        "compare",
        "comparison",
        "difference",
        "diff",
        "change",
        "growth",
        "yoy",
        "mom",
        "year-over-year",
        "month-over-month",
        "year over year",
        "month over month",
    }
)

_DIRECT_PREFIXES = (
    "which",
    "what is",
    "what was",
    "what are",
    "what were",
    "who is",
    "who are",
    "how much",
    "how many",
    "is there",
    "are there",
    "what's",
    "who's",
)

_LIST_PATTERNS = re.compile(
    r"\b(top\s+\d+|bottom\s+\d+|ranking|ranked|list\s+of|best\s+\d+|worst\s+\d+)\b",
    re.IGNORECASE,
)

_REPORT_PATTERNS = re.compile(
    r"\b(full\s+report|generate\s+report|report)\b",
    re.IGNORECASE,
)


def _detect_response_mode(query: str, steps: list[dict]) -> str:
    """
    Determine response mode from query text and execution plan steps.

    Priority: chart > anomaly > report > summary > comparison > list > direct > default
    """
    agent_names = {s.get("agent", "") for s in steps}
    q = query.lower().strip()

    # Plan-based modes
    if "visualization" in agent_names:
        return "chart"
    if "anomaly" in agent_names:
        return "anomaly"

    # Explicit report request
    if _REPORT_PATTERNS.search(q):
        return "report"

    # Summary mode
    if "summary" in q:
        return "summary"

    # Comparison keywords
    if any(kw in q for kw in _COMPARISON_KEYWORDS):
        return "comparison"

    # List / ranking
    if _LIST_PATTERNS.search(q):
        return "list"

    # Direct factual question + single step
    if (
        any(q.startswith(p) for p in _DIRECT_PREFIXES)
        and len(steps) == 1
        and steps[0].get("agent") not in {"visualization", "executive_summary", "anomaly"}
    ):
        return "direct"

    return "default"


# ── System prompt for query classification ────────────────────────────────────

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

agent: "executive_summary"
  run(query)
    query : {
      "context": str,  # describe the scope, e.g. "Q4 2024 global retail review"
    }
    IMPORTANT: prior agent results are injected automatically — ALL preceding steps
    are passed in as input.  Always place executive_summary as the LAST step.
    Use when the user asks for: executive summary, narrative, report overview,
    summary of findings, or a high-level briefing.
    For a standalone summary, precede it with 2-3 data steps (yoy_growth,
    profit_margins, top_n) so there is enough material to synthesise.

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
- "executive summary / narrative / report overview / summarize / briefing" →
      kpi.yoy_growth + kpi.profit_margins + kpi.top_n + executive_summary.run
- Complex queries: use 2-3 steps combined into one report.

━━━ CHAINING EXAMPLES ━━━
"Revenue in Europe in 2024, then break it down to country"
  → cube.dice({"region":"Europe","year":2024}) + navigator.drill_down(geography, region, Europe)

"Top products in Asia Pacific with quarterly breakdown"
  → kpi.top_n(subcategory, revenue, 5, {"region":"Asia Pacific"})
    + kpi.mom_change(revenue, 2024)  [or yoy_growth]

"Full performance overview"
  → kpi.yoy_growth + kpi.profit_margins + cube.pivot

"Give me an executive summary"
  → kpi.yoy_growth(revenue, overall) + kpi.profit_margins(region)
    + kpi.top_n(country, revenue, 5, {}) + executive_summary.run
""".strip()


# ── Classification tool ──────────────────────────────────────────────────────

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
                            "enum": [
                                "navigator",
                                "cube",
                                "kpi",
                                "anomaly",
                                "visualization",
                                "executive_summary",
                            ],
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_rows(result: dict) -> list[dict]:
    """Pull data rows from any agent result dict."""
    return result.get("rows") or result.get("all_rows") or []


def _rows_preview(rows: list[dict], limit: int = 12) -> str:
    """JSON preview of rows for LLM prompts."""
    return json.dumps(rows[:limit], default=str)


def _last_user_topic(history: list[dict]) -> str:
    """Extract the topic from the most recent user message in history."""
    for msg in reversed(history):
        if msg["role"] == "user":
            return msg["content"]
    return ""


# ── Planner ──────────────────────────────────────────────────────────────────


class Planner:
    """
    Master orchestrator.  One instance per session; conversation_history
    persists across multiple run() / chat() calls.
    """

    # How many past messages to include as context for follow-up classification
    _HISTORY_WINDOW = 6  # 3 turns (user + assistant per turn)

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
        self.cube = CubeOperationsAgent(self.client, self.con)
        self.kpi = KPICalculatorAgent(self.client, self.con)
        self.reporter = ReportGeneratorAgent(self.client, self.con)
        self.anomaly = AnomalyDetectionAgent(self.client, self.con)
        self.visualization = VisualizationAgent(self.client, self.con)
        self.exec_summary = ExecutiveSummaryAgent(self.client, self.con)

        # Conversation history: list of {"role": "user"|"assistant", "content": str}
        self.conversation_history: list[dict] = []

    # ── public interface ──────────────────────────────────────────────────────

    def run(self, user_query: str) -> dict:
        """
        Process *user_query*, route to the correct agent(s), chain if needed,
        and return a structured response dict.

        Every response includes: status, response_mode, finding,
        follow_up_questions, plus mode-specific fields.
        """
        # 1. Record the new user turn
        self.conversation_history.append({"role": "user", "content": user_query})

        # 2. Classify intent → execution plan
        try:
            steps, reasoning = self._classify(user_query)
            response_mode = _detect_response_mode(user_query, steps)
        except Exception as exc:
            error = {
                "status": "error",
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
            # visualization steps so the agent doesn't have to query the DB.
            if step.get("agent") == "visualization" and agent_results:
                prev = agent_results[-1]
                data = prev.get("rows") or prev.get("all_rows") or []
                params = step.setdefault("params", {})
                query = params.setdefault("query", {})
                query["data"] = data

            # Inject ALL preceding results into executive_summary.
            if step.get("agent") == "executive_summary" and agent_results:
                raw = step.get("params") or {}
                inner = raw.get("query") if isinstance(raw.get("query"), dict) else raw
                context = inner.get("context", "")
                step["params"] = {
                    "query": {"context": context, "results": agent_results}
                }

            result = self._dispatch(step)
            result["title"] = step.get("title", "")
            agent_results.append(result)

        # 4. Collect all data rows across results + build sections
        all_rows: list[dict] = []
        sections: list[dict] = []
        for ar in agent_results:
            rows = _extract_rows(ar)
            all_rows.extend(rows)
            if rows:
                sections.append({
                    "title": ar.get("title", ""),
                    "rows": rows,
                    "explanation": ar.get("message", ""),
                })

        # 5. Build mode-specific response
        is_follow_up = len(self.conversation_history) > 2  # more than just this turn
        prev_topic = _last_user_topic(self.conversation_history[:-1]) if is_follow_up else ""

        response = self._build_response(
            user_query, agent_results, steps, reasoning,
            response_mode, all_rows, sections, is_follow_up, prev_topic,
        )

        # 6. Store finding (not full report) in conversation history
        finding = response.get("finding", "")
        self.conversation_history.append(
            {"role": "assistant", "content": finding}
        )

        return response

    def chat(self, user_query: str) -> None:
        """
        Interactive helper: calls run() and prints the finding to stdout.
        """
        result = self.run(user_query)
        text = result.get("finding") or result.get("message") or repr(result)
        sys.stdout.buffer.write((text + "\n").encode("utf-8"))
        sys.stdout.buffer.flush()

    # ── classification ────────────────────────────────────────────────────────

    def _classify(self, user_query: str) -> tuple[list[dict], str]:
        """
        Send the query (with recent history for context) to Claude and extract
        the execution plan returned by the create_execution_plan tool.

        Returns (steps, reasoning).
        """
        # Build message list: recent history + current query already appended
        window = self.conversation_history[-self._HISTORY_WINDOW:]
        messages = window if window else [{"role": "user", "content": user_query}]

        # Add follow-up context to system prompt if we have prior turns
        system = _SYSTEM_PROMPT
        prior_user_msgs = [
            m for m in self.conversation_history[:-1] if m["role"] == "user"
        ]
        if prior_user_msgs:
            last_topic = prior_user_msgs[-1]["content"]
            system += (
                f"\n\n━━━ FOLLOW-UP CONTEXT ━━━\n"
                f"Previous query was about: {last_topic}\n"
                f"If this query uses words like 'now', 'also', 'that', 'same', "
                f"'instead', treat it as a follow-up and preserve filters from "
                f"the previous query."
            )

        response = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=system,
            tools=[_PLAN_TOOL],
            tool_choice={"type": "any"},
            messages=messages,
        )

        tool_use = next(
            (block for block in response.content if block.type == "tool_use"),
            None,
        )
        if tool_use is None:
            raise RuntimeError("Claude did not return a tool call.")

        plan = tool_use.input
        steps = plan.get("steps", [])
        reasoning = plan.get("reasoning", "")
        return steps, reasoning

    # ── dispatch ──────────────────────────────────────────────────────────────

    def _dispatch(self, step: dict) -> dict:
        """
        Resolve a single plan step to the right agent method and call it.
        """
        agent_name = step.get("agent", "")
        method_name = step.get("method", "")
        params = step.get("params") or {}

        agent_map: dict[str, Any] = {
            "navigator": self.navigator,
            "cube": self.cube,
            "kpi": self.kpi,
            "anomaly": self.anomaly,
            "visualization": self.visualization,
            "executive_summary": self.exec_summary,
        }

        agent = agent_map.get(agent_name)
        if agent is None:
            return {
                "status": "error",
                "message": f"Unknown agent '{agent_name}'.",
            }

        method = getattr(agent, method_name, None)
        if method is None:
            return {
                "status": "error",
                "message": f"Agent '{agent_name}' has no method '{method_name}'.",
            }

        try:
            return method(**params)
        except Exception as exc:
            return {
                "status": "error",
                "operation": method_name,
                "message": f"{type(exc).__name__}: {exc}",
            }

    # ── response builders ─────────────────────────────────────────────────────

    def _build_response(
        self,
        user_query: str,
        agent_results: list[dict],
        steps: list[dict],
        reasoning: str,
        response_mode: str,
        all_rows: list[dict],
        sections: list[dict],
        is_follow_up: bool,
        prev_topic: str,
    ) -> dict:
        """Dispatch to the correct mode-specific builder."""

        # Generate common fields: finding + follow_up_questions
        if response_mode == "anomaly":
            finding = self._generate_anomaly_finding(
                user_query, agent_results, is_follow_up, prev_topic,
            )
        else:
            finding = self._generate_finding(
                user_query, all_rows, agent_results, is_follow_up, prev_topic,
            )
        follow_ups = self._generate_follow_up_questions(user_query, all_rows, response_mode)

        base = {
            "status": "ok",
            "response_mode": response_mode,
            "finding": finding,
            "follow_up_questions": follow_ups,
            "_routing": {"reasoning": reasoning, "steps": steps},
        }

        builder_map = {
            "direct": self._build_direct,
            "chart": self._build_chart,
            "comparison": self._build_comparison,
            "list": self._build_list,
            "summary": self._build_summary,
            "report": self._build_report,
            "anomaly": self._build_anomaly,
            "default": self._build_default,
        }

        builder = builder_map.get(response_mode, self._build_default)
        mode_fields = builder(user_query, agent_results, steps, all_rows)

        # Attach sections for multi-step rendering
        if len(sections) > 1:
            mode_fields["sections"] = sections

        base.update(mode_fields)

        return base

    def _build_direct(
        self, query: str, agent_results: list[dict],
        steps: list[dict], all_rows: list[dict],
    ) -> dict:
        """Direct mode: supporting_data (max 5 rows)."""
        rows = all_rows[:5]
        return {"supporting_data": rows}

    def _build_chart(
        self, query: str, agent_results: list[dict],
        steps: list[dict], all_rows: list[dict],
    ) -> dict:
        """Chart mode: figure_json + chart_analysis."""
        result: dict = {}

        for ar in agent_results:
            if "figure_json" in ar:
                result["figure_json"] = ar["figure_json"]
                result["chart_type"] = ar.get("chart_type", "")
                break

        # Generate chart analysis via sonnet
        result["chart_analysis"] = self._generate_chart_analysis(query, all_rows)

        return result

    def _build_comparison(
        self, query: str, agent_results: list[dict],
        steps: list[dict], all_rows: list[dict],
    ) -> dict:
        """Comparison mode: result_rows + comparison_takeaway."""
        result: dict = {"result_rows": all_rows}

        result["comparison_takeaway"] = self._generate_short_text(
            query, all_rows,
            "Write 1-2 sentences summarising the key comparison insight. "
            "Mention the biggest change or difference with specific numbers. "
            "Facts only — no recommendations. Be concise.",
        )

        # Also generate the full report for the expander
        _NO_TABLE_OPS = {"visualization", "executive_summary"}
        report_inputs = [
            ar for ar in agent_results if ar.get("operation") not in _NO_TABLE_OPS
        ]
        report = self.reporter.full_report(report_inputs)
        result["report"] = report.get("report", "")
        result["section_count"] = report.get("section_count", 0)

        return result

    def _build_list(
        self, query: str, agent_results: list[dict],
        steps: list[dict], all_rows: list[dict],
    ) -> dict:
        """List mode: result_rows + list_summary."""
        result: dict = {"result_rows": all_rows}

        result["list_summary"] = self._generate_short_text(
            query, all_rows,
            "Write exactly 1 sentence summarising the top result. "
            "Include the top item name and a specific number. "
            "Facts only — no recommendations. Be concise.",
        )

        # Also generate the full report for the expander
        _NO_TABLE_OPS = {"visualization", "executive_summary"}
        report_inputs = [
            ar for ar in agent_results if ar.get("operation") not in _NO_TABLE_OPS
        ]
        report = self.reporter.full_report(report_inputs)
        result["report"] = report.get("report", "")
        result["section_count"] = report.get("section_count", 0)

        return result

    def _build_summary(
        self, query: str, agent_results: list[dict],
        steps: list[dict], all_rows: list[dict],
    ) -> dict:
        """Summary mode: summary_text (short or long based on 'detailed')."""
        is_detailed = "detailed" in query.lower()
        result: dict = {}

        result["summary_text"] = self._generate_summary_text(
            query, all_rows, agent_results, is_detailed,
        )

        # Bubble up exec summary fields if present
        for ar in agent_results:
            if ar.get("operation") == "executive_summary":
                result["exec_headline"] = ar.get("headline", "")
                result["exec_insights"] = ar.get("insights", [])
                result["exec_action"] = ar.get("recommended_action", "")
                result["exec_risks"] = ar.get("risks", [])
                break

        return result

    def _build_report(
        self, query: str, agent_results: list[dict],
        steps: list[dict], all_rows: list[dict],
    ) -> dict:
        """Report mode: full report text."""
        _NO_TABLE_OPS = {"visualization", "executive_summary"}
        report_inputs = [
            ar for ar in agent_results if ar.get("operation") not in _NO_TABLE_OPS
        ]
        report = self.reporter.full_report(report_inputs)

        result: dict = {
            "report": report.get("report", ""),
            "section_count": report.get("section_count", 0),
        }

        # Bubble up special sections
        for ar in agent_results:
            if "figure_json" in ar and "figure_json" not in result:
                result["figure_json"] = ar["figure_json"]
                result["chart_type"] = ar.get("chart_type", "")
            if ar.get("anomaly_count", 0) > 0 and "anomalies" not in result:
                result["anomalies"] = ar["anomalies"]
                result["anomaly_count"] = ar["anomaly_count"]
                result["interpretation"] = ar.get("interpretation", "")
            if ar.get("operation") == "executive_summary" and "exec_headline" not in result:
                result["exec_headline"] = ar.get("headline", "")
                result["exec_insights"] = ar.get("insights", [])
                result["exec_action"] = ar.get("recommended_action", "")
                result["exec_risks"] = ar.get("risks", [])

        return result

    def _build_anomaly(
        self, query: str, agent_results: list[dict],
        steps: list[dict], all_rows: list[dict],
    ) -> dict:
        """Anomaly mode: anomalies + interpretation."""
        result: dict = {}

        for ar in agent_results:
            if ar.get("anomaly_count", 0) > 0:
                result["anomalies"] = ar["anomalies"]
                result["anomaly_count"] = ar["anomaly_count"]
                result["interpretation"] = ar.get("interpretation", "")
                break
        else:
            result["anomalies"] = []
            result["anomaly_count"] = 0
            result["interpretation"] = "No anomalies were detected."

        return result

    def _build_default(
        self, query: str, agent_results: list[dict],
        steps: list[dict], all_rows: list[dict],
    ) -> dict:
        """Default mode: result_rows or report."""
        _NO_TABLE_OPS = {"visualization", "executive_summary"}
        report_inputs = [
            ar for ar in agent_results if ar.get("operation") not in _NO_TABLE_OPS
        ]
        report = self.reporter.full_report(report_inputs)

        result: dict = {
            "result_rows": all_rows,
            "report": report.get("report", ""),
            "section_count": report.get("section_count", 0),
        }

        # Bubble up special sections
        for ar in agent_results:
            if "figure_json" in ar and "figure_json" not in result:
                result["figure_json"] = ar["figure_json"]
                result["chart_type"] = ar.get("chart_type", "")
            if ar.get("anomaly_count", 0) > 0 and "anomalies" not in result:
                result["anomalies"] = ar["anomalies"]
                result["anomaly_count"] = ar["anomaly_count"]
                result["interpretation"] = ar.get("interpretation", "")
            if ar.get("operation") == "executive_summary" and "exec_headline" not in result:
                result["exec_headline"] = ar.get("headline", "")
                result["exec_insights"] = ar.get("insights", [])
                result["exec_action"] = ar.get("recommended_action", "")
                result["exec_risks"] = ar.get("risks", [])

        return result

    # ── LLM text generators ──────────────────────────────────────────────────

    def _generate_finding(
        self,
        query: str,
        rows: list[dict],
        agent_results: list[dict],
        is_follow_up: bool,
        prev_topic: str,
    ) -> str:
        """
        Generate a 1-2 sentence finding using Haiku.

        Friendly-professional tone, cites key numbers.
        If follow-up, starts with "Based on your previous question about [topic]..."
        Never suggests actions or recommendations.
        Never starts with "I" or "The data shows".
        """
        if not rows and not agent_results:
            return ""

        # Gather context from agent messages
        agent_messages = [
            ar.get("message", "") for ar in agent_results
            if ar.get("message")
        ]

        preview = _rows_preview(rows, 10) if rows else "[]"
        context_str = " | ".join(agent_messages[:3]) if agent_messages else ""

        follow_up_instruction = ""
        if is_follow_up and prev_topic:
            short_topic = prev_topic[:80]
            follow_up_instruction = (
                f'Start with "Based on your previous question about {short_topic}..." '
            )

        prompt = (
            f"Question: {query}\n"
            f"Agent context: {context_str}\n"
            f"Data: {preview}\n\n"
            "Write 1-2 sentences about what the data shows. "
            "Cite the top number or key insight. "
            "Describe and analyze facts and trends only. "
            "Never suggest business actions, strategies, or recommendations. "
            "Never say 'you should', 'consider', 'invest', 'prioritize', 'replicate'. "
            "Never start with 'I' or 'The data shows'. "
            f"{follow_up_instruction}"
            "Tone: friendly and professional, like a smart analyst briefing "
            "a senior manager. Report findings and trends only. "
            "Never give business recommendations or action items. "
            "Return only the finding text, no formatting."
        )

        try:
            resp = self.client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text.strip() if resp.content else ""
        except Exception:
            return ""

    def _generate_anomaly_finding(
        self,
        query: str,
        agent_results: list[dict],
        is_follow_up: bool,
        prev_topic: str,
    ) -> str:
        """
        Generate a neutral anomaly finding using Haiku.

        Describes the anomaly with specific numbers and z-scores.
        No alarm language, no recommendations.
        """
        # Collect anomaly data from results
        anomalies: list[dict] = []
        for ar in agent_results:
            anomalies.extend(ar.get("anomalies", []))

        if not anomalies:
            return "No statistically significant anomalies were detected in the data."

        preview = _rows_preview(anomalies, 6)

        follow_up_instruction = ""
        if is_follow_up and prev_topic:
            short_topic = prev_topic[:80]
            follow_up_instruction = (
                f'Start with "Based on your previous question about {short_topic}..." '
            )

        prompt = (
            f"Question: {query}\n"
            f"Anomaly data: {preview}\n\n"
            "Write 1-2 sentences describing the anomaly neutrally with specific numbers "
            "and z-scores. State what the value is, the mean, and how many standard "
            "deviations away it falls. "
            "No alarm language. No recommendations. No causes. "
            "Example tone: 'Laptops recorded a z-score of 2.79, significantly above "
            "the subcategory mean of $394K at $1.86M in total profit.' "
            f"{follow_up_instruction}"
            "Tone: friendly and professional, like a smart analyst briefing "
            "a senior manager. Report findings and trends only. "
            "Never give business recommendations or action items. "
            "Never start with I or The data shows. "
            "Return only the finding text, no formatting."
        )

        try:
            resp = self.client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text.strip() if resp.content else ""
        except Exception:
            return ""

    def _generate_follow_up_questions(
        self, query: str, rows: list[dict], mode: str,
    ) -> list[str]:
        """
        Generate 3 suggested follow-up questions using Haiku.

        Short, natural language, max 10 words each. Questions only.
        """
        preview = _rows_preview(rows, 6) if rows else "[]"

        prompt = (
            f"User asked: {query}\n"
            f"Response mode: {mode}\n"
            f"Data preview: {preview}\n\n"
            "Generate 3 short follow-up questions (max 10 words each). "
            "Questions must be about exploring the data further. "
            "No recommendations. No action items. Only analytical questions. "
            "Tone: friendly and professional, like a smart analyst briefing "
            "a senior manager. Report findings and trends only. "
            "Never give business recommendations or action items. "
            "Never start with I or The data shows. "
            "Return as a JSON array of 3 strings, no other text."
        )

        try:
            resp = self.client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text.strip() if resp.content else "[]"
            questions = json.loads(text)
            if isinstance(questions, list) and len(questions) >= 3:
                return [str(q) for q in questions[:3]]
            return []
        except Exception:
            return []

    def _generate_chart_analysis(self, query: str, rows: list[dict]) -> str:
        """
        Generate a full paragraph (4-6 sentences) analyzing chart data using Sonnet.

        Professional analyst tone — facts and observations only.
        """
        if not rows:
            return ""

        preview = _rows_preview(rows, 20)

        prompt = (
            f"Question: {query}\n"
            f"Chart data: {preview}\n\n"
            "Write a full paragraph (4-6 sentences) analyzing what this chart shows. "
            "Mention trends, highest and lowest values with specific numbers. "
            "Professional analyst tone — facts and observations only. "
            "No recommendations, no action items, no suggestions. "
            "Plain prose only."
        )

        try:
            resp = self.client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=350,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text.strip() if resp.content else ""
        except Exception:
            return ""

    def _generate_summary_text(
        self, query: str, rows: list[dict],
        agent_results: list[dict], is_detailed: bool,
    ) -> str:
        """
        Generate summary text using Sonnet.

        Detailed: 6-8 sentences covering multiple metrics.
        Concise: 3-4 sentences, top insight only.
        """
        if not rows and not agent_results:
            return ""

        preview = _rows_preview(rows, 20)
        agent_messages = [
            ar.get("message", "") for ar in agent_results if ar.get("message")
        ]
        context_str = " | ".join(agent_messages[:4])

        if is_detailed:
            instruction = (
                "Write a detailed summary of 6-8 sentences covering multiple metrics "
                "and dimensions. Include specific numbers for all key findings."
            )
        else:
            instruction = (
                "Write a concise summary of 3-4 sentences focusing on the top insight. "
                "Include specific numbers."
            )

        prompt = (
            f"Question: {query}\n"
            f"Agent context: {context_str}\n"
            f"Data: {preview}\n\n"
            f"{instruction} "
            "Summarize findings and trends only. "
            "No recommendations, no action items, no strategic advice. "
            "Plain prose only."
        )

        try:
            resp = self.client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text.strip() if resp.content else ""
        except Exception:
            return ""

    def _generate_short_text(
        self, query: str, rows: list[dict], instruction: str,
    ) -> str:
        """
        Generate short text (list_summary, comparison_takeaway) using Haiku.
        """
        if not rows:
            return ""
        preview = _rows_preview(rows, 12)
        tone_rule = (
            "Tone: friendly and professional, like a smart analyst briefing "
            "a senior manager. Report findings and trends only. "
            "Never give business recommendations or action items. "
            "Never start with I or The data shows."
        )
        try:
            resp = self.client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=120,
                messages=[
                    {
                        "role": "user",
                        "content": f"Query: {query}\nData: {preview}\n\n{instruction}\n{tone_rule}",
                    }
                ],
            )
            return resp.content[0].text.strip() if resp.content else ""
        except Exception:
            return ""

    def reset(self) -> None:
        """Clear conversation history to start a fresh session."""
        self.conversation_history.clear()
