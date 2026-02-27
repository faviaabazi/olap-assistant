"""
Planner — master orchestrator for the OLAP assistant.
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
    r"\b(top\s+\d+|bottom\s+\d+|ranking|filter|filter by|show|list|ranked|list\s+of|best\s+\d+|worst\s+\d+)\b",
    re.IGNORECASE,
)

_REPORT_PATTERNS = re.compile(
    r"\b(full\s+report|generate\s+report|report)\b",
    re.IGNORECASE,
)

# Maps Q-string aliases → integer quarter value
_QUARTER_ALIAS: dict[str, int] = {
    "q1": 1,
    "q2": 2,
    "q3": 3,
    "q4": 4,
    "1q": 1,
    "2q": 2,
    "3q": 3,
    "4q": 4,
    "quarter 1": 1,
    "quarter 2": 2,
    "quarter 3": 3,
    "quarter 4": 4,
}


def _detect_response_mode(query: str, steps: list[dict]) -> str:
    agent_names = {s.get("agent", "") for s in steps}
    q = query.lower().strip()

    if "visualization" in agent_names:
        return "chart"
    _TREND_KEYWORDS = {"trend", "over time", "trendline", "trend line"}
    if any(kw in q for kw in _TREND_KEYWORDS):
        return "chart"

    if "anomaly" in agent_names:
        return "anomaly"
    if _REPORT_PATTERNS.search(q):
        return "report"
    if "summary" in q:
        return "summary"
    if any(kw in q for kw in _COMPARISON_KEYWORDS):
        return "comparison"
    if _LIST_PATTERNS.search(q):
        return "list"
    if (
        any(q.startswith(p) for p in _DIRECT_PREFIXES)
        and len(steps) == 1
        and steps[0].get("agent")
        not in {"visualization", "executive_summary", "anomaly"}
    ):
        return "direct"

    return "default"


# ── Quarter normalisation ─────────────────────────────────────────────────────


def _coerce_quarter(value: Any) -> Any:
    """
    Convert any quarter representation to a plain integer 1-4.

    Handles: "Q4", "q4", "2024-Q4", "Quarter 4", 4, "4", etc.
    Returns the original value unchanged if it cannot be recognised.
    """
    if isinstance(value, int) and 1 <= value <= 4:
        return value

    if isinstance(value, (int, float)):
        iv = int(value)
        if 1 <= iv <= 4:
            return iv

    if isinstance(value, str):
        s = value.strip().lower()

        # Plain integer string "1".."4"
        if s.isdigit() and 1 <= int(s) <= 4:
            return int(s)

        # Direct alias: "q4", "quarter 4", …
        if s in _QUARTER_ALIAS:
            return _QUARTER_ALIAS[s]

        # Compound: "2024-q4", "2024 q4", "q4-2024", …
        for alias, qint in _QUARTER_ALIAS.items():
            if alias in s:
                return qint

    return value  # unchanged — let the agent surface a proper error


def _normalize_params(agent: str, method: str, params: dict) -> dict:
    """
    Post-process LLM-generated params to fix common type errors before
    they reach agent code.

    Currently handles:
      • quarter values in any string form  → integer 1-4
    """
    if not params:
        return params

    params = dict(params)  # shallow copy — don't mutate the original step

    # ── navigator.drill_down / roll_up ────────────────────────────────────
    # current_value may carry a quarter alias when dimension == "time"
    if agent == "navigator" and method in {"drill_down", "roll_up"}:
        if (
            params.get("dimension") == "time"
            and params.get("current_level") == "quarter"
        ):
            params["current_value"] = _coerce_quarter(params.get("current_value"))

    # ── cube.slice ────────────────────────────────────────────────────────
    if agent == "cube" and method == "slice":
        if params.get("dimension") == "quarter":
            params["value"] = _coerce_quarter(params.get("value"))

    # ── cube.dice ─────────────────────────────────────────────────────────
    if agent == "cube" and method == "dice":
        filters = params.get("filters", {})
        if isinstance(filters, dict) and "quarter" in filters:
            filters = dict(filters)
            filters["quarter"] = _coerce_quarter(filters["quarter"])
            params["filters"] = filters

    # ── kpi.* — filters kwarg ─────────────────────────────────────────────
    if agent == "kpi":
        filters = params.get("filters", {})
        if isinstance(filters, dict) and "quarter" in filters:
            filters = dict(filters)
            filters["quarter"] = _coerce_quarter(filters["quarter"])
            params["filters"] = filters

    # ── anomaly.run — query dict ──────────────────────────────────────────
    if agent == "anomaly" and method == "run":
        q = params.get("query", {})
        if isinstance(q, dict) and "quarter" in q:
            q = dict(q)
            q["quarter"] = _coerce_quarter(q["quarter"])
            params["query"] = q

    return params


# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """
You are the query planner for an OLAP retail sales assistant (Jan 2022 – Dec 2024).
Your only job is to call the `create_execution_plan` tool with an ordered list of
agent steps that answers the user's question.  Never reply in plain text.

━━━ DATA MODEL ━━━

── Dimension Fields ──────────────────────────────────────────────────────────
  Time      : year, quarter, month, month_name, order_date

                ╔══════════════════════════════════════════════════════════╗
                ║  QUARTER VALUES ARE ALWAYS PLAIN INTEGERS: 1 | 2 | 3 | 4 ║
                ║  NEVER pass strings like "Q1", "Q2", "Q3", "Q4",        ║
                ║  "q4", "2024-Q4", "Quarter 4", or any other variant.    ║
                ║  Q1 → 1   Q2 → 2   Q3 → 3   Q4 → 4                     ║
                ║  Example: "Q4 2024"  → year=2024, quarter=4             ║
                ║  Example: "Q1 2023"  → year=2023, quarter=1             ║
                ╚══════════════════════════════════════════════════════════╝

  Geography : region, country

  Product   : category, subcategory

  Customer  : customer_segment

── Fact / Measure Fields ─────────────────────────────────────────────────────
  Measures  : quantity, unit_price, revenue, cost, profit, profit_margin
                profit_margin is a decimal ratio (e.g. 0.3542 = 35.42%)

── Sample Values ─────────────────────────────────────────────────────────────
  region    : North America | Europe | Asia Pacific | Latin America

  country   : United States | Canada | Mexico | Germany | United Kingdom |
              France | Italy | Spain | Netherlands | China | Japan |
              Australia | India | South Korea | Singapore |
              Brazil | Argentina | Colombia | Chile | Peru

  category  : Electronics | Furniture | Office Supplies | Clothing

── Surrogate / Join Keys (internal use only — do not filter or group by) ─────
  date_key, geography_key, product_key, customer_key, order_id

━━━ AVAILABLE AGENTS & METHODS ━━━

agent: "navigator"
  drill_down(dimension, current_level, current_value)
    dimension      : "time" | "geography" | "product"
    current_level  : year|quarter|month  /  region|country  /  category|subcategory
    current_value  : the value to filter at current_level before going deeper
                     For time/quarter: pass an INTEGER (1-4), never a string like "Q4"
  roll_up(dimension, current_level, current_value)
    same signature, navigates one level up instead

agent: "cube"
  slice(dimension, value)
    dimension : any field name
    value     : the value to filter by
                For quarter dimension: INTEGER only (1, 2, 3, or 4)
  dice(filters)
    filters   : dict of field->value for dimensions AND/OR measure thresholds.
                Dimension fields : year, quarter, month, region, country,
                                   category, subcategory, customer_segment
                Measure fields   : revenue, profit, cost, quantity,
                                   order_count, profit_margin
                                   — pass as operator strings:
                                   {"revenue": ">500"} or {"profit": ">=1000"}
                Quarter example  : {"quarter": 4, "year": 2024}  ← integer 4, not "Q4"
                Full example     : {"region": "Asia Pacific", "year": 2023, "revenue": ">500"}
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
                Quarter in filters: integer only, e.g. {"quarter": 4}
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
- "filter by one value" / "list orders by X" / "show orders where X"  → cube.slice
- "filter by multiple values" / "list orders by X and Y"               → cube.dice
- "list all orders" / "show me orders" / "orders in [year/country/etc]" → cube.slice or cube.dice
  Always return individual order rows — never aggregate for slice/dice.
- "pivot / cross-tab / matrix"            → cube.pivot
- "year over year / YoY / annual growth"  → kpi.yoy_growth
- "month over month / MoM / monthly"      → kpi.mom_change
- "top N / best / highest / ranking"      → kpi.top_n
- "margin / profitability / profit %"     → kpi.profit_margins
- "anomaly / outlier / unusual / spike / dip / abnormal" → anomaly.run
- "chart / plot / graph / visualize / show me visually" → prior data step + visualization.run
- "executive summary / narrative / report overview / summarize / briefing" →
      kpi.yoy_growth + kpi.profit_margins + kpi.top_n + executive_summary.run
- "show totals then drill into X" / "totals + breakdown" → always 2 steps:
      cube.slice or cube.dice first, then navigator.drill_down second
- "revenue > X" / "profit above X" / "orders over $X" / any measure threshold →
      cube.dice with measure filter
- Complex queries: use 2-3 steps combined into one report.

━━━ DRILL-DOWN WITH QUARTER — CRITICAL EXAMPLES ━━━
  "Drill Q4 2024 down to months"
    → navigator.drill_down(dimension="time", current_level="quarter", current_value=4)
      with year filter applied via dice first if needed
      OR: cube.dice({"year": 2024, "quarter": 4}) + navigator.drill_down("time","quarter",4)

  "Break Q1 2023 into months"
    → cube.dice({"year": 2023, "quarter": 1}) then drill_down or mom_change

  "Show monthly breakdown of Q3"
    → navigator.drill_down(dimension="time", current_level="quarter", current_value=3)

  Quarter integer mapping — MEMORISE THIS:
    Q1 = 1    Q2 = 2    Q3 = 3    Q4 = 4

━━━ SINGLE-STEP FILTER RULE ━━━
When the user asks to filter / slice / dice the data (cube.slice or cube.dice),
that MUST be a single-step plan.  NEVER combine slice or dice with yoy_growth,
top_n, mom_change, or any other kpi method in the same plan.

DEFAULT behaviour for slice and dice is to return individual order rows
(one row per transaction). Pass summarize=false (or omit it) for this.

Only pass summarize=true when the user EXPLICITLY asks for a summary,
totals, breakdown, or grouped view.

━━━ CHAINING EXAMPLES ━━━
"Revenue in Europe in 2024, then break it down to country"
  → cube.dice({"region":"Europe","year":2024}) + navigator.drill_down(geography, region, Europe)

"Show category totals, then drill into Electronics"
  → kpi.top_n(category, revenue, 10, {}) + navigator.drill_down(product, category, Electronics)

"Full performance overview"
  → kpi.yoy_growth + kpi.profit_margins + cube.pivot

"Give me an executive summary"
  → kpi.yoy_growth(revenue, overall) + kpi.profit_margins(region)
    + kpi.top_n(country, revenue, 5, {}) + executive_summary.run
""".strip()


# ── Classification tool ───────────────────────────────────────────────────────

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
    return result.get("rows") or result.get("all_rows") or []


def _rows_preview(rows: list[dict], limit: int = 12) -> str:
    return json.dumps(rows[:limit], default=str)


def _last_user_topic(history: list[dict]) -> str:
    for msg in reversed(history):
        if msg["role"] == "user":
            return msg["content"]
    return ""


# ── Planner ───────────────────────────────────────────────────────────────────


class Planner:
    _HISTORY_WINDOW = 6

    def __init__(self, api_key: str | None = None, db_path: str | None = None) -> None:
        load_dotenv(_ROOT / ".env")
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY not found.  "
                "Copy .env.example to .env and set your key."
            )

        self.client = anthropic.Anthropic(api_key=resolved_key)

        resolved_db = db_path or str(_ROOT / "olap.duckdb")
        self.con = duckdb.connect(resolved_db, read_only=True)

        self.navigator = DimensionNavigatorAgent(self.client, self.con)
        self.cube = CubeOperationsAgent(self.client, self.con)
        self.kpi = KPICalculatorAgent(self.client, self.con)
        self.reporter = ReportGeneratorAgent(self.client, self.con)
        self.anomaly = AnomalyDetectionAgent(self.client, self.con)
        self.visualization = VisualizationAgent(self.client, self.con)
        self.exec_summary = ExecutiveSummaryAgent(self.client, self.con)

        self.conversation_history: list[dict] = []

    # ── public interface ──────────────────────────────────────────────────────

    def run(self, user_query: str) -> dict:
        with open("debug.log", "a") as f:
            f.write(f"DEBUG RUN CALLED: {user_query}\n")

        self.conversation_history.append({"role": "user", "content": user_query})

        try:
            steps, reasoning = self._classify(user_query)
            with open("debug.log", "a") as f:
                f.write(f"PLAN: {json.dumps(steps, indent=2)}\n")
            response_mode = _detect_response_mode(user_query, steps)
        except Exception as exc:
            error = {"status": "error", "message": f"Classification failed: {exc}"}
            self.conversation_history.append(
                {"role": "assistant", "content": str(error)}
            )
            return error

        agent_results: list[dict] = []
        for step in steps:
            # Normalise params before dispatch (fixes Q-string → int, etc.)
            step["params"] = _normalize_params(
                step.get("agent", ""),
                step.get("method", ""),
                step.get("params") or {},
            )

            if step.get("agent") == "visualization" and agent_results:
                prev = agent_results[-1]
                data = prev.get("rows") or prev.get("all_rows") or []
                params = step.setdefault("params", {})
                query = params.setdefault("query", {})
                query["data"] = data

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

        all_rows: list[dict] = []
        for ar in agent_results:
            all_rows.extend(_extract_rows(ar))

        is_follow_up = len(self.conversation_history) > 2
        prev_topic = (
            _last_user_topic(self.conversation_history[:-1]) if is_follow_up else ""
        )

        response = self._build_response(
            user_query,
            agent_results,
            steps,
            reasoning,
            response_mode,
            all_rows,
            is_follow_up,
            prev_topic,
        )

        finding = response.get("finding", "")
        self.conversation_history.append({"role": "assistant", "content": finding})

        return response

    def chat(self, user_query: str) -> None:
        result = self.run(user_query)
        text = result.get("finding") or result.get("message") or repr(result)
        sys.stdout.buffer.write((text + "\n").encode("utf-8"))
        sys.stdout.buffer.flush()

    # ── classification ────────────────────────────────────────────────────────

    def _classify(self, user_query: str) -> tuple[list[dict], str]:
        window = self.conversation_history[-self._HISTORY_WINDOW :]
        messages = window if window else [{"role": "user", "content": user_query}]

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
            (block for block in response.content if block.type == "tool_use"), None
        )
        if tool_use is None:
            raise RuntimeError("Claude did not return a tool call.")

        plan = tool_use.input
        return plan.get("steps", []), plan.get("reasoning", "")

    # ── dispatch ──────────────────────────────────────────────────────────────

    def _dispatch(self, step: dict) -> dict:
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
            return {"status": "error", "message": f"Unknown agent '{agent_name}'."}

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
        is_follow_up: bool,
        prev_topic: str,
    ) -> dict:
        if response_mode == "anomaly":
            finding = self._generate_anomaly_finding(
                user_query, agent_results, is_follow_up, prev_topic
            )
        else:
            finding = self._generate_finding(
                user_query, all_rows, agent_results, is_follow_up, prev_topic
            )

        follow_ups = self._generate_follow_up_questions(
            user_query, all_rows, response_mode
        )

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

        results_with_rows = [ar for ar in agent_results if _extract_rows(ar)]
        if len(results_with_rows) > 1:
            sections = []
            for ar in agent_results:
                rows = _extract_rows(ar)
                if rows:
                    sections.append(
                        {
                            "title": ar.get("title", ar.get("operation", "")),
                            "rows": rows,
                        }
                    )
            mode_fields["sections"] = sections
            mode_fields.pop("result_rows", None)

        base.update(mode_fields)
        return base

    def _build_direct(self, query, agent_results, steps, all_rows):
        return {"supporting_data": all_rows[:5]}

    def _build_chart(self, query, agent_results, steps, all_rows):
        result: dict = {}
        for ar in agent_results:
            if "figure_json" in ar:
                result["figure_json"] = ar["figure_json"]
                result["chart_type"] = ar.get("chart_type", "")
                break
        result["chart_analysis"] = self._generate_chart_analysis(query, all_rows)
        return result

    def _build_comparison(self, query, agent_results, steps, all_rows):
        result: dict = {"result_rows": all_rows}
        result["comparison_takeaway"] = self._generate_short_text(
            query,
            all_rows,
            "Write 1-2 sentences summarising the key comparison insight. "
            "Mention the biggest change or difference with specific numbers. "
            "Facts only — no recommendations. Be concise.",
        )
        _NO_TABLE_OPS = {"visualization", "executive_summary"}
        report_inputs = [
            ar for ar in agent_results if ar.get("operation") not in _NO_TABLE_OPS
        ]
        report = self.reporter.full_report(report_inputs)
        result["report"] = report.get("report", "")
        result["section_count"] = report.get("section_count", 0)
        return result

    def _build_list(self, query, agent_results, steps, all_rows):
        result: dict = {"result_rows": all_rows}
        result["list_summary"] = self._generate_short_text(
            query,
            all_rows,
            "Write exactly 1 sentence summarising the top result. "
            "Include the top item name and a specific number. "
            "Facts only — no recommendations. Be concise.",
        )
        _NO_TABLE_OPS = {"visualization", "executive_summary"}
        report_inputs = [
            ar for ar in agent_results if ar.get("operation") not in _NO_TABLE_OPS
        ]
        report = self.reporter.full_report(report_inputs)
        result["report"] = report.get("report", "")
        result["section_count"] = report.get("section_count", 0)
        return result

    def _build_summary(self, query, agent_results, steps, all_rows):
        is_detailed = "detailed" in query.lower()
        result: dict = {}
        result["summary_text"] = self._generate_summary_text(
            query, all_rows, agent_results, is_detailed
        )
        for ar in agent_results:
            if ar.get("operation") == "executive_summary":
                result["exec_headline"] = ar.get("headline", "")
                result["exec_insights"] = ar.get("insights", [])
                result["exec_action"] = ar.get("recommended_action", "")
                result["exec_risks"] = ar.get("risks", [])
                break
        return result

    def _build_report(self, query, agent_results, steps, all_rows):
        _NO_TABLE_OPS = {"visualization", "executive_summary"}
        report_inputs = [
            ar for ar in agent_results if ar.get("operation") not in _NO_TABLE_OPS
        ]
        report = self.reporter.full_report(report_inputs)
        result: dict = {
            "report": report.get("report", ""),
            "section_count": report.get("section_count", 0),
        }
        for ar in agent_results:
            if "figure_json" in ar and "figure_json" not in result:
                result["figure_json"] = ar["figure_json"]
                result["chart_type"] = ar.get("chart_type", "")
            if ar.get("anomaly_count", 0) > 0 and "anomalies" not in result:
                result["anomalies"] = ar["anomalies"]
                result["anomaly_count"] = ar["anomaly_count"]
                result["interpretation"] = ar.get("interpretation", "")
            if (
                ar.get("operation") == "executive_summary"
                and "exec_headline" not in result
            ):
                result["exec_headline"] = ar.get("headline", "")
                result["exec_insights"] = ar.get("insights", [])
                result["exec_action"] = ar.get("recommended_action", "")
                result["exec_risks"] = ar.get("risks", [])
        return result

    def _build_anomaly(self, query, agent_results, steps, all_rows):
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

    def _build_default(self, query, agent_results, steps, all_rows):
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
        for ar in agent_results:
            if "figure_json" in ar and "figure_json" not in result:
                result["figure_json"] = ar["figure_json"]
                result["chart_type"] = ar.get("chart_type", "")
            if ar.get("anomaly_count", 0) > 0 and "anomalies" not in result:
                result["anomalies"] = ar["anomalies"]
                result["anomaly_count"] = ar["anomaly_count"]
                result["interpretation"] = ar.get("interpretation", "")
            if (
                ar.get("operation") == "executive_summary"
                and "exec_headline" not in result
            ):
                result["exec_headline"] = ar.get("headline", "")
                result["exec_insights"] = ar.get("insights", [])
                result["exec_action"] = ar.get("recommended_action", "")
                result["exec_risks"] = ar.get("risks", [])
        return result

    # ── LLM text generators ──────────────────────────────────────────────────

    def _generate_finding(self, query, rows, agent_results, is_follow_up, prev_topic):
        if not rows and not agent_results:
            return ""
        agent_messages = [
            ar.get("message", "") for ar in agent_results if ar.get("message")
        ]
        preview = _rows_preview(rows, 10) if rows else "[]"
        context_str = " | ".join(agent_messages[:3]) if agent_messages else ""
        follow_up_instruction = ""
        if is_follow_up and prev_topic:
            follow_up_instruction = f'Start with "Based on your previous question about {prev_topic[:80]}..." '
        prompt = (
            f"Question: {query}\nAgent context: {context_str}\nData: {preview}\n\n"
            "Write 2-3 sentences about what the data is and shows. "
            "Cite the top number or key insight. Describe and analyze facts and trends only. "
            "Never suggest business actions, strategies, or recommendations. "
            "Never say 'you should', 'consider', 'invest', 'prioritize', 'replicate'. "
            f"Never start with 'I' or 'The data shows'. {follow_up_instruction}"
            "Tone: friendly and professional. Return only the finding text, no formatting."
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

    def _generate_anomaly_finding(self, query, agent_results, is_follow_up, prev_topic):
        anomalies: list[dict] = []
        for ar in agent_results:
            anomalies.extend(ar.get("anomalies", []))
        if not anomalies:
            return "No statistically significant anomalies were detected in the data."
        preview = _rows_preview(anomalies, 6)
        follow_up_instruction = ""
        if is_follow_up and prev_topic:
            follow_up_instruction = f'Start with "Based on your previous question about {prev_topic[:80]}..."'
        prompt = (
            f"Question: {query}\nAnomaly data: {preview}\n\n"
            "Write 1-2 sentences describing the anomaly neutrally with specific numbers and z-scores. "
            "No alarm language. No recommendations. No causes. "
            f"{follow_up_instruction} Return only the finding text, no formatting."
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

    def _generate_follow_up_questions(self, query, rows, mode):
        preview = _rows_preview(rows, 6) if rows else "[]"
        prompt = (
            f"User asked: {query}\nResponse mode: {mode}\nData preview: {preview}\n\n"
            "Generate 3 short follow-up questions (max 10 words each). "
            "Only analytical questions. Return as a JSON array of 3 strings, no other text."
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

    def _generate_chart_analysis(self, query, rows):
        if not rows:
            return ""
        preview = _rows_preview(rows, 20)
        prompt = (
            f"Question: {query}\nChart data: {preview}\n\n"
            "Write a full paragraph (4-6 sentences) analyzing what this chart shows. "
            "Mention trends, highest and lowest values with specific numbers. "
            "Professional analyst tone — facts only. No recommendations. Plain prose only."
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

    def _generate_summary_text(self, query, rows, agent_results, is_detailed):
        if not rows and not agent_results:
            return ""
        preview = _rows_preview(rows, 20)
        agent_messages = [
            ar.get("message", "") for ar in agent_results if ar.get("message")
        ]
        context_str = " | ".join(agent_messages[:4])
        instruction = (
            "Write a detailed summary of 6-8 sentences covering multiple metrics and dimensions. "
            "Include specific numbers for all key findings."
            if is_detailed
            else "Write a concise summary of 3-4 sentences focusing on the top insight. Include specific numbers."
        )
        prompt = (
            f"Question: {query}\nAgent context: {context_str}\nData: {preview}\n\n"
            f"{instruction} Summarize findings and trends only. Plain prose only."
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

    def _generate_short_text(self, query, rows, instruction):
        if not rows:
            return ""
        preview = _rows_preview(rows, 12)
        tone_rule = (
            "Tone: friendly and professional. Report findings and trends only. "
            "Never give business recommendations. Never start with I or The data shows."
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
        self.conversation_history.clear()
