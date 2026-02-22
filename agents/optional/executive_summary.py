"""
ExecutiveSummaryAgent — synthesises multiple OLAP results into a C-suite narrative.

Operations
──────────
  summarize(agent_results, business_context)  — structured headline + bullets + action
  highlight_risks(data, metric)               — flag underperforming rows
  run(query)                                  — summarize + risk highlights in one call

Input contract for run()
────────────────────────
  {
    "results": list[dict],   # agent result dicts collected from prior OLAP steps
    "context": str,          # optional — free-text description of the analysis
  }

Output of summarize()
──────────────────────
  headline            — single sentence with the top finding and a specific number
  insights            — 3-5 bullet strings, each with specific numbers
  recommended_action  — 1-2 sentences on what a business leader should do next

Output of highlight_risks()
────────────────────────────
  Rows are flagged as at-risk when their metric value is:
    • negative, or
    • below 75 % of the group mean

  No external statistics libraries are required.

Usage
─────
    import anthropic, duckdb
    from agents.optional.executive_summary import ExecutiveSummaryAgent

    client = anthropic.Anthropic()
    con    = duckdb.connect("olap.duckdb")
    agent  = ExecutiveSummaryAgent(client, con)

    result = agent.run({
        "results": [yoy_result, top_n_result, margin_result],
        "context": "Q4 2024 global retail performance review",
    })
    # result["headline"]            — top finding sentence
    # result["insights"]            — list of 3-5 bullet strings
    # result["recommended_action"]  — what to do next
    # result["risks"]               — at-risk rows from the primary dataset
"""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Any

from agents.base_agent import BaseAgent


# ── Helpers ────────────────────────────────────────────────────────────────────

# Substrings that identify metric columns (checked in order)
_METRIC_HINTS = (
    "revenue", "profit", "cost", "quantity",
    "growth", "margin", "change", "total",
)


def _safe_json(obj: Any) -> Any:
    """Recursively convert Decimal → float so json.dumps works."""
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_json(i) for i in obj]
    return obj


def _to_float(v: Any) -> float | None:
    """Return *v* as float, or None if it is not numeric."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, Decimal):
        return float(v)
    return None


def _extract_rows(result: dict) -> list[dict]:
    """Pull the data rows from any agent result dict."""
    return result.get("rows") or result.get("all_rows") or []


def _detect_metric_col(rows: list[dict], preferred: str = "") -> str:
    """
    Return the best metric column name in *rows*.

    Tries *preferred* first, then column names containing metric hint strings,
    then falls back to the first column whose first-row value is numeric.
    """
    if not rows:
        return ""
    if preferred and preferred in rows[0]:
        if _to_float(rows[0][preferred]) is not None:
            return preferred
    for col in rows[0].keys():
        if any(h in col.lower() for h in _METRIC_HINTS):
            if _to_float(rows[0][col]) is not None:
                return col
    for col, val in rows[0].items():
        if _to_float(val) is not None:
            return col
    return ""


# ── Claude tool schema ─────────────────────────────────────────────────────────

_SUMMARY_TOOL = {
    "name": "produce_executive_summary",
    "description": (
        "Write a structured executive summary for a C-suite audience "
        "based on multiple OLAP analysis results."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "headline": {
                "type": "string",
                "description": (
                    "Single sentence capturing the single most important finding. "
                    "Must include at least one specific number."
                ),
            },
            "insights": {
                "type": "array",
                "minItems": 3,
                "maxItems": 5,
                "items": {"type": "string"},
                "description": (
                    "3 to 5 bullet-point insights, each one sentence with specific numbers. "
                    "Cover performance highlights, trends, and risk areas."
                ),
            },
            "recommended_action": {
                "type": "string",
                "description": (
                    "1 to 2 sentences describing the single most important action "
                    "a business leader should take based on these findings."
                ),
            },
        },
        "required": ["headline", "insights", "recommended_action"],
    },
}


# ── Agent ──────────────────────────────────────────────────────────────────────

class ExecutiveSummaryAgent(BaseAgent):
    """Synthesises multiple OLAP agent results into a structured C-suite narrative."""

    # ── public methods ─────────────────────────────────────────────────────────

    def summarize(
        self,
        agent_results: list[dict],
        business_context: str = "",
    ) -> dict:
        """
        Extract key metrics from multiple agent results and produce a structured
        executive narrative with a headline, 3-5 bullet insights, and a
        recommended action.

        Parameters
        ----------
        agent_results:
            List of result dicts as returned by any OLAP agent.  Each should
            have at minimum: operation, message, and rows (or all_rows).
        business_context:
            Optional description of the analysis scope, e.g.
            "Q4 2024 global retail performance review".

        Returns
        -------
        dict with keys:
            status, operation, headline, insights, recommended_action, message
        """
        ok_results = [r for r in agent_results if r.get("status") != "error"]
        if not ok_results:
            return {
                "status":  "error",
                "message": "No successful agent results to summarise.",
            }

        # Build a condensed payload for Claude: operation + message + sample rows
        condensed: list[dict] = []
        for r in ok_results:
            rows = _extract_rows(r)
            condensed.append({
                "operation":   r.get("operation", "unknown"),
                "message":     r.get("message", ""),
                "metric":      r.get("metric") or r.get("measure", ""),
                "dimension":   r.get("dimension") or r.get("row_dim", ""),
                "sample_rows": _safe_json(rows[:8]),
            })

        data_json = json.dumps(condensed, indent=2)

        prompt = (
            "You are a senior analyst preparing an executive briefing for the "
            "C-suite of a global retail company (data covers Jan 2022 – Dec 2024).\n\n"
            f"Business context: {business_context or 'Retail OLAP sales performance analysis'}\n\n"
            "The following OLAP analysis results have been gathered:\n\n"
            f"{data_json}\n\n"
            "Produce a structured executive summary using the tool provided. "
            "Write for decision-makers: be specific with numbers and concise. "
            "Summarize findings and trends only. No recommendations, "
            "no action items, no strategic advice."
        )

        response = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=700,
            tools=[_SUMMARY_TOOL],
            tool_choice={"type": "any"},
            messages=[{"role": "user", "content": prompt}],
        )

        tool_use = next(
            (block for block in response.content if block.type == "tool_use"),
            None,
        )
        if tool_use is None:
            return {
                "status":  "error",
                "message": "Claude did not return a structured summary.",
            }

        args     = tool_use.input
        headline = args.get("headline", "")
        insights = args.get("insights", [])
        action   = args.get("recommended_action", "")

        return {
            "status":             "ok",
            "operation":          "executive_summary",
            "headline":           headline,
            "insights":           insights,
            "recommended_action": action,
            "message":            headline,
        }

    def highlight_risks(
        self,
        data: list[dict],
        metric: str,
    ) -> dict:
        """
        Identify underperforming rows in *data* for *metric*.

        A row is flagged as at-risk when its metric value is either:
          • negative, or
          • below 75 % of the group mean.

        All statistics are computed in plain Python — no external libraries.

        Parameters
        ----------
        data:
            List of row dicts, each containing a column matching *metric*.
        metric:
            The column name (or partial name) to analyse,
            e.g. "total_revenue", "profit", "yoy_growth_pct".

        Returns
        -------
        dict with keys:
            status, operation, metric, mean, at_risk, risk_count, message
        """
        if not data:
            return {"status": "error", "message": "No data provided."}

        # Resolve actual column (exact match, then hint-based detection)
        actual_col = _detect_metric_col(data, metric)
        if not actual_col:
            return {
                "status":  "error",
                "message": (
                    f"Metric column '{metric}' not found. "
                    f"Available columns: {list(data[0].keys())}"
                ),
            }

        # Collect rows with valid numeric values
        pairs: list[tuple[dict, float]] = [
            (row, _to_float(row.get(actual_col)))          # type: ignore[arg-type]
            for row in data
            if _to_float(row.get(actual_col)) is not None
        ]

        if not pairs:
            return {
                "status":  "error",
                "message": f"No numeric values in column '{actual_col}'.",
            }

        nums      = [v for _, v in pairs]
        mean      = sum(nums) / len(nums)
        threshold = mean * 0.75

        at_risk: list[dict] = []
        for row, v in pairs:
            if v < 0:
                reason = (
                    f"Negative {actual_col.replace('_', ' ')} "
                    f"({v:,.2f})"
                )
            elif v < threshold:
                pct    = (v / mean * 100) if mean else 0
                reason = (
                    f"Below average — {pct:.0f}% of mean "
                    f"({mean:,.2f})"
                )
            else:
                continue

            at_risk.append({**_safe_json(row), "risk_reason": reason})

        label = actual_col.replace("_", " ").title()
        return {
            "status":     "ok",
            "operation":  "highlight_risks",
            "metric":     actual_col,
            "mean":       round(mean, 2),
            "at_risk":    at_risk,
            "risk_count": len(at_risk),
            "message": (
                f"Identified {len(at_risk)} at-risk area(s) for {label} "
                f"(mean = {mean:,.2f}, threshold = {threshold:,.2f})."
            ),
        }

    def run(self, query: dict) -> dict:
        """
        Produce an executive summary and risk highlights from a set of prior
        agent results in a single call.

        Parameters
        ----------
        query : dict
            {
                "results": list[dict],  # agent result dicts from preceding steps
                "context": str,         # optional — business context description
            }

        Returns
        -------
        dict with keys:
            status, operation, headline, insights, recommended_action,
            risks, risk_count, rows, message
        """
        if not isinstance(query, dict):
            return {
                "status":  "error",
                "message": "run() expects a dict with keys: results, context.",
            }

        results = query.get("results", [])
        context = query.get("context", "")

        if not results:
            return {"status": "error", "message": "No agent results provided."}

        # ── summarize ──────────────────────────────────────────────────────
        summary = self.summarize(results, context)
        if summary.get("status") == "error":
            return summary

        # ── highlight risks from the first available dataset ───────────────
        all_rows: list[dict] = []
        metric_col: str = ""
        for r in results:
            rows = _extract_rows(r)
            if rows:
                col = _detect_metric_col(rows, r.get("metric", ""))
                if col:
                    all_rows  = rows
                    metric_col = col
                    break

        risks_result: dict = {}
        if all_rows and metric_col:
            risks_result = self.highlight_risks(all_rows, metric_col)

        at_risk    = risks_result.get("at_risk", [])
        risk_count = risks_result.get("risk_count", 0)

        return {
            "status":             "ok",
            "operation":          "executive_summary",
            "headline":           summary["headline"],
            "insights":           summary["insights"],
            "recommended_action": summary["recommended_action"],
            "risks":              at_risk,
            "risk_count":         risk_count,
            "rows":               [],   # no raw table — content is in structured fields
            "message":            summary["headline"],
        }
