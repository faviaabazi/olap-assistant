"""
ReportGeneratorAgent — formats OLAP results into structured table data.

Operations
──────────
  formatted_table(data, title)     — structured table dict with metadata
  full_report(agent_results)       — combine multiple agent results into sections

run(query) interface
────────────────────
  query must be a dict with:
    "type"    : "table" | "report"
    "data"    : list[dict]  (rows for table)  or  list[dict]  (agent results for report)
    "title"   : str  (optional)
    "context" : str  (optional)
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from agents.base_agent import BaseAgent


# ── Formatting helpers ────────────────────────────────────────────────────────

_SUMMABLE_SUFFIXES = (
    "revenue", "profit", "cost", "quantity", "count",
)


def _is_summable_col(col: str) -> bool:
    return col.lower().endswith(_SUMMABLE_SUFFIXES)


def _safe_json(obj: Any) -> Any:
    """Recursively convert Decimal to float so json.dumps works."""
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_json(i) for i in obj]
    return obj


# ── Agent ─────────────────────────────────────────────────────────────────────

class ReportGeneratorAgent(BaseAgent):
    """Formats OLAP query results into structured data."""

    # ── public methods ────────────────────────────────────────────────────────

    def formatted_table(
        self,
        data: list[dict],
        title: str = "",
    ) -> dict:
        """
        Return structured table metadata from *data*.

        Returns
        -------
        dict with keys: status, result, message
        """
        if not data:
            return {
                "status":  "ok",
                "rows":    [],
                "message": f"No data for {title}." if title else "No data.",
            }

        # Compute totals for summable columns
        columns = list(data[0].keys())
        totals: dict[str, float] = {}
        for col in columns:
            if _is_summable_col(col):
                try:
                    totals[col] = sum(
                        float(row[col])
                        for row in data
                        if row.get(col) is not None
                    )
                except (TypeError, ValueError):
                    pass

        return {
            "status":    "ok",
            "operation": "formatted_table",
            "title":     title,
            "rows":      _safe_json(data),
            "totals":    totals,
            "message":   f"{title} — {len(data)} row(s)." if title else f"{len(data)} row(s).",
        }

    def full_report(
        self,
        agent_results: list[dict],
    ) -> dict:
        """
        Combine multiple agent result dicts into one structured report.

        Returns
        -------
        dict with keys: status, result, message, sections
        """
        if not agent_results:
            return {
                "status":  "error",
                "rows":    [],
                "message": "No agent results provided.",
            }

        sections: list[dict] = []
        all_rows: list[dict] = []

        for idx, result in enumerate(agent_results, start=1):
            if result.get("status") == "error":
                sections.append({
                    "index": idx,
                    "title": "Error",
                    "message": result.get("message", "Unknown error"),
                    "rows": [],
                })
                continue

            rows = result.get("rows", [])
            title = result.get("title") or _operation_title(result)

            sections.append({
                "index": idx,
                "title": title,
                "operation": result.get("operation", ""),
                "message": result.get("message", ""),
                "rows": _safe_json(rows),
            })
            if isinstance(rows, list):
                all_rows.extend(rows)

        return {
            "status":        "ok",
            "operation":     "full_report",
            "all_rows":      _safe_json(all_rows),
            "sections":      sections,
            "section_count": len(sections),
            "message":       f"Report with {len(sections)} section(s), {len(all_rows)} total row(s).",
        }

    # ── run ───────────────────────────────────────────────────────────────────

    def run(self, query: dict) -> dict:
        """
        Dispatch to the appropriate method based on query["type"].
        """
        if not isinstance(query, dict):
            return {
                "status":  "error",
                "rows":    [],
                "message": "run() expects a dict with keys: type, data.",
            }

        report_type = query.get("type", "").lower()
        data        = query.get("data", [])
        title       = query.get("title", "")

        if report_type == "table":
            return self.formatted_table(data, title=title)

        if report_type == "report":
            return self.full_report(agent_results=data)

        return {
            "status":  "error",
            "rows":    [],
            "message": (
                f"Unknown type '{report_type}'. "
                "Valid options: 'table', 'report'."
            ),
        }


def _operation_title(result: dict) -> str:
    """Derive a human-friendly section title from an agent result dict."""
    op = result.get("operation", "result")

    label_map = {
        "yoy_growth":       lambda r: f"YoY {r.get('metric','').title()} Growth by {r.get('dimension','').title()}",
        "mom_change":       lambda r: f"MoM {r.get('metric','').title()} Change — {r.get('year','')}",
        "top_n":            lambda r: f"Top {r.get('n','')} {r.get('dimension','').title()}s by {r.get('metric','').title()}",
        "profit_margins":   lambda r: f"Profit Margins by {r.get('dimension','').title()}",
        "drill_down":       lambda r: f"Drill-Down: {r.get('dimension','').title()} {r.get('from_level','')} -> {r.get('to_level','')}",
        "roll_up":          lambda r: f"Roll-Up: {r.get('dimension','').title()} {r.get('from_level','')} -> {r.get('to_level','')}",
        "slice":            lambda r: f"Slice: {r.get('dimension','').title()} = {r.get('value','')}",
        "dice":             lambda r: "Dice: " + ", ".join(f"{k}={v}" for k, v in (r.get("filters") or {}).items()),
        "pivot":            lambda r: f"Pivot: {r.get('measure','').title()} by {r.get('row_dim','')} × {r.get('col_dim','')}",
    }

    fn = label_map.get(op)
    return fn(result) if fn else op.replace("_", " ").title()
