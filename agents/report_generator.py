"""
ReportGeneratorAgent — formats OLAP results into tables, narratives, and reports.

Operations
──────────
  formatted_table(data, title)              — plain-text table with optional totals row
  executive_summary(data, context)          — 3-5 sentence business narrative via Claude
  full_report(agent_results)                — multi-section report combining multiple results

run(query) interface
────────────────────
  query must be a dict with:
    "type"    : "table" | "summary" | "report"
    "data"    : list[dict]  (for table/summary)  or  list[dict]  (agent results for report)
    "title"   : str  (optional, used by table and report sections)
    "context" : str  (optional, passed to executive_summary for framing)

Usage
─────
    import anthropic, duckdb
    from agents.report_generator import ReportGeneratorAgent

    client = anthropic.Anthropic()
    con    = duckdb.connect("olap.duckdb")
    agent  = ReportGeneratorAgent(client, con)

    table = agent.formatted_table(rows, title="Revenue by Region")
    summ  = agent.executive_summary(rows, context="YoY revenue growth by region")
    rpt   = agent.full_report([result1, result2, result3])
"""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Any

from agents.base_agent import BaseAgent


# ── Formatting helpers ────────────────────────────────────────────────────────

# Columns whose values should be right-aligned (numbers)
_NUMERIC_SUFFIXES = (
    "revenue", "profit", "cost", "quantity", "count",
    "pct", "margin", "price", "total", "growth", "change",
)

# Columns that should have a SUM totals row (additive measures only)
_SUMMABLE_SUFFIXES = (
    "revenue", "profit", "cost", "quantity", "count",
)

# Max characters a single cell value is allowed before being truncated
_MAX_CELL_WIDTH = 30


def _is_numeric_col(col: str) -> bool:
    # Also catches pivot columns whose headers are raw year numbers (e.g. "2022")
    return col.lower().endswith(_NUMERIC_SUFFIXES) or col.lstrip("-").isdigit()


def _is_summable_col(col: str) -> bool:
    return col.lower().endswith(_SUMMABLE_SUFFIXES)


def _fmt_value(value: Any, col: str = "") -> str:
    """Format a cell value for display."""
    if value is None:
        return "—"
    if isinstance(value, Decimal):
        # Show integers without decimals; otherwise 2 dp
        if value == value.to_integral_value():
            return f"{int(value):,}"
        return f"{float(value):,.2f}"
    if isinstance(value, float):
        if value != value:          # NaN
            return "—"
        return f"{value:,.2f}"
    if isinstance(value, int):
        # Year column: suppress thousands separator (2022 not 2,022)
        if col == "year":
            return str(value)
        return f"{value:,}"
    s = str(value)
    if len(s) > _MAX_CELL_WIDTH:
        return s[: _MAX_CELL_WIDTH - 1] + "…"
    return s


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
    """Formats OLAP query results into readable tables, narratives, and reports."""

    # ── public methods ────────────────────────────────────────────────────────

    def formatted_table(
        self,
        data: list[dict],
        title: str = "",
    ) -> dict:
        """
        Render *data* as a plain-text table with column alignment and an
        optional totals row for summable numeric columns.

        Returns
        -------
        dict with keys: status, operation, title, text, row_count
        """
        if not data:
            text = f"  {title}\n  (no data)" if title else "  (no data)"
            return {
                "status":    "ok",
                "operation": "formatted_table",
                "title":     title,
                "text":      text,
                "row_count": 0,
            }

        columns = list(data[0].keys())

        # ── build formatted cell matrix ────────────────────────────────────
        header_row = [col.replace("_", " ").title() for col in columns]
        body_rows  = [[_fmt_value(row.get(col), col) for col in columns] for row in data]

        # ── totals row ─────────────────────────────────────────────────────
        totals_row: list[str] | None = None
        summable_indices = [
            i for i, col in enumerate(columns) if _is_summable_col(col)
        ]
        if summable_indices:
            totals: list[str] = []
            first_non_sum = True
            for i, col in enumerate(columns):
                if i in summable_indices:
                    try:
                        col_sum = sum(
                            float(row[col])
                            for row in data
                            if row.get(col) is not None
                        )
                        totals.append(_fmt_value(Decimal(str(col_sum)).quantize(Decimal("0.01"))))
                    except (TypeError, ValueError):
                        totals.append("—")
                elif first_non_sum:
                    totals.append("TOTAL")
                    first_non_sum = False
                else:
                    totals.append("")
            totals_row = totals

        # ── compute column widths ──────────────────────────────────────────
        all_rows = [header_row] + body_rows + ([totals_row] if totals_row else [])
        col_widths = [
            max(len(str(row[i])) for row in all_rows)
            for i in range(len(columns))
        ]

        # ── render ─────────────────────────────────────────────────────────
        def render_row(cells: list[str], separator: str = " ") -> str:
            parts = []
            for i, cell in enumerate(cells):
                w   = col_widths[i]
                pad = cell.rjust(w) if _is_numeric_col(columns[i]) else cell.ljust(w)
                parts.append(pad)
            return separator.join(parts)

        divider      = "-+-".join("-" * w for w in col_widths)
        total_width  = sum(col_widths) + 3 * (len(col_widths) - 1)
        header_line  = render_row(header_row)

        lines: list[str] = []
        if title:
            lines += [title.upper(), "=" * max(len(title), total_width), ""]
        lines += [header_line, divider]
        lines += [render_row(r) for r in body_rows]
        if totals_row:
            lines += [divider, render_row(totals_row)]

        text = "\n".join(lines)
        return {
            "status":    "ok",
            "operation": "formatted_table",
            "title":     title,
            "text":      text,
            "row_count": len(data),
        }

    def executive_summary(
        self,
        data: list[dict],
        context: str = "",
    ) -> dict:
        """
        Use Claude to write a 3-5 sentence business narrative from *data*.

        Parameters
        ----------
        data:
            The query result rows to summarise.
        context:
            A short description of what the data represents, e.g.
            "Year-over-year revenue growth by region, 2022-2024."

        Returns
        -------
        dict with keys: status, operation, context, summary (the narrative text)
        """
        if not data:
            return {
                "status":    "error",
                "operation": "executive_summary",
                "message":   "No data provided to summarise.",
            }

        data_json = json.dumps(_safe_json(data), indent=2)

        prompt = (
            f"You are a senior business analyst writing for a C-suite audience.\n\n"
            f"Context: {context or 'OLAP sales data analysis'}\n\n"
            f"Data:\n{data_json}\n\n"
            "Write a concise executive summary of 3 to 5 sentences. "
            "Focus on the most important trends, standout values, and "
            "key findings. Use specific numbers. "
            "Do not use bullet points or headers — plain prose only. "
            "Report findings and trends only. Never suggest "
            "business actions, recommendations, or strategies. Neutral "
            "analyst tone."
        )

        response = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )

        summary = response.content[0].text.strip()
        return {
            "status":    "ok",
            "operation": "executive_summary",
            "context":   context,
            "summary":   summary,
        }

    def full_report(
        self,
        agent_results: list[dict],
    ) -> dict:
        """
        Combine multiple agent result dicts into one structured text report.

        Each element of *agent_results* should be a dict as returned by any
        agent method (must contain at least "operation" and "rows" or "summary").
        An optional "title" key overrides the auto-generated section heading.

        Structure of the output report
        ────────────────────────────────
          OLAP SALES REPORT
          ═══════════════════
          [timestamp]

          SECTION 1 — <operation>
          ────────────────────────
          <formatted table>

          ...

          EXECUTIVE SUMMARY
          ──────────────────
          <narrative for every section combined>

        Returns
        -------
        dict with keys: status, operation, report (full text), section_count
        """
        if not agent_results:
            return {
                "status":  "error",
                "message": "No agent results provided.",
            }

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        sections: list[str] = []
        all_data: list[dict] = []     # accumulated for the overall summary

        for idx, result in enumerate(agent_results, start=1):
            if result.get("status") == "error":
                heading = f"SECTION {idx} — ERROR"
                body    = f"  {result.get('message', 'Unknown error')}"
                sections.append(f"{heading}\n{'─' * len(heading)}\n{body}")
                continue

            operation = result.get("operation", f"result_{idx}")
            rows      = result.get("rows", [])
            title     = result.get("title") or _operation_title(result)

            heading = f"SECTION {idx} — {title.upper()}"
            rule    = "─" * len(heading)

            # Render the data table
            table_result = self.formatted_table(rows, title="")
            table_text   = table_result["text"].rstrip()

            # Include message as a sub-heading if present
            message = result.get("message", "")
            body_parts = []
            if message:
                body_parts.append(f"  {message}")
            body_parts.append(table_text)

            sections.append(f"{heading}\n{rule}\n" + "\n".join(body_parts))
            all_data.extend(rows)

        # ── executive summary across all sections ──────────────────────────
        context_parts = []
        for r in agent_results:
            if r.get("status") == "ok" and r.get("message"):
                context_parts.append(r["message"])
        combined_context = (
            "Multi-section OLAP sales report covering: "
            + "; ".join(context_parts)
            if context_parts
            else "OLAP sales report"
        )

        summary_result = self.executive_summary(
            data=all_data[:50],       # cap tokens — first 50 rows representative
            context=combined_context,
        )
        summary_text = summary_result.get("summary", "(summary unavailable)")

        # ── assemble final report ──────────────────────────────────────────
        width = 72
        banner = [
            "OLAP SALES REPORT",
            "=" * width,
            f"Generated: {timestamp}",
            f"Sections:  {len(sections)}",
            "",
        ]

        exec_heading = "EXECUTIVE SUMMARY"
        exec_block   = [
            "",
            exec_heading,
            "─" * len(exec_heading),
            summary_text,
        ]

        report = "\n".join(banner) + "\n\n" + "\n\n".join(sections) + "\n" + "\n".join(exec_block)

        return {
            "status":        "ok",
            "operation":     "full_report",
            "report":        report,
            "section_count": len(sections),
        }

    # ── run ───────────────────────────────────────────────────────────────────

    def run(self, query: dict) -> dict:
        """
        Dispatch to the appropriate method based on query["type"].

        Parameters
        ----------
        query : dict
            {
                "type":    "table" | "summary" | "report",
                "data":    list[dict],        # rows for table/summary; agent results for report
                "title":   str,               # optional
                "context": str,               # optional, used by summary
            }
        """
        if not isinstance(query, dict):
            return {
                "status":  "error",
                "message": "run() expects a dict with keys: type, data, title, context.",
            }

        report_type = query.get("type", "").lower()
        data        = query.get("data", [])
        title       = query.get("title", "")
        context     = query.get("context", "")

        if report_type == "table":
            return self.formatted_table(data, title=title)

        if report_type == "summary":
            return self.executive_summary(data, context=context)

        if report_type == "report":
            # data is expected to be a list of agent result dicts
            return self.full_report(agent_results=data)

        return {
            "status":  "error",
            "message": (
                f"Unknown type '{report_type}'. "
                "Valid options: 'table', 'summary', 'report'."
            ),
        }

    # ── private helpers ───────────────────────────────────────────────────────

    def _noop(self, *_, **__) -> None:
        """Satisfy BaseAgent's execute_sql requirement — not used by this agent."""


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
