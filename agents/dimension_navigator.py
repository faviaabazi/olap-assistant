"""
DimensionNavigatorAgent — drill-down and roll-up across OLAP dimensions.

Supported dimensions and their level hierarchies
─────────────────────────────────────────────────
  time       :  year  →  quarter  →  month
  geography  :  region  →  country
  product    :  category  →  subcategory

Usage
─────
    import anthropic, duckdb
    from agents.dimension_navigator import DimensionNavigatorAgent

    client = anthropic.Anthropic()
    con    = duckdb.connect("olap.duckdb")
    agent  = DimensionNavigatorAgent(client, con)

    result = agent.drill_down("time", "year", 2024)
    result = agent.drill_down("time", "quarter", 4)       # Q4 → months
    result = agent.drill_down("time", "quarter", 4, year=2024)  # Q4 2024 → months
"""

from __future__ import annotations

import json
from typing import Any

import anthropic

from agents.base_agent import BaseAgent

# ── Dimension metadata ────────────────────────────────────────────────────────

_HIERARCHIES: dict[str, list[tuple[str, str]]] = {
    "time": [
        ("year", "d.year"),
        ("quarter", "d.quarter"),
        ("month", "d.month"),
    ],
    "geography": [
        ("region", "g.region"),
        ("country", "g.country"),
    ],
    "product": [
        ("category", "p.category"),
        ("subcategory", "p.subcategory"),
    ],
}

_JOINS: dict[str, str] = {
    "time": "JOIN dim_date      d ON f.date_key      = d.date_key",
    "geography": "JOIN dim_geography g ON f.geography_key = g.geography_key",
    "product": "JOIN dim_product   p ON f.product_key   = p.product_key",
}

# Quarter → month numbers for validation / display
_QUARTER_MONTHS: dict[int, tuple[int, int, int]] = {
    1: (1, 2, 3),
    2: (4, 5, 6),
    3: (7, 8, 9),
    4: (10, 11, 12),
}


# ── Agent ─────────────────────────────────────────────────────────────────────


class DimensionNavigatorAgent(BaseAgent):
    """Handles drill-down and roll-up navigation across OLAP dimensions."""

    # ── public navigation methods ─────────────────────────────────────────────

    def drill_down(
        self,
        dimension: str,
        current_level: str,
        current_value: Any,
        year: int | None = None,
    ) -> dict:
        """
        Navigate one level deeper within *dimension*.

        For time/quarter → month: aggregates revenue/profit/quantity by month
        for only the months that belong to that quarter (e.g. Q4 = Oct/Nov/Dec).
        Pass *year* to restrict to a specific year.

        Examples
        --------
        drill_down("time", "year", 2024)
            → quarterly aggregates for 2024

        drill_down("time", "quarter", 4)
            → monthly aggregates for Q4 (Oct, Nov, Dec) across all years

        drill_down("time", "quarter", 4, year=2024)
            → monthly aggregates for Q4 2024 specifically
        """
        levels = self._get_levels(dimension)
        from_idx = self._level_index(dimension, current_level)

        if from_idx == len(levels) - 1:
            return {
                "status": "error",
                "message": f"'{current_level}' is already the lowest level of '{dimension}'.",
            }

        to_level = levels[from_idx + 1][0]

        # ── Special case: time/quarter → month ────────────────────────────────
        # Standard drill_down would filter WHERE quarter = 'value' (string),
        # which fails on an integer column. We also need to show month names
        # and restrict to only the 3 months in that quarter.
        if dimension == "time" and current_level == "quarter":
            return self._drill_quarter_to_months(current_value, year)

        # ── General case ──────────────────────────────────────────────────────
        from_col = levels[from_idx][1]
        to_col = levels[from_idx + 1][1]
        join = _JOINS[dimension]
        select_ancestors = ", ".join(col for _, col in levels[: from_idx + 2])

        # Quote value: integers for numeric time fields, strings otherwise
        _NUMERIC_LEVELS = {"year", "quarter", "month"}
        if current_level in _NUMERIC_LEVELS:
            quoted = str(int(current_value))
        else:
            quoted = "'" + str(current_value).replace("'", "''") + "'"

        # Optional year filter for time dimension (year→quarter drill)
        year_clause = ""
        if dimension == "time" and year is not None and current_level != "year":
            year_clause = f"AND d.year = {int(year)}"

        sql = f"""
SELECT
    {select_ancestors},
    SUM(f.revenue)      AS total_revenue,
    SUM(f.profit)       AS total_profit,
    SUM(f.quantity)     AS total_quantity,
    COUNT(f.order_id)   AS order_count
FROM fact_sales f
{join}
WHERE {from_col} = {quoted}
{year_clause}
GROUP BY {select_ancestors}
ORDER BY {levels[from_idx + 1][1]}
        """.strip()

        rows = self.execute_sql(sql)
        return {
            "status": "ok",
            "operation": "drill_down",
            "dimension": dimension,
            "from_level": current_level,
            "from_value": current_value,
            "to_level": to_level,
            "sql": sql,
            "rows": rows,
            "message": (
                f"Drilled down from {dimension}/{current_level}={current_value} "
                f"to {to_level} — {len(rows)} row(s) returned."
            ),
        }

    def _drill_quarter_to_months(
        self,
        quarter_value: Any,
        year: int | None = None,
    ) -> dict:
        """
        Aggregate revenue/profit/quantity by month for the given quarter.

        Only the 3 months belonging to that quarter are returned, in month order.
        Optionally restricted to a specific year.

        Q1 → Jan (1), Feb (2), Mar (3)
        Q2 → Apr (4), May (5), Jun (6)
        Q3 → Jul (7), Aug (8), Sep (9)
        Q4 → Oct (10), Nov (11), Dec (12)
        """
        quarter_int = self._coerce_quarter(quarter_value)
        if quarter_int not in _QUARTER_MONTHS:
            return {
                "status": "error",
                "message": f"Invalid quarter '{quarter_value}'. Expected 1–4.",
            }

        months = _QUARTER_MONTHS[quarter_int]
        month_list = ", ".join(str(m) for m in months)

        # Year filter is additive — narrow to specific year if provided
        year_clause = f"AND d.year = {int(year)}" if year is not None else ""

        sql = f"""
SELECT
    d.year,
    d.quarter,
    d.month,
    d.month_name,
    SUM(f.revenue)      AS total_revenue,
    SUM(f.profit)       AS total_profit,
    SUM(f.quantity)     AS total_quantity,
    COUNT(f.order_id)   AS order_count
FROM fact_sales f
JOIN dim_date d ON f.date_key = d.date_key
WHERE d.quarter = {quarter_int}
  AND d.month IN ({month_list})
  {year_clause}
GROUP BY d.year, d.quarter, d.month, d.month_name
ORDER BY d.year, d.month
        """.strip()

        rows = self.execute_sql(sql)
        year_label = f" {year}" if year is not None else ""
        return {
            "status": "ok",
            "operation": "drill_down",
            "dimension": "time",
            "from_level": "quarter",
            "from_value": quarter_int,
            "to_level": "month",
            "sql": sql,
            "rows": rows,
            "message": (
                f"Q{quarter_int}{year_label} drilled down to monthly breakdown "
                f"— {len(rows)} month(s) returned."
            ),
        }

    def roll_up(
        self,
        dimension: str,
        current_level: str,
        current_value: Any,
    ) -> dict:
        """
        Navigate one level higher within *dimension*.

        Example: dimension="time", current_level="month", current_value=3
                 → returns the quarter that contains month 3.
        """
        levels = self._get_levels(dimension)
        from_idx = self._level_index(dimension, current_level)

        if from_idx == 0:
            return {
                "status": "error",
                "message": f"'{current_level}' is already the highest level of '{dimension}'.",
            }

        from_col = levels[from_idx][1]
        to_level = levels[from_idx - 1][0]
        join = _JOINS[dimension]
        select_parent = ", ".join(col for _, col in levels[:from_idx])

        _NUMERIC_LEVELS = {"year", "quarter", "month"}
        if current_level in _NUMERIC_LEVELS:
            quoted = str(int(current_value))
        else:
            quoted = "'" + str(current_value).replace("'", "''") + "'"

        sql = f"""
SELECT
    {select_parent},
    SUM(f.revenue)      AS total_revenue,
    SUM(f.profit)       AS total_profit,
    SUM(f.quantity)     AS total_quantity,
    COUNT(f.order_id)   AS order_count
FROM fact_sales f
{join}
WHERE {from_col} = {quoted}
GROUP BY {select_parent}
ORDER BY total_revenue DESC
        """.strip()

        rows = self.execute_sql(sql)
        return {
            "status": "ok",
            "operation": "roll_up",
            "dimension": dimension,
            "from_level": current_level,
            "from_value": current_value,
            "to_level": to_level,
            "sql": sql,
            "rows": rows,
            "message": (
                f"Rolled up from {dimension}/{current_level}={current_value} "
                f"to {to_level} — {len(rows)} row(s) returned."
            ),
        }

    # ── run — NL interface via Claude ─────────────────────────────────────────

    def run(self, query: str) -> dict:
        tools = [
            {
                "name": "drill_down",
                "description": (
                    "Navigate one level deeper in an OLAP dimension hierarchy. "
                    "Use when the user wants more detail, e.g. 'drill down from year 2023', "
                    "'drill Q4 into months', 'break Q3 2024 into months'."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "dimension": {
                            "type": "string",
                            "enum": ["time", "geography", "product"],
                            "description": "The hierarchy to navigate.",
                        },
                        "current_level": {
                            "type": "string",
                            "description": (
                                "The current level: year|quarter|month for time; "
                                "region|country for geography; "
                                "category|subcategory for product."
                            ),
                        },
                        "current_value": {
                            "description": (
                                "The value at current_level to filter by. "
                                "For quarter: pass an INTEGER 1-4 (Q1=1, Q2=2, Q3=3, Q4=4). "
                                "For year: pass the integer year (e.g. 2024). "
                                "For geography/product: pass the string value."
                            ),
                        },
                        "year": {
                            "type": "integer",
                            "description": (
                                "Optional year to restrict the quarter→month drill to a specific year. "
                                "E.g. for 'Q4 2024 into months', pass current_value=4, year=2024."
                            ),
                        },
                    },
                    "required": ["dimension", "current_level", "current_value"],
                },
            },
            {
                "name": "roll_up",
                "description": (
                    "Navigate one level higher in an OLAP dimension hierarchy. "
                    "Use when the user wants less detail, e.g. 'roll up from month 3 to quarter'."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "dimension": {
                            "type": "string",
                            "enum": ["time", "geography", "product"],
                        },
                        "current_level": {
                            "type": "string",
                        },
                        "current_value": {
                            "description": "The value at current_level to roll up from.",
                        },
                    },
                    "required": ["dimension", "current_level", "current_value"],
                },
            },
        ]

        system_prompt = (
            "You are an OLAP navigation assistant. "
            "When the user asks to drill down or roll up, extract the dimension, "
            "current level, and current value from their query and call the "
            "appropriate tool. Always call exactly one tool — never answer in plain text. "
            "IMPORTANT: for quarter values always pass an integer (Q1=1, Q2=2, Q3=3, Q4=4). "
            "When drilling a quarter into months and a year is mentioned, also pass the year param."
        )

        response = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            system=system_prompt,
            tools=tools,
            messages=[{"role": "user", "content": query}],
        )

        tool_use = next(
            (block for block in response.content if block.type == "tool_use"), None
        )

        if tool_use is None:
            text = " ".join(
                block.text for block in response.content if hasattr(block, "text")
            )
            return {"status": "error", "message": text or "No tool call was made."}

        args = tool_use.input
        if tool_use.name == "drill_down":
            return self.drill_down(
                dimension=args["dimension"],
                current_level=args["current_level"],
                current_value=args["current_value"],
                year=args.get("year"),
            )
        if tool_use.name == "roll_up":
            return self.roll_up(
                dimension=args["dimension"],
                current_level=args["current_level"],
                current_value=args["current_value"],
            )

        return {"status": "error", "message": f"Unknown tool: {tool_use.name}"}

    # ── private helpers ───────────────────────────────────────────────────────

    def _get_levels(self, dimension: str) -> list[tuple[str, str]]:
        if dimension not in _HIERARCHIES:
            raise ValueError(
                f"Unknown dimension '{dimension}'. "
                f"Valid options: {list(_HIERARCHIES.keys())}"
            )
        return _HIERARCHIES[dimension]

    def _level_index(self, dimension: str, level: str) -> int:
        levels = self._get_levels(dimension)
        names = [name for name, _ in levels]
        if level not in names:
            raise ValueError(
                f"Unknown level '{level}' for dimension '{dimension}'. "
                f"Valid levels: {names}"
            )
        return names.index(level)

    def _coerce_quarter(self, value: Any) -> int:
        """Convert any quarter representation to an integer 1-4."""
        if isinstance(value, int) and 1 <= value <= 4:
            return value
        s = str(value).strip().upper().lstrip("Q").strip()
        try:
            n = int(s)
            if 1 <= n <= 4:
                return n
        except ValueError:
            pass
        raise ValueError(f"Invalid quarter value '{value}'. Expected 1-4 or Q1-Q4.")
