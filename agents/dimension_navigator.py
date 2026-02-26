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

    result = agent.run("drill down from year 2023 in the time dimension")
    print(result)
"""

from __future__ import annotations

import json
from typing import Any

import anthropic

from agents.base_agent import BaseAgent


# ── Dimension metadata ────────────────────────────────────────────────────────

# Each hierarchy is an ordered list of (level_name, table_column) tuples.
# level_name is what callers use; table_column is the actual SQL column.
_HIERARCHIES: dict[str, list[tuple[str, str]]] = {
    "time": [
        ("year",    "d.year"),
        ("quarter", "d.quarter"),
        ("month",   "d.month"),
    ],
    "geography": [
        ("region",  "g.region"),
        ("country", "g.country"),
    ],
    "product": [
        ("category",    "p.category"),
        ("subcategory", "p.subcategory"),
    ],
}

# SQL fragments to join each dimension table into fact_sales
_JOINS: dict[str, str] = {
    "time":      "JOIN dim_date      d ON f.date_key      = d.date_key",
    "geography": "JOIN dim_geography g ON f.geography_key = g.geography_key",
    "product":   "JOIN dim_product   p ON f.product_key   = p.product_key",
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
    ) -> dict:
        """
        Navigate one level deeper within *dimension*.

        Example: dimension="time", current_level="year", current_value=2024
                 → returns quarterly aggregates for 2024.

        Returns
        -------
        dict with keys:
            dimension, from_level, from_value, to_level, sql, rows
        """
        levels   = self._get_levels(dimension)
        from_idx = self._level_index(dimension, current_level)

        if from_idx == len(levels) - 1:
            return {
                "status":  "error",
                "message": f"'{current_level}' is already the lowest level of '{dimension}'.",
            }

        from_col  = levels[from_idx][1]
        to_level  = levels[from_idx + 1][0]
        to_col    = levels[from_idx + 1][1]
        join      = _JOINS[dimension]
        # Include all ancestor columns for context
        select_ancestors = ", ".join(col for _, col in levels[:from_idx + 2])

        sql = f"""
            SELECT
                {select_ancestors},
                SUM(f.revenue)      AS total_revenue,
                SUM(f.profit)       AS total_profit,
                SUM(f.quantity)     AS total_quantity,
                COUNT(f.order_id)   AS order_count
            FROM fact_sales f
            {join}
            WHERE {from_col} = '{current_value}'
            GROUP BY {select_ancestors}
            ORDER BY {to_col}
        """.strip()

        rows = self.execute_sql(sql)
        return {
            "status":     "ok",
            "operation":  "drill_down",
            "dimension":  dimension,
            "from_level": current_level,
            "from_value": current_value,
            "to_level":   to_level,
            "sql":        sql,
            "result":     rows,
            "message":    (
                f"Drilled down from {dimension}/{current_level}={current_value} "
                f"to {to_level} — {len(rows)} row(s) returned."
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

        Returns
        -------
        dict with keys:
            dimension, from_level, from_value, to_level, sql, rows
        """
        levels   = self._get_levels(dimension)
        from_idx = self._level_index(dimension, current_level)

        if from_idx == 0:
            return {
                "status":  "error",
                "message": f"'{current_level}' is already the highest level of '{dimension}'.",
            }

        from_col = levels[from_idx][1]
        to_level = levels[from_idx - 1][0]
        to_col   = levels[from_idx - 1][1]
        join     = _JOINS[dimension]
        # Select only the parent level
        select_parent = ", ".join(col for _, col in levels[:from_idx])

        sql = f"""
            SELECT
                {select_parent},
                SUM(f.revenue)      AS total_revenue,
                SUM(f.profit)       AS total_profit,
                SUM(f.quantity)     AS total_quantity,
                COUNT(f.order_id)   AS order_count
            FROM fact_sales f
            {join}
            WHERE {from_col} = '{current_value}'
            GROUP BY {select_parent}
            ORDER BY {to_col}
        """.strip()

        rows = self.execute_sql(sql)
        return {
            "status":     "ok",
            "operation":  "roll_up",
            "dimension":  dimension,
            "from_level": current_level,
            "from_value": current_value,
            "to_level":   to_level,
            "sql":        sql,
            "result":     rows,
            "message":    (
                f"Rolled up from {dimension}/{current_level}={current_value} "
                f"to {to_level} — {len(rows)} row(s) returned."
            ),
        }

    # ── run — NL interface via Claude ─────────────────────────────────────────

    def run(self, query: str) -> dict:
        """
        Parse a natural-language *query* with Claude and call the right method.

        Claude is given a tool-use schema that maps directly onto drill_down /
        roll_up so the response is always structured and executable.
        """
        tools = [
            {
                "name":        "drill_down",
                "description": (
                    "Navigate one level deeper in an OLAP dimension hierarchy. "
                    "Use when the user wants more detail, e.g. 'drill down from year 2023'."
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
                            "description": "The value at current_level to filter by (e.g. 2024, 'Q1', 'Electronics').",
                        },
                    },
                    "required": ["dimension", "current_level", "current_value"],
                },
            },
            {
                "name":        "roll_up",
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
                            "description": "The value at current_level to roll up from (e.g. 3, 'Germany', 'Laptops').",
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
            "appropriate tool. Always call exactly one tool — never answer in plain text."
        )

        response = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            system=system_prompt,
            tools=tools,
            messages=[{"role": "user", "content": query}],
        )

        # Extract the single tool-use block
        tool_use = next(
            (block for block in response.content if block.type == "tool_use"),
            None,
        )

        if tool_use is None:
            # Claude chose not to call a tool — surface its text response
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
        names  = [name for name, _ in levels]
        if level not in names:
            raise ValueError(
                f"Unknown level '{level}' for dimension '{dimension}'. "
                f"Valid levels: {names}"
            )
        return names.index(level)
