"""
CubeOperationsAgent — slice, dice, and pivot the OLAP sales cube.

Operations
──────────
  slice(dimension, value)       — filter one dimension, return aggregate totals
  dice(filters)                 — filter multiple dimensions simultaneously
  pivot(row_dim, col_dim, meas) — cross-tabulate two dimensions

Supported field names (usable as dimension / row_dim / col_dim)
───────────────────────────────────────────────────────────────
  Time        : year, quarter, month, month_name
  Geography   : region, country
  Product     : category, subcategory
  Customer    : customer_segment

Supported measures
──────────────────
  revenue, profit, cost, quantity, profit_margin, order_count

Usage
─────
    import anthropic, duckdb
    from agents.cube_operations import CubeOperationsAgent

    client = anthropic.Anthropic()
    con    = duckdb.connect("olap.duckdb")
    agent  = CubeOperationsAgent(client, con)

    result = agent.run("slice by region Europe")
    result = agent.run("dice year 2024 and category Electronics")
    result = agent.run("pivot revenue with region as rows and year as columns")
"""

from __future__ import annotations

from typing import Any

from agents.base_agent import BaseAgent


# ── Field → SQL mapping ───────────────────────────────────────────────────────

# Maps user-facing field name → (qualified_sql_col, join_clause)
_FIELD_MAP: dict[str, tuple[str, str]] = {
    "year":             ("d.year",             "JOIN dim_date      d ON f.date_key      = d.date_key"),
    "quarter":          ("d.quarter",          "JOIN dim_date      d ON f.date_key      = d.date_key"),
    "month":            ("d.month",            "JOIN dim_date      d ON f.date_key      = d.date_key"),
    "month_name":       ("d.month_name",       "JOIN dim_date      d ON f.date_key      = d.date_key"),
    "region":           ("g.region",           "JOIN dim_geography g ON f.geography_key = g.geography_key"),
    "country":          ("g.country",          "JOIN dim_geography g ON f.geography_key = g.geography_key"),
    "category":         ("p.category",         "JOIN dim_product   p ON f.product_key   = p.product_key"),
    "subcategory":      ("p.subcategory",      "JOIN dim_product   p ON f.product_key   = p.product_key"),
    "customer_segment": ("c.customer_segment", "JOIN dim_customer  c ON f.customer_key  = c.customer_key"),
}

# Fields whose values are integers — everything else is treated as a string
_NUMERIC_FIELDS: frozenset[str] = frozenset({"year", "month"})

# Maps measure name → (source column for inner SELECT, aggregation function)
_MEASURES: dict[str, tuple[str, str]] = {
    "revenue":       ("f.revenue",       "SUM"),
    "profit":        ("f.profit",        "SUM"),
    "cost":          ("f.cost",          "SUM"),
    "quantity":      ("f.quantity",      "SUM"),
    "profit_margin": ("f.profit_margin", "AVG"),
    "order_count":   ("f.order_id",      "COUNT"),
}

_VALID_FIELDS   = sorted(_FIELD_MAP)
_VALID_MEASURES = sorted(_MEASURES)


# ── Agent ─────────────────────────────────────────────────────────────────────

class CubeOperationsAgent(BaseAgent):
    """Handles slice, dice, and pivot operations on the OLAP sales cube."""

    # ── public cube operations ────────────────────────────────────────────────

    def slice(self, dimension: str, value: Any) -> dict:
        """
        Filter on a single *dimension* = *value* and return aggregate totals.

        Example: slice("region", "Europe")
                 → revenue/profit/quantity totals for European orders.
        """
        self._validate_field(dimension)
        col, join = _FIELD_MAP[dimension]
        where     = f"WHERE {col} = {self._quote(dimension, value)}"

        sql = f"""
SELECT
    SUM(f.revenue)      AS total_revenue,
    SUM(f.profit)       AS total_profit,
    SUM(f.cost)         AS total_cost,
    SUM(f.quantity)     AS total_quantity,
    COUNT(f.order_id)   AS order_count,
    ROUND(AVG(f.profit_margin), 4) AS avg_profit_margin
FROM fact_sales f
{join}
{where}
        """.strip()

        rows = self.execute_sql(sql)
        return {
            "status":    "ok",
            "operation": "slice",
            "dimension": dimension,
            "value":     value,
            "sql":       sql,
            "rows":      rows,
            "message":   (
                f"Slice on {dimension}={value!r} — "
                f"revenue={rows[0]['total_revenue']}, "
                f"profit={rows[0]['total_profit']}, "
                f"quantity={rows[0]['total_quantity']}."
            ),
        }

    def dice(self, filters: dict[str, Any]) -> dict:
        """
        Filter on multiple dimensions simultaneously and return aggregate totals.

        Example: dice({"year": 2024, "region": "Europe"})
                 → totals for European orders placed in 2024.

        Parameters
        ----------
        filters:
            Mapping of field name → value, e.g. {"year": 2024, "category": "Electronics"}.
        """
        if not filters:
            return {"status": "error", "message": "dice() requires at least one filter."}

        for field in filters:
            self._validate_field(field)

        # Collect unique JOINs (preserving insertion order)
        joins       = self._collect_joins(filters.keys())
        conditions  = [
            f"{_FIELD_MAP[field][0]} = {self._quote(field, val)}"
            for field, val in filters.items()
        ]
        where = "WHERE " + "\n  AND ".join(conditions)

        sql = f"""
SELECT
    SUM(f.revenue)      AS total_revenue,
    SUM(f.profit)       AS total_profit,
    SUM(f.cost)         AS total_cost,
    SUM(f.quantity)     AS total_quantity,
    COUNT(f.order_id)   AS order_count,
    ROUND(AVG(f.profit_margin), 4) AS avg_profit_margin
FROM fact_sales f
{joins}
{where}
        """.strip()

        rows     = self.execute_sql(sql)
        filter_s = ", ".join(f"{k}={v!r}" for k, v in filters.items())
        return {
            "status":    "ok",
            "operation": "dice",
            "filters":   filters,
            "sql":       sql,
            "rows":      rows,
            "message":   (
                f"Dice on [{filter_s}] — "
                f"revenue={rows[0]['total_revenue']}, "
                f"profit={rows[0]['total_profit']}, "
                f"quantity={rows[0]['total_quantity']}."
            ),
        }

    def pivot(self, row_dim: str, col_dim: str, measure: str) -> dict:
        """
        Cross-tabulate *row_dim* vs *col_dim*, aggregating *measure*.

        Example: pivot("region", "year", "revenue")
                 → table with one row per region, one column per year,
                   cells = SUM(revenue).

        Parameters
        ----------
        row_dim:
            Field that becomes the row labels.
        col_dim:
            Field whose distinct values become column headers.
        measure:
            Metric to aggregate in each cell.
        """
        self._validate_field(row_dim)
        self._validate_field(col_dim)
        self._validate_measure(measure)

        if row_dim == col_dim:
            return {"status": "error", "message": "row_dim and col_dim must be different fields."}

        row_col, row_join = _FIELD_MAP[row_dim]
        col_col, col_join = _FIELD_MAP[col_dim]
        fact_col, agg_fn  = _MEASURES[measure]

        joins = self._collect_joins([row_dim, col_dim])

        # DuckDB native PIVOT: inner CTE exposes one column per semantic role
        # so PIVOT ON / GROUP BY use clean names rather than qualified aliases.
        sql = f"""
WITH base AS (
    SELECT
        {row_col} AS {row_dim},
        {col_col} AS {col_dim},
        {fact_col} AS {measure}
    FROM fact_sales f
    {joins}
)
PIVOT base
ON  {col_dim}
USING {agg_fn}({measure})
GROUP BY {row_dim}
ORDER BY {row_dim}
        """.strip()

        rows = self.execute_sql(sql)
        # The number of pivot columns = total cols minus the row_dim column
        col_count = (len(rows[0]) - 1) if rows else 0
        return {
            "status":    "ok",
            "operation": "pivot",
            "row_dim":   row_dim,
            "col_dim":   col_dim,
            "measure":   measure,
            "sql":       sql,
            "rows":      rows,
            "message":   (
                f"Pivot of {measure} — rows={row_dim}, cols={col_dim} "
                f"({col_count} column(s)) — {len(rows)} row(s) returned."
            ),
        }

    # ── run — NL interface via Claude ─────────────────────────────────────────

    def run(self, query: str) -> dict:
        """
        Parse a natural-language *query* with Claude and dispatch to the
        appropriate cube operation via tool use.
        """
        tools = [
            {
                "name": "slice",
                "description": (
                    "Filter the sales cube on a single dimension value and return "
                    "aggregate totals (revenue, profit, cost, quantity, order_count). "
                    "Use when the user mentions filtering by one field, "
                    "e.g. 'slice by region Europe' or 'show me data for Electronics'."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "dimension": {
                            "type": "string",
                            "enum": _VALID_FIELDS,
                            "description": "The field to filter on.",
                        },
                        "value": {
                            "description": "The value to filter by (e.g. 'Europe', 2024, 'Electronics').",
                        },
                    },
                    "required": ["dimension", "value"],
                },
            },
            {
                "name": "dice",
                "description": (
                    "Filter the sales cube on multiple dimension values simultaneously "
                    "and return aggregate totals. Use when the user specifies more than "
                    "one filter, e.g. 'dice year 2024 and region Europe' or "
                    "'sales for Electronics in North America in 2023'."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "filters": {
                            "type": "object",
                            "description": (
                                "Map of field name to filter value. "
                                f"Valid keys: {_VALID_FIELDS}. "
                                "Example: {\"year\": 2024, \"region\": \"Europe\"}."
                            ),
                        },
                    },
                    "required": ["filters"],
                },
            },
            {
                "name": "pivot",
                "description": (
                    "Cross-tabulate two dimensions and aggregate a measure. "
                    "Use when the user asks for a pivot table or cross-tab, "
                    "e.g. 'pivot revenue by region and year' or "
                    "'show profit with category as rows and quarter as columns'."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "row_dim": {
                            "type": "string",
                            "enum": _VALID_FIELDS,
                            "description": "Field to use as row labels.",
                        },
                        "col_dim": {
                            "type": "string",
                            "enum": _VALID_FIELDS,
                            "description": "Field whose distinct values become column headers.",
                        },
                        "measure": {
                            "type": "string",
                            "enum": _VALID_MEASURES,
                            "description": "Metric to aggregate in each cell.",
                        },
                    },
                    "required": ["row_dim", "col_dim", "measure"],
                },
            },
        ]

        system_prompt = (
            "You are an OLAP cube assistant. "
            "Given a natural-language query, call exactly one of the three tools: "
            "slice (one filter), dice (multiple filters), or pivot (cross-tabulation). "
            "Never respond in plain text — always use a tool."
        )

        response = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            system=system_prompt,
            tools=tools,
            messages=[{"role": "user", "content": query}],
        )

        tool_use = next(
            (block for block in response.content if block.type == "tool_use"),
            None,
        )

        if tool_use is None:
            text = " ".join(
                block.text for block in response.content if hasattr(block, "text")
            )
            return {"status": "error", "message": text or "No tool call was made."}

        args = tool_use.input
        if tool_use.name == "slice":
            return self.slice(dimension=args["dimension"], value=args["value"])
        if tool_use.name == "dice":
            return self.dice(filters=args["filters"])
        if tool_use.name == "pivot":
            return self.pivot(
                row_dim=args["row_dim"],
                col_dim=args["col_dim"],
                measure=args["measure"],
            )

        return {"status": "error", "message": f"Unknown tool: {tool_use.name}"}

    # ── private helpers ───────────────────────────────────────────────────────

    def _validate_field(self, field: str) -> None:
        if field not in _FIELD_MAP:
            raise ValueError(
                f"Unknown field '{field}'. Valid fields: {_VALID_FIELDS}"
            )

    def _validate_measure(self, measure: str) -> None:
        if measure not in _MEASURES:
            raise ValueError(
                f"Unknown measure '{measure}'. Valid measures: {_VALID_MEASURES}"
            )

    def _quote(self, field: str, value: Any) -> str:
        """Return a SQL-safe literal for *value*, quoting strings."""
        if field in _NUMERIC_FIELDS:
            return str(int(value))
        # Escape any single quotes in string values
        return "'" + str(value).replace("'", "''") + "'"

    def _collect_joins(self, fields: Any) -> str:
        """Return deduplicated JOIN clauses for the given *fields*, preserving order."""
        seen:   set[str]  = set()
        result: list[str] = []
        for field in fields:
            _, join = _FIELD_MAP[field]
            if join not in seen:
                seen.add(join)
                result.append(join)
        return "\n".join(result)
