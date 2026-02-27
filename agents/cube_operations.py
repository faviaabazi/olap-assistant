"""
CubeOperationsAgent — slice, dice, and pivot the OLAP sales cube.

Operations
──────────
  slice(dimension, value, summarize)  — filter one dimension
                                         summarize=False (default): individual order rows
                                         summarize=True: grouped aggregate totals
  dice(filters, summarize)            — filter multiple dimensions
                                         summarize=False (default): individual order rows
                                         summarize=True: grouped aggregate totals
  pivot(row_dim, col_dim, meas)       — cross-tabulate two dimensions

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

    # Row-level (default)
    result = agent.slice("year", 2024)
    result = agent.dice({"country": "Italy", "year": 2024})

    # Grouped summary (explicit)
    result = agent.slice("year", 2024, summarize=True)
    result = agent.dice({"region": "Europe"}, summarize=True)

    # Pivot always aggregates
    result = agent.pivot("region", "year", "revenue")
"""

from __future__ import annotations

import re
from typing import Any

from agents.base_agent import BaseAgent

# ── Field → SQL mapping ───────────────────────────────────────────────────────

# Maps user-facing field name → (qualified_sql_col, join_clause)
_FIELD_MAP: dict[str, tuple[str, str]] = {
    "year": ("d.year", "JOIN dim_date      d ON f.date_key      = d.date_key"),
    "quarter": ("d.quarter", "JOIN dim_date      d ON f.date_key      = d.date_key"),
    "month": ("d.month", "JOIN dim_date      d ON f.date_key      = d.date_key"),
    "month_name": (
        "d.month_name",
        "JOIN dim_date      d ON f.date_key      = d.date_key",
    ),
    "region": ("g.region", "JOIN dim_geography g ON f.geography_key = g.geography_key"),
    "country": (
        "g.country",
        "JOIN dim_geography g ON f.geography_key = g.geography_key",
    ),
    "category": (
        "p.category",
        "JOIN dim_product   p ON f.product_key   = p.product_key",
    ),
    "subcategory": (
        "p.subcategory",
        "JOIN dim_product   p ON f.product_key   = p.product_key",
    ),
    "customer_segment": (
        "c.customer_segment",
        "JOIN dim_customer  c ON f.customer_key  = c.customer_key",
    ),
}

# Fields whose values are integers — everything else is treated as a string
_NUMERIC_FIELDS: frozenset[str] = frozenset({"year", "month", "quarter"})

# Maps measure name → (source column for inner SELECT, aggregation function)
_MEASURES: dict[str, tuple[str, str]] = {
    "revenue": ("f.revenue", "SUM"),
    "profit": ("f.profit", "SUM"),
    "cost": ("f.cost", "SUM"),
    "quantity": ("f.quantity", "SUM"),
    "profit_margin": ("f.profit_margin", "AVG"),
    "order_count": ("f.order_id", "COUNT"),
}

_VALID_FIELDS = sorted(_FIELD_MAP)
_VALID_MEASURES = sorted(_MEASURES)

# Maps measure name → raw fact column for row-level WHERE filtering
_MEASURE_FILTER_MAP: dict[str, str] = {
    "revenue":       "f.revenue",
    "profit":        "f.profit",
    "cost":          "f.cost",
    "quantity":      "f.quantity",
    "profit_margin": "f.profit_margin",
}

# When slicing on a dimension and summarize=True, show breakdown by this secondary dimension
_SECONDARY_DIM: dict[str, str] = {
    "year": "category",
    "quarter": "category",
    "month": "category",
    "month_name": "category",
    "region": "category",
    "country": "category",
    "category": "region",
    "subcategory": "region",
    "customer_segment": "category",
}

# All joins required to hydrate every dimension field on order rows
_ALL_DIMENSION_JOINS = (
    "JOIN dim_date      d ON f.date_key      = d.date_key\n"
    "JOIN dim_geography g ON f.geography_key = g.geography_key\n"
    "JOIN dim_product   p ON f.product_key   = p.product_key\n"
    "JOIN dim_customer  c ON f.customer_key  = c.customer_key"
)


# ── Agent ─────────────────────────────────────────────────────────────────────


class CubeOperationsAgent(BaseAgent):
    """Handles slice, dice, and pivot operations on the OLAP sales cube."""

    # ── public cube operations ────────────────────────────────────────────────

    def slice(self, dimension: str, value: Any, summarize: bool = False) -> dict:
        """
        Filter on a single *dimension* = *value*.

        Parameters
        ----------
        dimension:
            Field to filter on (e.g. "year", "region", "category").
        value:
            The value to filter by (e.g. 2024, "Europe", "Electronics").
        summarize:
            False (default) — return individual order rows with all fields.
            True            — return grouped aggregate totals by secondary dimension.

        Examples
        --------
        slice("year", 2024)               → all orders placed in 2024
        slice("year", 2024, summarize=True) → per-category totals for 2024
        """
        self._validate_field(dimension)

        if summarize:
            return self._slice_summary(dimension, value)
        return self._list_orders({dimension: value})

    def dice(self, filters: dict[str, Any], summarize: bool = False) -> dict:
        """
        Filter on multiple dimensions and/or measure thresholds simultaneously.

        Dimension fields go to WHERE clause.
        Measure fields (revenue, profit, etc.) go to HAVING clause.
        Measure values support operators: ">500", ">=1000", "<200", "=500"

        Parameters
        ----------
        filters:
            Mapping of field name → value for dimensions AND/OR measure thresholds.
            Dimension fields: year, quarter, month, region, country, etc.
            Measure fields: revenue, profit, cost, quantity, order_count, profit_margin
                — pass as operator strings: {"revenue": ">500"} or {"profit": ">=1000"}
        summarize:
            False (default) — return individual order rows with all fields.
            True            — return grouped aggregate totals.

        Examples
        --------
        dice({"country": "Italy", "year": 2024})
            → all orders from Italy in 2024

        dice({"region": "Asia Pacific", "year": 2023, "revenue": ">500"})
            → orders from Asia Pacific in 2023 with revenue > 500

        dice({"region": "Europe"}, summarize=True)
            → per-category revenue/profit totals for Europe
        """
        if not filters:
            return {
                "status": "error",
                "message": "dice() requires at least one filter.",
            }

        # Separate dimension filters from measure filters
        dim_filters     = {k: v for k, v in filters.items() if k in _FIELD_MAP}
        measure_filters = {k: v for k, v in filters.items() if k in _MEASURE_FILTER_MAP}
        unknown         = set(filters) - set(dim_filters) - set(measure_filters)
        if unknown:
            return {
                "status": "error",
                "message": (
                    f"Unknown filter field(s): {unknown}. "
                    f"Valid dimension fields: {_VALID_FIELDS}. "
                    f"Valid measure fields: {list(_MEASURE_FILTER_MAP)}."
                ),
            }

        for field in dim_filters:
            self._validate_field(field)

        if summarize and not measure_filters:
            return self._dice_summary(dim_filters)
        return self._list_orders(dim_filters, measure_filters)

    def pivot(self, row_dim: str, col_dim: str, measure: str) -> dict:
        """
        Cross-tabulate *row_dim* vs *col_dim*, aggregating *measure*.

        Always returns aggregated data — there is no row-level mode for pivot.

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
            return {
                "status": "error",
                "message": "row_dim and col_dim must be different fields.",
            }

        row_col, _ = _FIELD_MAP[row_dim]
        col_col, _ = _FIELD_MAP[col_dim]
        fact_col, agg_fn = _MEASURES[measure]

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
        col_count = (len(rows[0]) - 1) if rows else 0
        return {
            "status": "ok",
            "operation": "pivot",
            "row_dim": row_dim,
            "col_dim": col_dim,
            "measure": measure,
            "sql": sql,
            "rows": rows,
            "message": (
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
                    "Filter the sales cube on a single dimension value. "
                    "By default returns individual order rows (one per transaction). "
                    "Pass summarize=true ONLY when the user explicitly asks for totals, "
                    "a breakdown, or a summary — e.g. 'show me a 2024 summary by category'. "
                    "Use for queries like 'list 2024 orders', 'show Electronics orders', "
                    "'filter by region Europe'."
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
                        "summarize": {
                            "type": "boolean",
                            "description": (
                                "False (default): return individual order rows. "
                                "True: return grouped aggregate totals by secondary dimension. "
                                "Only set to true when user explicitly asks for totals/summary/breakdown."
                            ),
                            "default": False,
                        },
                    },
                    "required": ["dimension", "value"],
                },
            },
            {
                "name": "dice",
                "description": (
                    "Filter the sales cube on multiple dimension values simultaneously. "
                    "By default returns individual order rows (one per transaction). "
                    "Pass summarize=true ONLY when the user explicitly asks for totals, "
                    "a breakdown, or a summary — e.g. 'revenue breakdown for Italy in 2024'. "
                    "Use for queries like 'filter orders by Italy and 2024', "
                    "'list Electronics orders in North America'."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "filters": {
                            "type": "object",
                            "description": (
                                "Map of field name to filter value. "
                                f"Valid keys: {_VALID_FIELDS}. "
                                'Example: {"year": 2024, "country": "Italy"}.'
                            ),
                        },
                        "summarize": {
                            "type": "boolean",
                            "description": (
                                "False (default): return individual order rows. "
                                "True: return grouped aggregate totals. "
                                "Only set to true when user explicitly asks for totals/summary/breakdown."
                            ),
                            "default": False,
                        },
                    },
                    "required": ["filters"],
                },
            },
            {
                "name": "pivot",
                "description": (
                    "Cross-tabulate two dimensions and aggregate a measure. "
                    "Always returns aggregated data. "
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
            "For slice and dice, only pass summarize=true when the user explicitly "
            "asks for totals, a breakdown, or a summary. "
            "Default behaviour is to return individual order rows. "
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
            return self.slice(
                dimension=args["dimension"],
                value=args["value"],
                summarize=args.get("summarize", False),
            )
        if tool_use.name == "dice":
            return self.dice(
                filters=args["filters"],
                summarize=args.get("summarize", False),
            )
        if tool_use.name == "pivot":
            return self.pivot(
                row_dim=args["row_dim"],
                col_dim=args["col_dim"],
                measure=args["measure"],
            )

        return {"status": "error", "message": f"Unknown tool: {tool_use.name}"}

    # ── private row-level query ───────────────────────────────────────────────

    def _list_orders(
        self,
        filters: dict[str, Any],
        measure_filters: dict[str, Any] | None = None,
    ) -> dict:
        """
        Return individual order rows matching *filters* and optional *measure_filters*.

        All dimension and measure fields are included — no aggregation.
        One row per transaction, ordered by date descending.

        Measure filters apply WHERE conditions on raw fact columns,
        e.g. {"revenue": ">500"} → WHERE f.revenue > 500.
        """
        conditions = [
            f"{_FIELD_MAP[field][0]} = {self._quote(field, val)}"
            for field, val in filters.items()
        ]

        # Add measure threshold conditions (row-level, no aggregation)
        if measure_filters:
            for measure, raw_val in measure_filters.items():
                col = _MEASURE_FILTER_MAP[measure]
                op, num = self._parse_measure_filter(str(raw_val))
                conditions.append(f"{col} {op} {num}")

        where = ("WHERE " + "\n  AND ".join(conditions)) if conditions else ""

        sql = f"""
SELECT
    f.order_id,
    d.order_date,
    d.year,
    d.quarter,
    d.month_name,
    g.region,
    g.country,
    p.category,
    p.subcategory,
    c.customer_segment,
    f.quantity,
    f.unit_price,
    f.revenue,
    f.cost,
    f.profit,
    ROUND(f.profit_margin, 4) AS profit_margin
FROM fact_sales f
JOIN dim_date      d ON f.date_key      = d.date_key
JOIN dim_geography g ON f.geography_key = g.geography_key
JOIN dim_product   p ON f.product_key   = p.product_key
JOIN dim_customer  c ON f.customer_key  = c.customer_key
{where}
ORDER BY d.order_date DESC
        """.strip()

        rows = self.execute_sql(sql)
        filter_s = (
            ", ".join(f"{k}={v!r}" for k, v in filters.items()) if filters else "none"
        )
        operation = "slice" if len(filters) == 1 else "dice"
        return {
            "status": "ok",
            "operation": operation,
            "filters": filters,
            "sql": sql,
            "rows": rows,
            "message": (
                f"Orders matching [{filter_s}] — "
                f"{len(rows)} order(s) returned, sorted by date descending."
            ),
        }

    # ── private summary queries ───────────────────────────────────────────────

    def _slice_summary(self, dimension: str, value: Any) -> dict:
        """
        Grouped aggregate breakdown for a single-dimension filter.
        Groups by a secondary dimension (e.g. slicing on year → per-category totals).
        """
        col, _ = _FIELD_MAP[dimension]
        where = f"WHERE {col} = {self._quote(dimension, value)}"

        sec_field = _SECONDARY_DIM.get(dimension, "category")
        sec_col, _ = _FIELD_MAP[sec_field]
        joins = self._collect_joins([dimension, sec_field])

        sql = f"""
SELECT
    {sec_col} AS {sec_field},
    SUM(f.revenue)                  AS total_revenue,
    SUM(f.profit)                   AS total_profit,
    SUM(f.cost)                     AS total_cost,
    SUM(f.quantity)                 AS total_quantity,
    COUNT(f.order_id)               AS order_count,
    ROUND(AVG(f.profit_margin), 4)  AS avg_profit_margin
FROM fact_sales f
{joins}
{where}
GROUP BY {sec_col}
ORDER BY total_revenue DESC
        """.strip()

        rows = self.execute_sql(sql)
        return {
            "status": "ok",
            "operation": "slice",
            "dimension": dimension,
            "value": value,
            "sql": sql,
            "rows": rows,
            "message": (
                f"Slice summary on {dimension}={value!r} — "
                f"{len(rows)} {sec_field}(s) returned, sorted by revenue."
            ),
        }

    def _dice_summary(self, filters: dict[str, Any]) -> dict:
        """
        Grouped aggregate breakdown for multi-dimension filters.
        Auto-selects a grouping column not present in the filters.
        """
        used_fields = set(filters.keys())

        group_field = ""
        # First pass: prefer non-temporal, non-filter dimension
        for candidate in (
            "category",
            "region",
            "subcategory",
            "country",
            "customer_segment",
        ):
            if candidate not in used_fields:
                group_field = candidate
                break
        # Second pass: any unused dimension
        if not group_field:
            for candidate in _FIELD_MAP:
                if candidate not in used_fields:
                    group_field = candidate
                    break

        all_fields = list(filters.keys())
        if group_field:
            all_fields.append(group_field)

        joins = self._collect_joins(all_fields)
        conditions = [
            f"{_FIELD_MAP[field][0]} = {self._quote(field, val)}"
            for field, val in filters.items()
        ]
        where = "WHERE " + "\n  AND ".join(conditions)

        if group_field:
            group_col = _FIELD_MAP[group_field][0]
            sql = f"""
SELECT
    {group_col} AS {group_field},
    SUM(f.revenue)                  AS total_revenue,
    SUM(f.profit)                   AS total_profit,
    SUM(f.cost)                     AS total_cost,
    SUM(f.quantity)                 AS total_quantity,
    COUNT(f.order_id)               AS order_count,
    ROUND(AVG(f.profit_margin), 4)  AS avg_profit_margin
FROM fact_sales f
{joins}
{where}
GROUP BY {group_col}
ORDER BY total_revenue DESC
            """.strip()
        else:
            sql = f"""
SELECT
    SUM(f.revenue)                  AS total_revenue,
    SUM(f.profit)                   AS total_profit,
    SUM(f.cost)                     AS total_cost,
    SUM(f.quantity)                 AS total_quantity,
    COUNT(f.order_id)               AS order_count,
    ROUND(AVG(f.profit_margin), 4)  AS avg_profit_margin
FROM fact_sales f
{joins}
{where}
            """.strip()

        rows = self.execute_sql(sql)
        filter_s = ", ".join(f"{k}={v!r}" for k, v in filters.items())
        group_msg = f", grouped by {group_field}" if group_field else ""
        return {
            "status": "ok",
            "operation": "dice",
            "filters": filters,
            "sql": sql,
            "rows": rows,
            "message": (
                f"Dice summary on [{filter_s}]{group_msg} — "
                f"{len(rows)} row(s) returned, sorted by revenue."
            ),
        }

    # ── private helpers ───────────────────────────────────────────────────────

    def _validate_field(self, field: str) -> None:
        if field not in _FIELD_MAP:
            raise ValueError(f"Unknown field '{field}'. Valid fields: {_VALID_FIELDS}")

    def _validate_measure(self, measure: str) -> None:
        if measure not in _MEASURES:
            raise ValueError(
                f"Unknown measure '{measure}'. Valid measures: {_VALID_MEASURES}"
            )

    def _normalize_quarter(self, value: Any) -> int:
        """
        Convert any quarter representation to an integer 1-4.
        Handles: 4, "4", "Q4", "q4", "quarter 4"
        """
        s = str(value).strip().upper().lstrip("Q").strip()
        try:
            n = int(s)
            if 1 <= n <= 4:
                return n
        except ValueError:
            pass
        raise ValueError(
            f"Invalid quarter value '{value}'. Expected 1-4, Q1-Q4, or 'quarter 1-4'."
        )

    def _quote(self, field: str, value: Any) -> str:
        """Return a SQL-safe literal for *value*, quoting strings."""
        if field == "quarter":
            return str(self._normalize_quarter(value))
        if field in _NUMERIC_FIELDS:
            return str(int(value))
        # Escape any single quotes in string values
        return "'" + str(value).replace("'", "''") + "'"

    def _parse_measure_filter(self, raw: str) -> tuple[str, str]:
        """Parse '>500', '>=1000', '<200', '500' → (operator, number_str)."""
        m = re.match(r"^\s*(>=|<=|!=|>|<|=)?\s*(-?\d+(?:\.\d+)?)\s*$", raw)
        if not m:
            raise ValueError(
                f"Invalid measure filter {raw!r}. Use e.g. '>500', '>=1000', '<200'."
            )
        op  = m.group(1) or ">"
        num = m.group(2)
        return op, num

    def _collect_joins(self, fields: Any) -> str:
        """Return deduplicated JOIN clauses for the given *fields*, preserving order."""
        seen: set[str] = set()
        result: list[str] = []
        for field in fields:
            _, join = _FIELD_MAP[field]
            if join not in seen:
                seen.add(join)
                result.append(join)
        return "\n".join(result)
