"""
KPICalculatorAgent — time-series KPIs and ranked performance metrics.

Operations
──────────
  yoy_growth(metric, dimension)           — year-over-year growth % per dimension
  mom_change(metric, year)                — month-over-month change % within a year
  top_n(dimension, metric, n, filters)    — top N performers, optional filters
  profit_margins(dimension)               — profit margin % grouped by dimension

Metrics accepted by yoy_growth / mom_change
────────────────────────────────────────────
  revenue, profit, cost, quantity

Metrics accepted by top_n
──────────────────────────
  revenue, profit, cost, quantity, order_count, profit_margin

Dimensions accepted by yoy_growth
───────────────────────────────────
  overall (no breakdown), region, country, category, subcategory,
  customer_segment, quarter

Dimensions accepted by profit_margins / top_n
──────────────────────────────────────────────
  year, quarter, month, region, country, category, subcategory, customer_segment

Usage
─────
    import anthropic, duckdb
    from agents.kpi_calculator import KPICalculatorAgent

    client = anthropic.Anthropic()
    con    = duckdb.connect("olap.duckdb")
    agent  = KPICalculatorAgent(client, con)

    result = agent.run("year-over-year revenue growth by region")
    result = agent.run("month-over-month profit change in 2023")
    result = agent.run("top 5 countries by revenue in 2024")
    result = agent.run("profit margins by category")
"""

from __future__ import annotations

from typing import Any

from agents.base_agent import BaseAgent


# ── Field → SQL mapping (shared with CubeOperationsAgent conventions) ─────────

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

_NUMERIC_FIELDS: frozenset[str] = frozenset({"year", "quarter", "month"})

# dim_date is always joined for year access in yoy/mom — stored separately
_DATE_JOIN = "JOIN dim_date d ON f.date_key = d.date_key"

# ── Metric definitions ────────────────────────────────────────────────────────

# Metrics valid for time-series growth calculations (raw additive measures only)
_GROWTH_METRICS: dict[str, str] = {
    "revenue":  "f.revenue",
    "profit":   "f.profit",
    "cost":     "f.cost",
    "quantity": "f.quantity",
}

# Metrics valid for top_n ranking (includes derived measures)
_RANKING_METRICS: dict[str, tuple[str, str]] = {
    # name: (aggregate_sql, output_alias)
    "revenue":       ("SUM(f.revenue)",                                          "total_revenue"),
    "profit":        ("SUM(f.profit)",                                           "total_profit"),
    "cost":          ("SUM(f.cost)",                                             "total_cost"),
    "quantity":      ("SUM(f.quantity)",                                         "total_quantity"),
    "order_count":   ("COUNT(f.order_id)",                                       "order_count"),
    "profit_margin": ("ROUND(SUM(f.profit)/NULLIF(SUM(f.revenue),0)*100, 4)",   "profit_margin_pct"),
}

# Valid dimensions for yoy_growth — excludes year/month/month_name since those
# are the time axes; "overall" is handled as a special no-grouping case.
_YOY_DIMENSIONS = ["overall", "quarter", "region", "country",
                   "category", "subcategory", "customer_segment"]

_VALID_FIELDS         = sorted(_FIELD_MAP)
_VALID_GROWTH_METRICS = sorted(_GROWTH_METRICS)
_VALID_RANKING_METRICS = sorted(_RANKING_METRICS)


# ── Agent ─────────────────────────────────────────────────────────────────────

class KPICalculatorAgent(BaseAgent):
    """Calculates time-series KPIs and ranked performance metrics."""

    # ── public KPI methods ────────────────────────────────────────────────────

    def yoy_growth(self, metric: str, dimension: str = "overall") -> dict:
        """
        Year-over-year growth percentage for *metric*, grouped by *dimension*.

        Uses a LAG window function to compare each year to the preceding year
        within the same dimension member.  The first year has NULL growth.

        Parameters
        ----------
        metric:
            One of: revenue, profit, cost, quantity.
        dimension:
            Grouping field, or "overall" for a single total-per-year series.
        """
        self._validate_growth_metric(metric)
        if dimension not in _YOY_DIMENSIONS:
            return {
                "status":  "error",
                "message": (
                    f"Invalid dimension '{dimension}' for yoy_growth. "
                    f"Valid options: {_YOY_DIMENSIONS}"
                ),
            }

        fact_col = _GROWTH_METRICS[metric]
        overall  = (dimension == "overall")

        # Time-based dimensions (year, quarter, month) all live in dim_date,
        # which _DATE_JOIN already covers — no extra join needed for those.
        _DATE_FIELDS = {"year", "quarter", "month", "month_name"}

        if overall:
            group_select  = ""
            group_by      = ""
            partition_by  = ""
            order_select  = "year,"
            order_final   = "year"
            extra_join    = ""
        else:
            dim_col, dim_join = _FIELD_MAP[dimension]
            group_select  = f"{dim_col} AS {dimension},"
            group_by      = f"{dim_col},"
            partition_by  = f"PARTITION BY {dimension}"
            order_select  = f"{dimension}, year,"
            order_final   = f"{dimension}, year"
            extra_join    = "" if dimension in _DATE_FIELDS else f"\n    {dim_join}"

        sql = f"""
WITH yearly AS (
    SELECT
        {group_select}
        d.year,
        SUM({fact_col}) AS total
    FROM fact_sales f
    {_DATE_JOIN}{extra_join}
    GROUP BY {group_by} d.year
),
with_lag AS (
    SELECT
        *,
        LAG(total) OVER ({partition_by} ORDER BY year) AS prev_year_total
    FROM yearly
)
SELECT
    {order_select}
    ROUND(total, 2)            AS total_{metric},
    ROUND(prev_year_total, 2)  AS prev_year_total_{metric},
    CASE
        WHEN prev_year_total IS NULL OR prev_year_total = 0 THEN NULL
        ELSE ROUND((total - prev_year_total) / prev_year_total * 100, 2)
    END AS yoy_growth_pct
FROM with_lag
ORDER BY {order_final}
        """.strip()

        rows  = self.execute_sql(sql)
        label = "overall" if overall else f"by {dimension}"
        return {
            "status":    "ok",
            "operation": "yoy_growth",
            "metric":    metric,
            "dimension": dimension,
            "sql":       sql,
            "rows":      rows,
            "message":   (
                f"YoY growth for {metric} {label} — "
                f"{len(rows)} row(s) returned."
            ),
        }

    def mom_change(self, metric: str, year: int) -> dict:
        """
        Month-over-month change percentage for *metric* within *year*.

        Uses LAG over month order so January always has NULL change.

        Parameters
        ----------
        metric:
            One of: revenue, profit, cost, quantity.
        year:
            The calendar year to analyse (e.g. 2023).
        """
        self._validate_growth_metric(metric)

        fact_col = _GROWTH_METRICS[metric]

        sql = f"""
WITH monthly AS (
    SELECT
        d.month,
        d.month_name,
        SUM({fact_col}) AS total
    FROM fact_sales f
    {_DATE_JOIN}
    WHERE d.year = {int(year)}
    GROUP BY d.month, d.month_name
),
with_lag AS (
    SELECT
        *,
        LAG(total) OVER (ORDER BY month) AS prev_month_total
    FROM monthly
)
SELECT
    month,
    month_name,
    ROUND(total, 2)             AS total_{metric},
    ROUND(prev_month_total, 2)  AS prev_month_{metric},
    CASE
        WHEN prev_month_total IS NULL OR prev_month_total = 0 THEN NULL
        ELSE ROUND((total - prev_month_total) / prev_month_total * 100, 2)
    END AS mom_change_pct
FROM with_lag
ORDER BY month
        """.strip()

        rows = self.execute_sql(sql)
        return {
            "status":    "ok",
            "operation": "mom_change",
            "metric":    metric,
            "year":      year,
            "sql":       sql,
            "rows":      rows,
            "message":   (
                f"MoM change for {metric} in {year} — "
                f"{len(rows)} month(s) returned."
            ),
        }

    def top_n(
        self,
        dimension: str,
        metric: str,
        n: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> dict:
        """
        Top *n* performers in *dimension* ranked by *metric*, with optional filters.

        Parameters
        ----------
        dimension:
            The field to group by (e.g. "country", "subcategory").
        metric:
            The measure to rank by (e.g. "revenue", "order_count").
        n:
            Number of top results to return (default 10).
        filters:
            Optional dict of field:value pairs to apply as WHERE conditions,
            e.g. {"year": 2024, "region": "Europe"}.
        """
        self._validate_field(dimension)
        if metric not in _RANKING_METRICS:
            return {
                "status":  "error",
                "message": (
                    f"Invalid metric '{metric}'. "
                    f"Valid options: {_VALID_RANKING_METRICS}"
                ),
            }

        filters = filters or {}
        for field in filters:
            self._validate_field(field)

        dim_col, dim_join = _FIELD_MAP[dimension]
        agg_sql, alias    = _RANKING_METRICS[metric]

        # Collect unique JOINs: dimension + all filter fields
        all_fields = [dimension] + list(filters.keys())
        joins      = self._collect_joins(all_fields)

        where = ""
        if filters:
            conditions = [
                f"{_FIELD_MAP[f][0]} = {self._quote(f, v)}"
                for f, v in filters.items()
            ]
            where = "WHERE " + "\n  AND ".join(conditions)

        sql = f"""
SELECT
    {dim_col}     AS {dimension},
    {agg_sql}     AS {alias},
    COUNT(f.order_id) AS order_count
FROM fact_sales f
{joins}
{where}
GROUP BY {dim_col}
ORDER BY {alias} DESC
LIMIT {int(n)}
        """.strip()

        rows      = self.execute_sql(sql)
        filter_s  = (
            " [" + ", ".join(f"{k}={v!r}" for k, v in filters.items()) + "]"
            if filters else ""
        )
        return {
            "status":    "ok",
            "operation": "top_n",
            "dimension": dimension,
            "metric":    metric,
            "n":         n,
            "filters":   filters,
            "sql":       sql,
            "rows":      rows,
            "message":   (
                f"Top {n} {dimension}s by {metric}{filter_s} — "
                f"{len(rows)} row(s) returned."
            ),
        }

    def profit_margins(self, dimension: str) -> dict:
        """
        Profit margin percentage grouped by *dimension*, highest margin first.

        Uses NULLIF to guard against divide-by-zero on zero-revenue rows.

        Parameters
        ----------
        dimension:
            The field to group by (e.g. "category", "region", "year").
        """
        self._validate_field(dimension)
        dim_col, join = _FIELD_MAP[dimension]

        sql = f"""
SELECT
    {dim_col}                                               AS {dimension},
    ROUND(SUM(f.revenue), 2)                                AS total_revenue,
    ROUND(SUM(f.profit),  2)                                AS total_profit,
    ROUND(SUM(f.profit) / NULLIF(SUM(f.revenue), 0) * 100, 4) AS profit_margin_pct,
    COUNT(f.order_id)                                       AS order_count
FROM fact_sales f
{join}
GROUP BY {dim_col}
ORDER BY profit_margin_pct DESC
        """.strip()

        rows = self.execute_sql(sql)
        best = rows[0] if rows else {}
        return {
            "status":    "ok",
            "operation": "profit_margins",
            "dimension": dimension,
            "sql":       sql,
            "rows":      rows,
            "message":   (
                f"Profit margins by {dimension} — {len(rows)} group(s). "
                + (
                    f"Highest: {best.get(dimension)!r} at "
                    f"{best.get('profit_margin_pct')}%."
                    if best else ""
                )
            ),
        }

    # ── run — NL interface via Claude ─────────────────────────────────────────

    def run(self, query: str) -> dict:
        """
        Parse a natural-language *query* with Claude and dispatch to the
        appropriate KPI method via tool use.
        """
        tools = [
            {
                "name": "yoy_growth",
                "description": (
                    "Calculate year-over-year growth percentage for a metric, "
                    "optionally broken down by a dimension. Use when the user asks "
                    "about yearly growth, annual comparison, or YoY trends, "
                    "e.g. 'revenue growth by region' or 'YoY profit overall'."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "metric": {
                            "type": "string",
                            "enum": _VALID_GROWTH_METRICS,
                            "description": "The measure to compute growth for.",
                        },
                        "dimension": {
                            "type": "string",
                            "enum": _YOY_DIMENSIONS,
                            "description": (
                                "Grouping field, or 'overall' for a single "
                                "total-per-year series."
                            ),
                        },
                    },
                    "required": ["metric", "dimension"],
                },
            },
            {
                "name": "mom_change",
                "description": (
                    "Calculate month-over-month change percentage for a metric "
                    "within a specific year. Use when the user asks about monthly "
                    "trends or MoM changes, e.g. 'monthly revenue change in 2023'."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "metric": {
                            "type": "string",
                            "enum": _VALID_GROWTH_METRICS,
                            "description": "The measure to track month-over-month.",
                        },
                        "year": {
                            "type": "integer",
                            "description": "The calendar year to analyse (2022–2024).",
                        },
                    },
                    "required": ["metric", "year"],
                },
            },
            {
                "name": "top_n",
                "description": (
                    "Return the top N performers in a dimension ranked by a metric, "
                    "with optional filters. Use when the user asks for top/best/highest, "
                    "e.g. 'top 5 countries by revenue' or "
                    "'best subcategories by profit in 2024'."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "dimension": {
                            "type": "string",
                            "enum": _VALID_FIELDS,
                            "description": "The field to rank (e.g. 'country', 'subcategory').",
                        },
                        "metric": {
                            "type": "string",
                            "enum": _VALID_RANKING_METRICS,
                            "description": "The measure to rank by.",
                        },
                        "n": {
                            "type": "integer",
                            "description": "How many top results to return (default 10).",
                        },
                        "filters": {
                            "type": "object",
                            "description": (
                                f"Optional dimension filters, e.g. {{\"year\": 2024}}. "
                                f"Valid keys: {_VALID_FIELDS}."
                            ),
                        },
                    },
                    "required": ["dimension", "metric", "n"],
                },
            },
            {
                "name": "profit_margins",
                "description": (
                    "Show profit margin percentage grouped by a dimension, ranked "
                    "highest first. Use when the user asks about margins, profitability, "
                    "or profit percentage, e.g. 'profit margins by category'."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "dimension": {
                            "type": "string",
                            "enum": _VALID_FIELDS,
                            "description": "The field to group margins by.",
                        },
                    },
                    "required": ["dimension"],
                },
            },
        ]

        system_prompt = (
            "You are an OLAP KPI assistant. "
            "Given a natural-language query, call exactly one of the four tools: "
            "yoy_growth, mom_change, top_n, or profit_margins. "
            "Never respond in plain text — always call a tool."
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
        if tool_use.name == "yoy_growth":
            return self.yoy_growth(
                metric=args["metric"],
                dimension=args.get("dimension", "overall"),
            )
        if tool_use.name == "mom_change":
            return self.mom_change(metric=args["metric"], year=int(args["year"]))
        if tool_use.name == "top_n":
            return self.top_n(
                dimension=args["dimension"],
                metric=args["metric"],
                n=int(args.get("n", 10)),
                filters=args.get("filters") or {},
            )
        if tool_use.name == "profit_margins":
            return self.profit_margins(dimension=args["dimension"])

        return {"status": "error", "message": f"Unknown tool: {tool_use.name}"}

    # ── private helpers ───────────────────────────────────────────────────────

    def _validate_field(self, field: str) -> None:
        if field not in _FIELD_MAP:
            raise ValueError(
                f"Unknown field '{field}'. Valid fields: {_VALID_FIELDS}"
            )

    def _validate_growth_metric(self, metric: str) -> None:
        if metric not in _GROWTH_METRICS:
            raise ValueError(
                f"Invalid metric '{metric}'. "
                f"Valid options: {_VALID_GROWTH_METRICS}"
            )

    def _quote(self, field: str, value: Any) -> str:
        if field in _NUMERIC_FIELDS:
            return str(int(value))
        return "'" + str(value).replace("'", "''") + "'"

    def _collect_joins(self, fields: list[str]) -> str:
        seen:   set[str]  = set()
        result: list[str] = []
        for field in fields:
            _, join = _FIELD_MAP[field]
            if join not in seen:
                seen.add(join)
                result.append(join)
        return "\n".join(result)
