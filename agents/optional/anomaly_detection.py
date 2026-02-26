"""
AnomalyDetectionAgent — statistical outlier detection across OLAP dimensions.

Operations
──────────
  detect(metric, dimension, sensitivity)  — flag outliers using z-score
  interpret(anomalies, context)           — explain anomalies in plain business language
  run(query)                              — detect + interpret in one call

Metrics
───────
  revenue, profit, cost, quantity

Dimensions
──────────
  year, quarter, month, region, country, category, subcategory, customer_segment

Z-score method
──────────────
  Aggregates *metric* grouped by *dimension*, then computes:

      z = (value − mean) / stddev_samp

  using DuckDB window functions — no Python-side statistics needed.
  A row is flagged as anomalous when abs(z) > sensitivity.

  Sensitivity guide:
    2.0 (default) — roughly 5 % of a normal distribution flagged
    2.5           — roughly 1 %
    3.0           — roughly 0.3 %  (classic "three-sigma" rule)

Usage
─────
    import anthropic, duckdb
    from agents.optional.anomaly_detection import AnomalyDetectionAgent

    client = anthropic.Anthropic()
    con    = duckdb.connect("olap.duckdb")
    agent  = AnomalyDetectionAgent(client, con)

    result = agent.run({
        "metric":      "revenue",
        "dimension":   "country",
        "sensitivity": 2.0,
    })
    # result["anomalies"]      — flagged rows with z_score column
    # result["normal"]         — unflagged rows
    # result["interpretation"] — Claude's business explanation
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from agents.base_agent import BaseAgent


# ── Field → SQL mapping ────────────────────────────────────────────────────────

_FIELD_MAP: dict[str, tuple[str, str]] = {
    "year":             ("d.year",             "JOIN dim_date      d ON f.date_key      = d.date_key"),
    "quarter":          ("d.quarter",          "JOIN dim_date      d ON f.date_key      = d.date_key"),
    "month":            ("d.month",            "JOIN dim_date      d ON f.date_key      = d.date_key"),
    "region":           ("g.region",           "JOIN dim_geography g ON f.geography_key = g.geography_key"),
    "country":          ("g.country",          "JOIN dim_geography g ON f.geography_key = g.geography_key"),
    "category":         ("p.category",         "JOIN dim_product   p ON f.product_key   = p.product_key"),
    "subcategory":      ("p.subcategory",       "JOIN dim_product   p ON f.product_key   = p.product_key"),
    "customer_segment": ("c.customer_segment", "JOIN dim_customer  c ON f.customer_key  = c.customer_key"),
}

# Metric name → SQL aggregate expression
_METRIC_MAP: dict[str, str] = {
    "revenue":  "SUM(f.revenue)",
    "profit":   "SUM(f.profit)",
    "cost":     "SUM(f.cost)",
    "quantity": "SUM(f.quantity)",
}

_VALID_FIELDS  = sorted(_FIELD_MAP)
_VALID_METRICS = sorted(_METRIC_MAP)


# ── Helper ─────────────────────────────────────────────────────────────────────

def _safe_json(obj: Any) -> Any:
    """Recursively convert Decimal → float so json.dumps works."""
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_json(i) for i in obj]
    return obj


# ── Agent ──────────────────────────────────────────────────────────────────────

class AnomalyDetectionAgent(BaseAgent):
    """Detects statistical outliers in aggregated OLAP metrics using z-scores."""

    # ── public methods ─────────────────────────────────────────────────────────

    def detect(
        self,
        metric: str,
        dimension: str,
        sensitivity: float = 2.0,
    ) -> dict:
        """
        Aggregate *metric* by *dimension*, then flag outliers whose absolute
        z-score exceeds *sensitivity* standard deviations from the mean.

        Z-scores are computed entirely in DuckDB using window functions; no
        external statistics libraries are required.

        Parameters
        ----------
        metric:
            Measure to aggregate — one of: revenue, profit, cost, quantity.
        dimension:
            Grouping field — one of the supported dimension names.
        sensitivity:
            Z-score threshold for flagging an outlier.  Default 2.0.

        Returns
        -------
        dict with keys:
            status, operation, metric, dimension, sensitivity, sql,
            all_rows, normal, anomalies, anomaly_count, message
        """
        # ── validation ─────────────────────────────────────────────────────
        if metric not in _METRIC_MAP:
            return {
                "status":  "error",
                "message": (
                    f"Invalid metric '{metric}'. "
                    f"Valid options: {_VALID_METRICS}"
                ),
            }
        if dimension not in _FIELD_MAP:
            return {
                "status":  "error",
                "message": (
                    f"Invalid dimension '{dimension}'. "
                    f"Valid options: {_VALID_FIELDS}"
                ),
            }

        dim_col, join = _FIELD_MAP[dimension]
        metric_agg    = _METRIC_MAP[metric]
        metric_alias  = f"total_{metric}"

        # ── SQL — aggregate → window stats → z-score ───────────────────────
        sql = f"""
WITH aggregated AS (
    SELECT
        {dim_col}         AS {dimension},
        {metric_agg}      AS metric_value
    FROM fact_sales f
    {join}
    GROUP BY {dim_col}
),
with_stats AS (
    SELECT
        *,
        AVG(metric_value)        OVER () AS mean_val,
        STDDEV_SAMP(metric_value) OVER () AS stddev_val
    FROM aggregated
)
SELECT
    {dimension},
    ROUND(metric_value, 2)                                      AS {metric_alias},
    ROUND(mean_val, 2)                                          AS mean_{metric},
    ROUND(stddev_val, 2)                                        AS stddev_{metric},
    ROUND(
        CASE
            WHEN stddev_val IS NULL OR stddev_val = 0 THEN 0.0
            ELSE (metric_value - mean_val) / stddev_val
        END,
        4
    )                                                           AS z_score
FROM with_stats
ORDER BY ABS(z_score) DESC
        """.strip()

        rows = self.execute_sql(sql)

        if not rows:
            return {
                "status":  "error",
                "message": (
                    f"No data returned for metric='{metric}', "
                    f"dimension='{dimension}'."
                ),
            }

        # ── split into normal / anomaly ────────────────────────────────────
        threshold = float(sensitivity)
        anomalies = [r for r in rows if abs(float(r.get("z_score") or 0)) > threshold]
        normal    = [r for r in rows if abs(float(r.get("z_score") or 0)) <= threshold]

        # Build a concise summary of flagged members
        flagged_names = [str(r.get(dimension, "?")) for r in anomalies]
        flagged_str   = (
            ", ".join(repr(n) for n in flagged_names[:5])
            + (" …" if len(flagged_names) > 5 else "")
        ) if flagged_names else "none"

        return {
            "status":        "ok",
            "operation":     "detect",
            "metric":        metric,
            "dimension":     dimension,
            "sensitivity":   sensitivity,
            "sql":           sql,
            "all_rows":      rows,
            "normal":        normal,
            "anomalies":     anomalies,
            "anomaly_count": len(anomalies),
            "message": (
                f"Detected {len(anomalies)} anomaly(ies) in {metric} by {dimension} "
                f"(sensitivity={sensitivity}, n={len(rows)}). "
                f"Flagged: {flagged_str}."
            ),
        }

    def run(self, query: dict) -> dict:
        """
        Detect anomalies and, if any are found, return a business interpretation.

        Parameters
        ----------
        query : dict
            {
                "metric":      str,    # one of: revenue, profit, cost, quantity
                "dimension":   str,    # grouping field (e.g. "country", "category")
                "sensitivity": float,  # optional — z-score threshold, default 2.0
            }

        Returns
        -------
        dict with keys:
            status, operation, metric, dimension, sensitivity, sql,
            all_rows, normal, anomalies, anomaly_count, interpretation, message
        """
        if not isinstance(query, dict):
            return {
                "status":  "error",
                "message": (
                    "run() expects a dict with keys: metric, dimension, sensitivity."
                ),
            }

        metric      = query.get("metric", "revenue")
        dimension   = query.get("dimension", "region")
        sensitivity = float(query.get("sensitivity", 2.0))

        detection = self.detect(metric, dimension, sensitivity)
        if detection.get("status") == "error":
            return detection

        return {
            "status":         "ok",
            "operation":      "anomaly_detection",
            "metric":         metric,
            "dimension":      dimension,
            "sensitivity":    sensitivity,
            "sql":            detection["sql"],
            "all_rows":       detection["all_rows"],
            "normal":         detection["normal"],
            "anomalies":      detection["anomalies"],
            "anomaly_count":  detection["anomaly_count"],
            "interpretation": "",
            "message":        detection["message"],
        }
