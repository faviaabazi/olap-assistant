"""
VisualizationAgent — selects and renders the best Plotly chart for OLAP data.

Operations
──────────
  recommend(data, context)           — use Claude to pick the best chart type + explain
  to_plotly(data, chart_type, title) — build a Plotly figure and return as JSON dict
  run(query)                         — recommend + render in one call

Supported chart types
─────────────────────
  bar      — grouped bars for comparing dimension members across one or more metrics
  line     — trend lines for time-series or ordered data
  pie      — part-of-whole proportions (best for ≤ 6 slices)
  scatter  — correlation between two numeric measures
  heatmap  — matrix of values across two categorical dimensions (also pivot output)

Usage
─────
    import anthropic, duckdb
    from agents.optional.visualization import VisualizationAgent

    client = anthropic.Anthropic()
    con    = duckdb.connect("olap.duckdb")
    agent  = VisualizationAgent(client, con)

    result = agent.run({
        "data":    rows,
        "context": "Year-over-year revenue growth by region",
        "title":   "YoY Revenue Growth",
    })
    # result["figure_json"] is a plain dict — pass to json.dumps or st.plotly_chart
    # result["chart_type"]  is the selected type string
    # result["reasoning"]   is Claude's 1-2 sentence explanation
"""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Any

import plotly.graph_objects as go

from agents.base_agent import BaseAgent


# ── Constants ─────────────────────────────────────────────────────────────────

SUPPORTED_CHARTS = ["bar", "line", "pie", "scatter", "heatmap"]


# ── Module-level helpers ──────────────────────────────────────────────────────

def _to_float(v: Any) -> float | None:
    """Return *v* as float, or None if it is not numeric."""
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, Decimal):
        return float(v)
    return None


def _safe_data(obj: Any) -> Any:
    """Recursively convert Decimal → float so json.dumps works."""
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _safe_data(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_data(i) for i in obj]
    return obj


def _split_columns(data: list[dict]) -> tuple[list[str], list[str]]:
    """
    Infer label (categorical) vs numeric columns from the first row.

    Returns
    -------
    (label_cols, numeric_cols)
    """
    if not data:
        return [], []
    label_cols: list[str] = []
    numeric_cols: list[str] = []
    for k, v in data[0].items():
        if _to_float(v) is not None:
            numeric_cols.append(k)
        else:
            label_cols.append(k)
    return label_cols, numeric_cols


def _extract(data: list[dict], col: str) -> list:
    """
    Pull *col* from every row, converting Decimal/numeric values to float
    and everything else to str (so Plotly can always serialise the list).
    """
    out = []
    for row in data:
        v = row.get(col)
        f = _to_float(v)
        if f is not None:
            out.append(f)
        elif v is None:
            out.append(None)
        else:
            out.append(str(v))
    return out


def _col_label(col: str) -> str:
    return col.replace("_", " ").title()


# ── Agent ─────────────────────────────────────────────────────────────────────

class VisualizationAgent(BaseAgent):
    """Recommends and renders the best Plotly chart for a given OLAP dataset."""

    # ── public methods ────────────────────────────────────────────────────────

    def recommend(self, data: list[dict], context: str = "") -> dict:
        """
        Ask Claude to select the best chart type for *data* given *context*.

        Parameters
        ----------
        data:
            List of row dicts (OLAP query result).
        context:
            Short description of what the data represents, e.g.
            "Year-over-year revenue growth by region, 2022-2024."

        Returns
        -------
        dict with keys: status, chart_type, reasoning
        """
        if not data:
            return {"status": "error", "message": "No data provided."}

        label_cols, numeric_cols = _split_columns(data)
        schema_info = {
            "row_count": len(data),
            "columns": {
                col: ("numeric" if col in numeric_cols else "categorical")
                for col in data[0].keys()
            },
            "sample_rows": _safe_data(data[:3]),
        }

        tool = {
            "name": "select_chart",
            "description": (
                "Select the single best chart type for the dataset "
                "and explain why in 1-2 sentences."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "chart_type": {
                        "type": "string",
                        "enum": SUPPORTED_CHARTS,
                        "description": "The most appropriate chart type.",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": (
                            "1-2 sentence explanation of why this chart type "
                            "best communicates the insight in the data."
                        ),
                    },
                },
                "required": ["chart_type", "reasoning"],
            },
        }

        prompt = (
            "You are a data visualisation expert advising a business analyst.\n\n"
            f"Context: {context or 'OLAP sales data'}\n\n"
            f"Dataset schema and sample:\n{json.dumps(schema_info, indent=2)}\n\n"
            f"Available chart types: {', '.join(SUPPORTED_CHARTS)}\n\n"
            "PRIORITY RULES (apply in order, first match wins):\n"
            "  1. If any column name contains 'margin', 'percent', 'pct', 'share', "
            "or '%', or the context mentions percentages → pie\n"
            "  2. If the grouping column is a time dimension (year, quarter, month, "
            "month_name) → line\n"
            "  3. If the data has ≤ 6 categorical rows and 1 numeric column → pie\n"
            "  4. If the data has categorical rows with multiple numeric columns → bar\n"
            "  5. NEVER use bar for percentage/share/margin data — use pie instead\n\n"
            "Guidelines:\n"
            "  bar     — comparisons across categories or multiple metrics\n"
            "  line    — trends over ordered/time dimensions\n"
            "  pie     — part-of-whole proportions with ≤ 6 categories\n"
            "  scatter — correlation between two numeric measures\n"
            "  heatmap — matrix of values across two categorical dimensions\n\n"
            "Select the single best chart type for this dataset."
        )

        response = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=256,
            tools=[tool],
            tool_choice={"type": "any"},
            messages=[{"role": "user", "content": prompt}],
        )

        tool_use = next(
            (block for block in response.content if block.type == "tool_use"),
            None,
        )
        if tool_use is None:
            return {
                "status": "error",
                "message": "Claude did not return a chart recommendation.",
            }

        args = tool_use.input
        return {
            "status":     "ok",
            "chart_type": args["chart_type"],
            "reasoning":  args["reasoning"],
        }

    def to_plotly(
        self,
        data: list[dict],
        chart_type: str,
        title: str = "",
    ) -> dict:
        """
        Build a Plotly figure for *data* using *chart_type*.

        Columns are classified automatically: numeric columns become y-axes /
        value axes; the first categorical column becomes the x-axis / label axis.
        For heatmap, wide (pivot) data and long (two-label) data are both handled.

        Parameters
        ----------
        data:
            List of row dicts.
        chart_type:
            One of: bar, line, pie, scatter, heatmap.
        title:
            Optional chart title shown at the top of the figure.

        Returns
        -------
        dict with keys: status, chart_type, title, figure_json
            ``figure_json`` is a plain dict (parsed from fig.to_json()) ready for
            ``json.dumps`` or ``plotly.io.from_json``.
        """
        if not data:
            return {"status": "error", "message": "No data to plot."}
        if chart_type not in SUPPORTED_CHARTS:
            return {
                "status":  "error",
                "message": (
                    f"Unsupported chart type '{chart_type}'. "
                    f"Valid options: {SUPPORTED_CHARTS}"
                ),
            }

        label_cols, numeric_cols = _split_columns(data)

        try:
            builders = {
                "bar":     self._build_bar,
                "line":    self._build_line,
                "pie":     self._build_pie,
                "scatter": self._build_scatter,
                "heatmap": self._build_heatmap,
            }
            fig = builders[chart_type](data, label_cols, numeric_cols, title)
        except Exception as exc:
            return {
                "status":  "error",
                "message": f"Failed to build '{chart_type}' chart: {exc}",
            }

        return {
            "status":      "ok",
            "chart_type":  chart_type,
            "title":       title,
            "figure_json": json.loads(fig.to_json()),
        }

    def run(self, query: dict) -> dict:
        """
        Recommend a chart type and return the rendered Plotly figure in one call.

        Parameters
        ----------
        query : dict
            {
                "data":    list[dict],  # OLAP result rows
                "context": str,         # optional — describes what the data shows
                "title":   str,         # optional — chart title (falls back to context)
            }

        Returns
        -------
        dict with keys:
            status, operation, chart_type, reasoning, title, figure_json
        """
        if not isinstance(query, dict):
            return {
                "status":  "error",
                "message": "run() expects a dict with keys: data, context, title.",
            }

        data    = query.get("data", [])
        context = query.get("context", "")
        title   = query.get("title", "") or context

        if not data:
            return {"status": "error", "message": "No data provided."}

        # ── Pre-process: detect grouping column and override chart type ────
        label_cols, numeric_cols = _split_columns(data)

        _TIME_COLS = {"year", "month", "month_name", "quarter"}
        _DIM_COLS = {"category", "subcategory", "region", "country", "customer_segment"}

        # Grouping column = first non-numeric, non-year column
        grouping_col = ""
        for lc in label_cols:
            if lc.lower() not in {"year"}:
                grouping_col = lc
                break
        if not grouping_col and label_cols:
            grouping_col = label_cols[0]

        # Determine chart type hint based on grouping column
        chart_type_hint = ""
        if grouping_col.lower() in _TIME_COLS:
            chart_type_hint = "line"
        elif grouping_col.lower() in _DIM_COLS:
            unique_vals = {str(row.get(grouping_col, "")) for row in data}
            if len(unique_vals) <= 6 and len(numeric_cols) == 1:
                chart_type_hint = "pie"
            else:
                chart_type_hint = "bar"

        # Enrich context with actual grouping column
        if grouping_col:
            context = f"{grouping_col.replace('_', ' ').title()} breakdown — {context}"

        rec = self.recommend(data, context)
        if rec.get("status") == "error":
            return rec

        chart_type = rec["chart_type"]

        # Override chart type when data structure clearly dictates it
        if chart_type_hint and chart_type != chart_type_hint:
            # Trust the data structure over the LLM recommendation
            if chart_type_hint == "line" and grouping_col.lower() in _TIME_COLS:
                chart_type = "line"
            elif chart_type_hint == "pie" and chart_type not in ("pie", "bar"):
                chart_type = chart_type_hint
        result = self.to_plotly(data, chart_type, title=title)
        if result.get("status") == "error":
            return result

        return {
            "status":      "ok",
            "operation":   "visualization",
            "chart_type":  chart_type,
            "reasoning":   rec["reasoning"],
            "title":       result["title"],
            "figure_json": result["figure_json"],
        }

    # ── private chart builders ────────────────────────────────────────────────

    def _build_bar(
        self,
        data: list[dict],
        label_cols: list[str],
        numeric_cols: list[str],
        title: str,
    ) -> go.Figure:
        """Grouped bar chart: first label col on x, all numeric cols as bar groups."""
        x_col  = label_cols[0] if label_cols else list(data[0].keys())[0]
        x_vals = _extract(data, x_col)
        y_cols = numeric_cols or [c for c in data[0].keys() if c != x_col]

        traces = [
            go.Bar(name=_col_label(col), x=x_vals, y=_extract(data, col))
            for col in y_cols
        ]

        fig = go.Figure(data=traces)
        fig.update_layout(
            title=title,
            xaxis_title=_col_label(x_col),
            barmode="group",
            template="plotly_dark",
        )
        return fig

    def _build_line(
        self,
        data: list[dict],
        label_cols: list[str],
        numeric_cols: list[str],
        title: str,
    ) -> go.Figure:
        """Line chart: first label col on x, all numeric cols as separate lines."""
        x_col  = label_cols[0] if label_cols else list(data[0].keys())[0]
        x_vals = _extract(data, x_col)
        y_cols = numeric_cols or [c for c in data[0].keys() if c != x_col]

        traces = [
            go.Scatter(
                name=_col_label(col),
                x=x_vals,
                y=_extract(data, col),
                mode="lines+markers",
            )
            for col in y_cols
        ]

        fig = go.Figure(data=traces)
        fig.update_layout(
            title=title,
            xaxis_title=_col_label(x_col),
            template="plotly_dark",
        )
        return fig

    def _build_pie(
        self,
        data: list[dict],
        label_cols: list[str],
        numeric_cols: list[str],
        title: str,
    ) -> go.Figure:
        """Donut-style pie chart: first label col as slices, first numeric as values."""
        label_col = label_cols[0] if label_cols else list(data[0].keys())[0]
        value_col = numeric_cols[0] if numeric_cols else list(data[0].keys())[-1]

        fig = go.Figure(data=[
            go.Pie(
                labels=_extract(data, label_col),
                values=_extract(data, value_col),
                hole=0.35,
            )
        ])
        fig.update_layout(title=title, template="plotly_dark")
        return fig

    def _build_scatter(
        self,
        data: list[dict],
        label_cols: list[str],
        numeric_cols: list[str],
        title: str,
    ) -> go.Figure:
        """
        Scatter plot.  If two or more numeric columns exist, uses the first two
        as x and y.  Falls back to label col on x when only one numeric exists.
        Label column values are shown as hover text.
        """
        if len(numeric_cols) >= 2:
            x_col, y_col = numeric_cols[0], numeric_cols[1]
        elif numeric_cols:
            x_col = label_cols[0] if label_cols else list(data[0].keys())[0]
            y_col = numeric_cols[0]
        else:
            cols  = list(data[0].keys())
            x_col, y_col = cols[0], cols[-1]

        hover_col = label_cols[0] if label_cols else None

        scatter = go.Scatter(
            x=_extract(data, x_col),
            y=_extract(data, y_col),
            mode="markers",
            text=_extract(data, hover_col) if hover_col else None,
            hovertemplate=(
                f"{_col_label(x_col)}: %{{x}}<br>"
                f"{_col_label(y_col)}: %{{y}}<br>"
                + ("%{text}<extra></extra>" if hover_col else "<extra></extra>")
            ),
            marker=dict(size=10),
        )

        fig = go.Figure(data=[scatter])
        fig.update_layout(
            title=title,
            xaxis_title=_col_label(x_col),
            yaxis_title=_col_label(y_col),
            template="plotly_dark",
        )
        return fig

    def _build_heatmap(
        self,
        data: list[dict],
        label_cols: list[str],
        numeric_cols: list[str],
        title: str,
    ) -> go.Figure:
        """
        Heatmap with two layout strategies:

        Long format (≥ 2 label columns):
            y = unique values of label_cols[0]
            x = unique values of label_cols[1]
            z = first numeric column

        Wide / pivot format (1 label column + N numeric columns):
            y = label column values (row names)
            x = numeric column names (e.g. years from a pivot)
            z = 2-D matrix of numeric values
        """
        if len(label_cols) >= 2:
            y_col, x_col = label_cols[0], label_cols[1]
            z_col = numeric_cols[0] if numeric_cols else list(data[0].keys())[-1]

            y_vals = sorted({str(r.get(y_col, "")) for r in data})
            x_vals = sorted({str(r.get(x_col, "")) for r in data})

            lookup: dict[tuple[str, str], float | None] = {
                (str(r.get(y_col, "")), str(r.get(x_col, ""))): _to_float(r.get(z_col))
                for r in data
            }
            z_matrix = [
                [lookup.get((y, x)) for x in x_vals]
                for y in y_vals
            ]
            colorbar_title = _col_label(z_col)

        else:
            # Wide / pivot: label col = row names, numeric cols = x-axis
            y_col  = label_cols[0] if label_cols else list(data[0].keys())[0]
            y_vals = [str(r.get(y_col, "")) for r in data]
            x_vals = numeric_cols or [c for c in data[0].keys() if c != y_col]
            z_matrix = [
                [_to_float(row.get(x)) for x in x_vals]
                for row in data
            ]
            colorbar_title = "Value"

        # Build formatted text annotations for each cell
        text_matrix = [
            [f"{v:,.2f}" if isinstance(v, float) else ("" if v is None else str(v))
             for v in row]
            for row in z_matrix
        ]

        fig = go.Figure(data=go.Heatmap(
            z=z_matrix,
            x=x_vals,
            y=y_vals,
            text=text_matrix,
            texttemplate="%{text}",
            colorscale="Viridis",
            colorbar=dict(title=colorbar_title),
        ))
        fig.update_layout(title=title, template="plotly_dark")
        return fig
