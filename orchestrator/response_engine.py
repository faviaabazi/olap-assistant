from __future__ import annotations

import re
from decimal import Decimal
from typing import Any, Dict, List, Optional


# ── Number formatting ────────────────────────────────────────────────────────

def _to_float(v: Any) -> Optional[float]:
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, Decimal):
        return float(v)
    return None


def format_number(value: Any) -> str:
    """Format a numeric value using $1.2M / $840K / $847.50 / 14,690 style."""
    num = _to_float(value)
    if num is None:
        return str(value)

    abs_v = abs(num)

    if abs_v >= 1_000_000:
        return f"${abs_v / 1_000_000:.1f}M" if num >= 0 else f"-${abs_v / 1_000_000:.1f}M"
    if abs_v >= 1_000:
        return f"${abs_v / 1_000:.1f}K" if num >= 0 else f"-${abs_v / 1_000:.1f}K"
    if abs_v == 0:
        return "$0.00"

    # Small currency or plain count — heuristic: values < 1 are likely percentages
    if abs_v < 1:
        pct = num * 100
        return f"{pct:+.1f}%" if num != abs(num) else f"{pct:.1f}%"

    return f"${abs_v:,.2f}" if num >= 0 else f"-${abs_v:,.2f}"


def compute_percentage(part: float, total: float) -> str:
    if total == 0:
        return "0.0%"
    pct = (part / total) * 100
    return f"{pct:.1f}%"


# ── Mode detection ───────────────────────────────────────────────────────────

_COMPARISON_WORDS = frozenset({
    "vs", "versus", "compare", "comparison", "difference",
    "growth", "yoy", "mom", "year-over-year", "month-over-month",
})

_TREND_WORDS = frozenset({
    "trend", "over time", "monthly", "quarterly", "timeline",
})

_SUMMARY_WORDS = frozenset({
    "summary", "summarize", "executive", "briefing", "overview",
})

_LIST_PATTERN = re.compile(
    r"\b(top\s+\d+|bottom\s+\d+|ranking|best\s+\d+|worst\s+\d+)\b",
    re.IGNORECASE,
)


def detect_mode(query: str) -> str:
    q = query.lower().strip()

    if any(w in q for w in _SUMMARY_WORDS):
        return "summary"
    if any(w in q for w in _COMPARISON_WORDS):
        return "comparison"
    if any(w in q for w in _TREND_WORDS):
        return "trend"
    if _LIST_PATTERN.search(q):
        return "list"
    return "direct"


# ── Symbolic response (no LLM) ──────────────────────────────────────────────

def _label_columns(row: dict) -> tuple[list[str], list[str]]:
    labels, numerics = [], []
    for k, v in row.items():
        if _to_float(v) is not None:
            numerics.append(k)
        else:
            labels.append(k)
    return labels, numerics


def symbolic_response(query: str, rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "No data returned."

    label_cols, numeric_cols = _label_columns(rows[0])

    # Single-row, single-numeric → direct formatted answer
    if len(rows) == 1 and len(numeric_cols) == 1:
        col = numeric_cols[0]
        return f"{col.replace('_', ' ').title()}: {format_number(rows[0][col])}"

    # Single-row, multiple numerics → key-value pairs
    if len(rows) == 1:
        parts = [
            f"{c.replace('_', ' ').title()}: {format_number(rows[0][c])}"
            for c in numeric_cols
        ]
        return " | ".join(parts)

    # Multiple rows → structured ranked lines
    lines = []
    for i, row in enumerate(rows, 1):
        label_parts = [str(row.get(c, "")) for c in label_cols]
        label = " — ".join(label_parts) if label_parts else f"Row {i}"
        metric_parts = [format_number(row.get(c)) for c in numeric_cols]
        lines.append(f"#{i} {label}: {', '.join(metric_parts)}")

    return "\n".join(lines)


def structured_summary(rows: List[Dict[str, Any]]) -> str:
    """Build a plain-text numeric summary suitable for LLM context."""
    if not rows:
        return "No data."

    label_cols, numeric_cols = _label_columns(rows[0])
    lines = []
    for row in rows:
        label = " | ".join(str(row.get(c, "")) for c in label_cols)
        metrics = ", ".join(
            f"{c}={format_number(row.get(c))}" for c in numeric_cols
        )
        lines.append(f"{label}: {metrics}" if label else metrics)

    return "\n".join(lines)


# ── Generative response (LLM) ───────────────────────────────────────────────

def generative_summary(
    client: Any,
    query: str,
    rows: List[Dict[str, Any]],
) -> str:
    summary = structured_summary(rows)

    prompt = (
        f"User question: {query}\n\n"
        f"Data (pre-formatted):\n{summary}\n\n"
        "Write 2-3 sentences analyzing the data above.\n"
        "Rules:\n"
        "- Use ONLY the numbers provided. Do not invent data.\n"
        "- Use the exact formatted numbers from the data.\n"
        "- Never start with 'I' or 'The data shows'.\n"
        "- Lead with the most important finding.\n"
        "- Facts and comparisons only — no recommendations.\n"
    )

    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip() if resp.content else ""


# ── Main entry point ─────────────────────────────────────────────────────────

def generate_response(
    client: Any,
    query: str,
    rows: List[Dict[str, Any]],
) -> str:
    mode = detect_mode(query)

    if mode in ("direct", "list"):
        return symbolic_response(query, rows)

    return generative_summary(client, query, rows)
