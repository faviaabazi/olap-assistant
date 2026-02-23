"""
Streamlit chat interface for the OLAP Assistant.

Run with:
    streamlit run frontend/app.py
"""

from __future__ import annotations

import base64
import html
import io
import json
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.io as pio
import requests
import streamlit as st
import streamlit.components.v1 as components
from fpdf import FPDF

# ── Constants ─────────────────────────────────────────────────────────────────

API_BASE = "http://localhost:8000"

_ICON_PATH = Path(__file__).parent / "logo.png"
_ICON_B64 = base64.b64encode(_ICON_PATH.read_bytes()).decode() if _ICON_PATH.exists() else ""

PROMPT_TEMPLATES = [
    ("\U0001f4ca", "Chart",      "Show me a chart of revenue by ______"),
    ("\U0001f4c8", "Growth",     "Show year-over-year growth of ______ by region"),
    ("\U0001f50d", "Drill Down", "Drill down into ______ by country"),
    ("\U0001f3c6", "Top 5",      "Top 5 ______ by revenue"),
    ("\u26a0\ufe0f", "Anomalies",  "Detect profit anomalies by ______"),
    ("\U0001f4cb", "Summary",    "Give me an executive summary of ______"),
]

# Modes that should show the PDF download button
_PDF_MODES = {"direct", "chart", "comparison", "list", "summary", "report"}

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="OLAP Assistant",
    page_icon=str(_ICON_PATH) if _ICON_PATH.exists() else "\U0001f4ca",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state ─────────────────────────────────────────────────────────────

_DEFAULTS: dict = {
    "messages":      [],
    "session_id":    str(uuid.uuid4()),
    "show_welcome":  True,
    "query_history": [],
    "input_value":   "",
    "last_query":    "",
    "pinned":        [],
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Global CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Base ── */
html, body { background: #111111 !important; }
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main,
.block-container {
    background-color: #111111 !important;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
}

/* ── Kill ALL white backgrounds ── */
[data-testid="stBottom"],
section[data-testid="stBottom"],
div[data-testid="stBottom"] > div,
[data-testid="stChatInput"],
div[data-testid="stChatInput"],
.stChatInputContainer,
[data-testid="stChatInputContainer"],
[class*="chatInputContainer"],
[class*="stBottom"] {
    background-color: #111111 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div:first-child {
    background-color: #1a1a1a !important;
    border-right: 1px solid #333333 !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] > div {
    background: #222222 !important;
    border: 1px solid #333333 !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"] textarea {
    background: #222222 !important;
    color: #fafafa !important;
    font-size: 14px !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"] textarea:focus {
    box-shadow: 0 0 0 2px rgba(245, 158, 11, 0.25) !important;
    outline: none !important;
    border-color: rgba(245, 158, 11, 0.5) !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #737373 !important;
}

/* ── Kill all default Streamlit button white/gray backgrounds ── */
[data-testid="stBaseButton-secondary"] > button,
.stButton > button {
    background: #111111 !important;
    border: 1px solid #333333 !important;
    color: #fafafa !important;
    border-radius: 8px !important;
    transition: background 0.2s, color 0.2s, border-color 0.2s !important;
}
[data-testid="stBaseButton-secondary"] > button:hover,
.stButton > button:hover {
    background: #111111 !important;
    border-color: #f59e0b !important;
    color: #f59e0b !important;
}

/* ── Scrollbars ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #333333; border-radius: 4px; }

/* ── Expanders ── */
[data-testid="stExpander"] {
    background: #1a1a1a !important;
    border: 1px solid #333333 !important;
    border-radius: 10px !important;
    margin: 8px 0;
}
[data-testid="stExpander"] summary {
    color: #f59e0b !important;
    font-size: 13px !important;
    font-weight: 600 !important;
}
[data-testid="stExpander"] summary:hover { color: #fbbf24 !important; }
[data-testid="stExpander"] details { background: #1a1a1a !important; }

/* ── Welcome animation ── */
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
.welcome-wrap {
    animation: fadeSlideIn 0.8s ease forwards;
    display: flex;
    justify-content: center;
    padding-top: 60px;
}
.welcome-card { text-align: center; max-width: 560px; }
.welcome-icon  { font-size: 64px; line-height: 1; margin-bottom: 16px; }
.welcome-title { font-size: 28px; font-weight: 700; color: #fafafa; margin-bottom: 10px; }
.welcome-sub   { font-size: 16px; color: #737373; margin-bottom: 24px; line-height: 1.7; }
.feature-pills {
    display: flex;
    gap: 10px;
    justify-content: center;
    flex-wrap: wrap;
    margin-bottom: 28px;
}
.feature-pill {
    background: #222222;
    border: 1px solid #333333;
    padding: 4px 12px;
    border-radius: 20px;
    color: #fafafa;
    font-size: 13px;
}
.welcome-hint { font-size: 13px; color: #737373; font-style: italic; }

/* ── User bubble ── */
.user-row {
    display: flex;
    justify-content: flex-end;
    margin: 14px 0 2px 15%;
}
.user-bubble {
    background: linear-gradient(135deg, #f59e0b, #fbbf24);
    color: #111111;
    padding: 12px 16px;
    border-radius: 18px 18px 4px 18px;
    font-size: 14px;
    line-height: 1.6;
    max-width: 70%;
    word-break: break-word;
    font-weight: 500;
}
.ts-right { font-size: 11px; color: #737373; text-align: right; margin: 2px 4px 12px; }
.ts-left  { font-size: 11px; color: #737373; text-align: left;  margin: 2px 4px 8px; }

/* ── Assistant bubble ── */
.asst-row { display: flex; margin: 14px 15% 2px 0; }
.asst-bubble {
    background: #222222;
    border-left: 3px solid #f59e0b;
    border-radius: 4px 18px 18px 18px;
    padding: 16px;
    font-size: 14px;
    line-height: 1.6;
    color: #fafafa;
    word-break: break-word;
    max-width: 85%;
    width: 100%;
}

/* ── Error bubble ── */
.error-row { display: flex; margin: 14px 15% 12px 0; }
.error-bubble {
    background: rgba(239, 68, 68, 0.07);
    border: 1px solid rgba(239, 68, 68, 0.28);
    border-radius: 4px 18px 18px 18px;
    padding: 14px 16px;
    color: #fca5a5;
    font-size: 14px;
    line-height: 1.6;
    max-width: 85%;
}

/* ── Report pre ── */
.report-pre {
    font-family: 'JetBrains Mono', 'Cascadia Code', 'Courier New', monospace;
    font-size: 13px;
    line-height: 1.65;
    color: #e2e2e2;
    white-space: pre-wrap;
    background: #1a1a1a;
    padding: 16px 18px;
    border-radius: 8px;
    border: 1px solid #333333;
    overflow-x: auto;
}

/* ── Anomaly section ── */
.anomaly-section { margin: 10px 0; }
.anomaly-header {
    font-size: 12px; font-weight: 700; color: #ef4444;
    text-transform: uppercase; letter-spacing: .09em; margin-bottom: 8px;
}
.anomaly-interp {
    font-size: 13px; color: #fafafa;
    background: rgba(239,68,68,.06);
    border: 1px solid rgba(239,68,68,.22);
    border-radius: 8px; padding: 10px 14px; margin-bottom: 10px; line-height: 1.6;
}
.anomaly-table {
    width: 100%; border-collapse: collapse;
    font-family: 'JetBrains Mono','Courier New',monospace; font-size: 11.5px;
}
.anomaly-table th {
    background: rgba(239,68,68,.12); color: #ef4444;
    padding: 6px 10px; text-align: left; border-bottom: 1px solid rgba(239,68,68,.25);
}
.anomaly-table td {
    background: rgba(239,68,68,.05); color: #fca5a5;
    padding: 5px 10px; border-bottom: 1px solid rgba(239,68,68,.1);
}
.anomaly-table tr:last-child td { border-bottom: none; }

/* ── Executive summary ── */
.exec-card {
    background: #1a1a1a; border: 1px solid #333333;
    border-radius: 14px; overflow: hidden; margin: 10px 0;
}
.exec-card-header { padding: 12px 20px 10px; border-bottom: 1px solid #333333; }
.exec-card-label {
    font-size: 10.5px; font-weight: 700; color: #737373;
    text-transform: uppercase; letter-spacing: .13em;
}
.exec-headline {
    padding: 16px 20px 14px; font-size: 15px; font-weight: 600;
    color: #f59e0b; line-height: 1.55; border-bottom: 1px solid #333333;
}
.exec-section-label {
    font-size: 10px; font-weight: 700; color: #737373;
    text-transform: uppercase; letter-spacing: .13em; padding: 12px 20px 6px;
}
.exec-insights { list-style: none; margin: 0; padding: 0 20px 14px; }
.exec-insights li {
    font-size: 13px; color: #fafafa; line-height: 1.65;
    padding: 5px 0 5px 20px; position: relative; border-bottom: 1px solid #333333;
}
.exec-insights li:last-child { border-bottom: none; }
.exec-insights li::before {
    content: "\u25b8"; position: absolute; left: 0; color: #f59e0b; font-size: 11px; top: 7px;
}
.exec-action-wrap { padding: 0 20px 18px; }
.exec-action {
    background: rgba(245,158,11,.07);
    border: 1px solid rgba(245,158,11,.2); border-radius: 8px; padding: 10px 14px;
}
.exec-action-label {
    font-size: 10px; font-weight: 700; color: #f59e0b;
    text-transform: uppercase; letter-spacing: .1em; margin-bottom: 5px;
}
.exec-action-text { font-size: 13px; color: #fbbf24; line-height: 1.6; }

/* ── Risk table ── */
.risk-section { margin: 10px 0; }
.risk-header {
    font-size: 12px; font-weight: 700; color: #f59e0b;
    text-transform: uppercase; letter-spacing: .09em; margin-bottom: 7px;
}
.risk-table {
    width: 100%; border-collapse: collapse;
    font-family: 'JetBrains Mono','Courier New',monospace; font-size: 11.5px;
}
.risk-table th {
    background: rgba(245,158,11,.09); color: #f59e0b;
    padding: 6px 10px; text-align: left; border-bottom: 1px solid rgba(245,158,11,.2);
}
.risk-table td {
    background: rgba(245,158,11,.04); color: #fcd34d;
    padding: 5px 10px; border-bottom: 1px solid rgba(245,158,11,.08);
}
.risk-table tr:last-child td { border-bottom: none; }
.risk-table td.risk-reason { color: #b45309; font-style: italic; }

/* ── Typing indicator ── */
.typing-row { display: flex; margin: 14px 15% 12px 0; }
.typing-bubble {
    background: #222222;
    border-left: 3px solid #f59e0b;
    border-radius: 4px 18px 18px 18px;
    padding: 16px 20px;
    display: inline-flex;
    align-items: center;
    gap: 12px;
}
.typing-label { font-size: 13px; color: #737373; font-style: italic; }
.typing-dots { display: flex; gap: 5px; align-items: center; }
.typing-dots span {
    width: 6px; height: 6px;
    background: #f59e0b;
    border-radius: 50%;
    animation: dotPulse 1.4s ease-in-out infinite;
    display: inline-block;
}
.typing-dots span:nth-child(2) { animation-delay: 0.2s; }
.typing-dots span:nth-child(3) { animation-delay: 0.4s; }
@keyframes dotPulse {
    0%, 80%, 100% { opacity: 0.2; transform: scale(0.8); }
    40%           { opacity: 1;   transform: scale(1.1); }
}

/* ── New conversation button ── */
.new-convo-btn > div > button {
    background: #111111 !important;
    border: 1px solid #333333 !important;
    border-radius: 8px !important;
    color: #fafafa !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    padding: 8px 16px !important;
    width: 100% !important;
    transition: background 0.2s, color 0.2s, border-color 0.2s !important;
}
.new-convo-btn > div > button:hover {
    background: #111111 !important;
    border-color: #f59e0b !important;
    color: #f59e0b !important;
}

/* ── Quick action card buttons ── */
.card-btn > div > button {
    width: 100% !important;
    background: #111111 !important;
    border: 1px solid #333333 !important;
    border-radius: 10px !important;
    padding: 10px 8px !important;
    color: #fafafa !important;
    font-size: 12px !important;
    text-align: left !important;
    white-space: normal !important;
    height: auto !important;
    min-height: 52px !important;
    line-height: 1.4 !important;
    transition: background 0.2s, color 0.2s, border-color 0.2s !important;
}
.card-btn > div > button:hover {
    background: #111111 !important;
    border-color: #f59e0b !important;
    color: #f59e0b !important;
}

/* ── History buttons ── */
.hist-btn > div > button {
    width: 100% !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 1px solid #2a2a2a !important;
    border-radius: 0 !important;
    color: #737373 !important;
    font-size: 12px !important;
    text-align: left !important;
    padding: 6px 4px !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    max-width: 100% !important;
    transition: color 0.2s !important;
}
.hist-btn > div > button:hover {
    color: #f59e0b !important;
    background: transparent !important;
}

/* ── Response action ghost buttons (Copy, Pin, Re-run) ── */
.action-ghost > div > button {
    background: #111111 !important;
    border: 1px solid #333333 !important;
    color: #fafafa !important;
    font-size: 12px !important;
    padding: 4px 8px !important;
    border-radius: 6px !important;
    transition: color 0.2s, background 0.2s, border-color 0.2s !important;
    min-width: auto !important;
    white-space: nowrap !important;
}
.action-ghost > div > button:hover {
    color: #f59e0b !important;
    background: #111111 !important;
    border-color: #f59e0b !important;
}

/* ── PDF download button ── */
.pdf-btn [data-testid="stDownloadButton"] > button {
    background: #f59e0b !important;
    border: none !important;
    border-radius: 8px !important;
    color: #111111 !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    padding: 8px 16px !important;
    white-space: nowrap !important;
    transition: background 0.2s !important;
    min-width: auto !important;
}
.pdf-btn [data-testid="stDownloadButton"] > button:hover {
    background: #fbbf24 !important;
    color: #111111 !important;
}

/* ── Sidebar helpers ── */
.sidebar-section-label {
    font-size: 10px; font-weight: 700; color: #737373;
    text-transform: uppercase; letter-spacing: 0.12em;
    margin: 18px 0 8px; display: block;
}
.sidebar-divider {
    border: none; border-top: 1px solid #f59e0b;
    opacity: 0.35; margin: 16px 0;
}

/* ── KB hint ── */
.kb-hint { font-size: 11px; color: #737373; text-align: center; padding: 4px 0 8px; }

/* ── Finding sentence ── */
.you-asked {
    font-size: 12px; color: #737373; font-style: italic;
    margin-bottom: 6px; line-height: 1.5;
}
.finding-box {
    background: #222222;
    border-left: 3px solid #f59e0b;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 12px;
    font-size: 16px;
    line-height: 1.6;
    color: #fafafa;
}

/* ── Styled result table (comparison / list) ── */
.result-table {
    width: 100%; border-collapse: collapse; font-size: 13px; margin: 10px 0;
}
.result-table th {
    background: #2a2a2a; color: #f59e0b; padding: 8px 12px; text-align: left;
    border-bottom: 2px solid rgba(245,158,11,0.3);
    font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em;
    white-space: nowrap;
}
.result-table td {
    padding: 8px 12px; border-bottom: 1px solid #2a2a2a; color: #fafafa;
    white-space: nowrap;
}
.result-table tr:last-child td { border-bottom: none; }
.result-table tr:hover td { background: #2a2a2a; }
.result-table td.null-cell { color: #555555; }

/* ── Muted takeaway text ── */
.muted-takeaway {
    font-size: 13px; color: #a0a0a0; font-style: italic;
    margin-top: 14px; padding: 8px 14px;
    border-left: 3px solid rgba(245,158,11,0.25); line-height: 1.65;
}

/* ── Chart analysis ── */
.chart-analysis {
    font-size: 14px; color: #a0a0a0; font-style: italic;
    line-height: 1.65; margin-top: 10px; padding: 8px 0;
}

/* ── Summary text ── */
.summary-text {
    font-size: 16px; color: #fafafa; line-height: 1.6;
    padding: 8px 0;
}

/* ── Direct table ── */
.direct-table {
    width: 100%; border-collapse: collapse; font-size: 12px; margin: 8px 0 12px;
}
.direct-table th {
    background: #2a2a2a; color: #737373; padding: 6px 10px; text-align: left;
    border-bottom: 1px solid #333333;
    font-size: 10px; text-transform: uppercase; letter-spacing: 0.06em;
    white-space: nowrap;
}
.direct-table td {
    padding: 6px 10px; border-bottom: 1px solid #2a2a2a; color: #e2e2e2;
    white-space: nowrap; font-size: 12px;
}
.direct-table tr:last-child td { border-bottom: none; }

/* ── Follow-up suggestions ── */
.followup-label {
    font-size: 10px; font-weight: 700; color: #737373;
    text-transform: uppercase; letter-spacing: 0.1em;
    margin: 16px 0 8px; display: block;
}
.followup-chip > div > button {
    background: #111111 !important;
    border: 1px solid #333333 !important;
    border-radius: 20px !important;
    color: #fafafa !important;
    font-size: 12px !important;
    padding: 4px 14px !important;
    transition: background 0.2s, color 0.2s, border-color 0.2s !important;
    white-space: nowrap !important;
}
.followup-chip > div > button:hover {
    background: #111111 !important;
    border-color: #f59e0b !important;
    color: #f59e0b !important;
}

/* ── Pin buttons ── */
.pin-btn > div > button {
    width: 100% !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 1px solid #2a2a2a !important;
    border-radius: 0 !important;
    color: #f59e0b !important;
    font-size: 12px !important;
    text-align: left !important;
    padding: 6px 4px !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    max-width: 100% !important;
    transition: color 0.2s !important;
}
.pin-btn > div > button:hover {
    color: #fbbf24 !important;
    background: transparent !important;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now().strftime("%H:%M")


def _update_history(query: str) -> None:
    hist = st.session_state.query_history
    if query in hist:
        hist.remove(query)
    hist.insert(0, query)
    st.session_state.query_history = hist[:5]


def _fmt_cell(val) -> str:
    """Format a table cell value: numbers get thousands separators, None -> em-dash."""
    if val is None:
        return None  # caller handles null-cell class
    try:
        f = float(val)
        if f == int(f):
            return f"{int(f):,}"
        return f"{f:,.2f}"
    except (TypeError, ValueError):
        return html.escape(str(val))


def _is_numeric(val) -> bool:
    """Check if a value is numeric."""
    if val is None:
        return False
    try:
        float(val)
        return True
    except (TypeError, ValueError):
        return False


def _compute_col_extremes(rows: list[dict], cols: list[str]) -> dict:
    """
    For each numeric column, find the index of the row with the max and min value.
    Returns {col: {"max_idx": int, "min_idx": int}} for columns with 2+ distinct values.
    """
    extremes: dict = {}
    for c in cols:
        vals = []
        for i, row in enumerate(rows):
            v = row.get(c)
            if _is_numeric(v):
                vals.append((i, float(v)))
        if len(vals) >= 2:
            max_entry = max(vals, key=lambda x: x[1])
            min_entry = min(vals, key=lambda x: x[1])
            if max_entry[1] != min_entry[1]:
                extremes[c] = {"max_idx": max_entry[0], "min_idx": min_entry[0]}
    return extremes


# ── PDF generation ────────────────────────────────────────────────────────────

def _generate_pdf(query: str, finding: str, result: dict) -> bytes:
    """Generate a PDF report with query, finding, table data, and footer."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=20)

    # Title
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "OLAP Assistant Report", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(4)

    # Timestamp
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(130, 130, 130)
    pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # Query
    pdf.set_font("Helvetica", "I", 11)
    pdf.set_text_color(80, 80, 80)
    safe_query = query.encode("latin-1", "replace").decode("latin-1")
    pdf.multi_cell(0, 6, f'You asked: "{safe_query}"')
    pdf.ln(4)

    # Finding
    if finding:
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 8, "Key Finding", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 11)
        safe_finding = finding.encode("latin-1", "replace").decode("latin-1")
        pdf.multi_cell(0, 6, safe_finding)
        pdf.ln(4)

    # Table data
    rows = (
        result.get("result_rows")
        or result.get("supporting_data")
        or []
    )
    if rows:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 8, "Data", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

        cols = list(rows[0].keys())
        n_cols = len(cols)
        col_w = min(40, int((pdf.w - 20) / max(n_cols, 1)))

        # Header
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_fill_color(240, 240, 240)
        for c in cols:
            label = c.replace("_", " ").title()[:18]
            safe_label = label.encode("latin-1", "replace").decode("latin-1")
            pdf.cell(col_w, 7, safe_label, border=1, fill=True)
        pdf.ln()

        # Rows (max 30 for PDF)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_fill_color(255, 255, 255)
        for row in rows[:30]:
            for c in cols:
                val = row.get(c, "")
                display = str(val) if val is not None else "-"
                safe_display = display[:18].encode("latin-1", "replace").decode("latin-1")
                pdf.cell(col_w, 6, safe_display, border=1)
            pdf.ln()
        pdf.ln(4)

    # Chart analysis or summary
    analysis = result.get("chart_analysis") or result.get("summary_text") or ""
    if analysis:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 8, "Analysis", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        safe_analysis = analysis.encode("latin-1", "replace").decode("latin-1")
        pdf.multi_cell(0, 5, safe_analysis)
        pdf.ln(4)

    # Interpretation (anomaly)
    interp = result.get("interpretation", "")
    if interp:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 8, "Interpretation", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        safe_interp = interp.encode("latin-1", "replace").decode("latin-1")
        pdf.multi_cell(0, 5, safe_interp)
        pdf.ln(4)

    # Footer
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(160, 160, 160)
    pdf.cell(0, 6, "OLAP Assistant", new_x="LMARGIN", new_y="NEXT", align="C")

    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(f"""
        <div style='padding:24px 0 12px; text-align:center;'>
            <img src='data:image/png;base64,{_ICON_B64}' style='width:48px; height:48px;' />
            <div style='font-size:20px; font-weight:700; color:#fafafa; margin:10px 0 4px;'>
                OLAP Assistant
            </div>
            <div style='font-size:11px; color:#737373; letter-spacing:0.08em;'>
                Retail Sales \u00b7 2022\u20132024
            </div>
        </div>
        <hr class='sidebar-divider' />
    """, unsafe_allow_html=True)

    # New conversation
    st.markdown('<div class="new-convo-btn">', unsafe_allow_html=True)
    if st.button("\u2295  New Conversation", use_container_width=True, key="new_convo"):
        try:
            requests.post(
                f"{API_BASE}/reset",
                json={"session_id": st.session_state.session_id},
                timeout=5,
            )
        except Exception:
            pass
        st.session_state.session_id    = str(uuid.uuid4())
        st.session_state.messages      = []
        st.session_state.show_welcome  = True
        st.session_state.query_history = []
        st.session_state.last_query    = ""
        st.session_state.input_value   = ""
        st.session_state.pinned        = []
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # Query history
    st.markdown('<span class="sidebar-section-label">RECENT</span>', unsafe_allow_html=True)
    if st.session_state.query_history:
        for qi, q in enumerate(st.session_state.query_history):
            label = f"\U0001f550 {q[:35]}{'...' if len(q) > 35 else ''}"
            st.markdown('<div class="hist-btn">', unsafe_allow_html=True)
            if st.button(label, key=f"hist_{qi}", use_container_width=True):
                st.session_state.input_value = q
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            "<div style='font-size:12px; color:#737373; padding:6px 0 10px;'>"
            "No recent queries</div>",
            unsafe_allow_html=True,
        )

    # Pinned findings
    st.markdown('<span class="sidebar-section-label">PINNED</span>', unsafe_allow_html=True)
    if st.session_state.pinned:
        for pi, pin in enumerate(st.session_state.pinned):
            pin_label = f"\U0001f4cc {pin[:40]}{'...' if len(pin) > 40 else ''}"
            st.markdown('<div class="pin-btn">', unsafe_allow_html=True)
            if st.button(pin_label, key=f"unpin_{pi}", use_container_width=True):
                st.session_state.pinned.pop(pi)
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            "<div style='font-size:12px; color:#737373; padding:6px 0 10px;'>"
            "No pinned findings</div>",
            unsafe_allow_html=True,
        )

    # Quick action cards
    st.markdown('<span class="sidebar-section-label">QUICK ACTIONS</span>', unsafe_allow_html=True)
    for row_start in range(0, len(PROMPT_TEMPLATES), 2):
        pair = PROMPT_TEMPLATES[row_start:row_start + 2]
        cols = st.columns(2)
        for col, (icon, label, template) in zip(cols, pair):
            with col:
                st.markdown('<div class="card-btn">', unsafe_allow_html=True)
                if st.button(f"{icon} {label}", key=f"tpl_{label}", use_container_width=True):
                    st.session_state.input_value = template
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)


# ── API helpers ───────────────────────────────────────────────────────────────

def call_query_api(query: str, session_id: str) -> dict:
    try:
        resp = requests.post(
            f"{API_BASE}/query",
            json={"query": query, "session_id": session_id},
            timeout=90,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {
            "status": "error",
            "message": (
                "Cannot connect to the API server at localhost:8000. "
                "Start it with:  uvicorn api.main:app --reload"
            ),
        }
    except requests.exceptions.Timeout:
        return {"status": "error", "message": "Request timed out (90 s). The query may be too complex."}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


# ── Render: common elements ──────────────────────────────────────────────────

def render_you_asked(query: str) -> None:
    """Show 'You asked: "..."' quote above the finding."""
    if query:
        safe = html.escape(query)
        st.markdown(
            f'<div class="you-asked">You asked: &ldquo;{safe}&rdquo;</div>',
            unsafe_allow_html=True,
        )


def render_finding(finding: str) -> None:
    """Show the finding sentence in styled box."""
    if finding:
        safe = html.escape(finding)
        st.markdown(
            f'<div class="finding-box">{safe}</div>',
            unsafe_allow_html=True,
        )


def render_follow_ups(questions: list[str], msg_idx: int) -> None:
    """Show follow-up suggestion chips."""
    if not questions:
        return
    st.markdown(
        '<span class="followup-label">You might also ask:</span>',
        unsafe_allow_html=True,
    )
    cols = st.columns(min(len(questions), 3))
    for i, q in enumerate(questions[:3]):
        with cols[i]:
            st.markdown('<div class="followup-chip">', unsafe_allow_html=True)
            if st.button(q, key=f"fu_{msg_idx}_{i}", use_container_width=True):
                st.session_state.input_value = q
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)


def render_pdf_download(query: str, result: dict, msg_idx: int) -> None:
    """Show PDF download button for modes with data."""
    response_mode = result.get("response_mode", "default")
    if response_mode not in _PDF_MODES:
        return
    finding = result.get("finding", "")
    pdf_bytes = _generate_pdf(query, finding, result)
    st.markdown('<div class="pdf-btn">', unsafe_allow_html=True)
    st.download_button(
        "\u2b07 Download as PDF",
        data=pdf_bytes,
        file_name="olap_report.pdf",
        mime="application/pdf",
        key=f"pdf_{msg_idx}",
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ── Render: result table with trend arrows ───────────────────────────────────

def render_result_table(rows: list[dict], table_class: str = "result-table") -> None:
    """Render an HTML table with trend arrows on numeric columns."""
    if not rows:
        return

    cols = list(rows[0].keys())
    extremes = _compute_col_extremes(rows, cols)

    hdr = "".join(
        f'<th>{html.escape(c.replace("_", " ").title())}</th>' for c in cols
    )
    body = ""
    for i, row in enumerate(rows):
        cells = ""
        for c in cols:
            val = row.get(c)
            if val is None:
                cells += '<td class="null-cell">\u2014</td>'
                continue

            # Format percentage columns
            if c.endswith("_pct") and _is_numeric(val):
                try:
                    pct = float(val)
                    color = "#10b981" if pct >= 0 else "#ef4444"
                    sign = "+" if pct > 0 else ""
                    cells += (
                        f'<td style="color:{color}; font-weight:600;">'
                        f'{sign}{pct:.2f}%</td>'
                    )
                    continue
                except (TypeError, ValueError):
                    pass

            display = _fmt_cell(val)
            if display is None:
                cells += '<td class="null-cell">\u2014</td>'
                continue

            # Add trend arrows
            arrow = ""
            if c in extremes:
                if i == extremes[c]["max_idx"]:
                    arrow = ' <span style="color:#10b981;">\u2191</span>'
                elif i == extremes[c]["min_idx"]:
                    arrow = ' <span style="color:#ef4444;">\u2193</span>'

            cells += f"<td>{display}{arrow}</td>"
        body += f"<tr>{cells}</tr>"

    st.markdown(
        f'<table class="{table_class}"><thead><tr>{hdr}</tr></thead>'
        f'<tbody>{body}</tbody></table>',
        unsafe_allow_html=True,
    )


def render_sections(sections: list[dict]) -> None:
    """Render multiple data sections, each with its own title and table."""
    for sec in sections:
        title = sec.get("title", "")
        rows = sec.get("rows", [])
        explanation = sec.get("explanation", "")

        if title:
            st.markdown(
                f'<div style="font-size:13px; font-weight:700; color:#f59e0b; '
                f'text-transform:uppercase; letter-spacing:0.08em; margin:18px 0 8px;">'
                f'{html.escape(title)}</div>',
                unsafe_allow_html=True,
            )

        if rows:
            render_result_table(rows)

        if explanation:
            st.markdown(
                f'<div style="font-size:13px; color:#a0a0a0; font-style:italic; '
                f'line-height:1.6; margin:4px 0 14px; padding:0 4px;">'
                f'{html.escape(explanation)}</div>',
                unsafe_allow_html=True,
            )


# ── Render: mode-specific content ────────────────────────────────────────────

def render_mode_direct(result: dict) -> None:
    """Direct: small supporting data table with trend arrows, no expander."""
    rows = result.get("supporting_data", [])
    if not rows:
        return

    cols = list(rows[0].keys())
    extremes = _compute_col_extremes(rows, cols)

    hdr = "".join(
        f'<th>{html.escape(c.replace("_", " ").title())}</th>' for c in cols
    )
    body = ""
    for i, row in enumerate(rows):
        cells = ""
        for c in cols:
            val = row.get(c)
            if val is None:
                cells += '<td style="color:#555;">\u2014</td>'
            else:
                display = _fmt_cell(val)
                arrow = ""
                if c in extremes:
                    if i == extremes[c]["max_idx"]:
                        arrow = ' <span style="color:#10b981;">\u2191</span>'
                    elif i == extremes[c]["min_idx"]:
                        arrow = ' <span style="color:#ef4444;">\u2193</span>'
                cells += f"<td>{display}{arrow}</td>"
        body += f"<tr>{cells}</tr>"
    st.markdown(
        f'<table class="direct-table"><thead><tr>{hdr}</tr></thead>'
        f'<tbody>{body}</tbody></table>',
        unsafe_allow_html=True,
    )


def render_mode_chart(result: dict) -> None:
    """Chart: Plotly chart + chart_analysis paragraph."""
    figure_json = result.get("figure_json")
    if figure_json:
        try:
            fig = pio.from_json(json.dumps(figure_json))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        except Exception:
            pass

    analysis = result.get("chart_analysis", "")
    if analysis:
        st.markdown(
            f'<div class="chart-analysis">{html.escape(analysis)}</div>',
            unsafe_allow_html=True,
        )


def render_mode_comparison(result: dict) -> None:
    """Comparison: result table(s) with trend arrows."""
    sections = result.get("sections")
    if sections:
        render_sections(sections)
    else:
        rows = result.get("result_rows", [])
        render_result_table(rows)


def render_mode_list(result: dict) -> None:
    """List: result table(s) with trend arrows."""
    sections = result.get("sections")
    if sections:
        render_sections(sections)
    else:
        rows = result.get("result_rows", [])
        render_result_table(rows)


def render_mode_summary(result: dict) -> None:
    """Summary: clean readable text."""
    text = result.get("summary_text", "")
    if text:
        st.markdown(
            f'<div class="summary-text">{html.escape(text)}</div>',
            unsafe_allow_html=True,
        )

    # Render exec summary card if present
    render_executive_summary(result)


def render_mode_report(result: dict, is_last: bool) -> None:
    """Report: full monospace expander + optional chart/anomaly/exec."""
    report_text = result.get("report", "")
    if report_text:
        with st.expander("\U0001f4c4  View Full Report", expanded=is_last):
            safe_r = html.escape(report_text).replace("$", "&#36;")
            st.markdown(f'<div class="report-pre">{safe_r}</div>', unsafe_allow_html=True)

    # Bubble up chart
    if result.get("figure_json"):
        try:
            fig = pio.from_json(json.dumps(result["figure_json"]))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        except Exception:
            pass

    render_anomaly_table(result)
    render_executive_summary(result)


def render_mode_anomaly(result: dict) -> None:
    """Anomaly: red-styled anomaly table."""
    render_anomaly_table(result)


def render_mode_default(result: dict, is_last: bool) -> None:
    """Default: result table(s) if available, otherwise report expander."""
    sections = result.get("sections")
    rows = result.get("result_rows", [])
    if sections:
        render_sections(sections)
    elif rows:
        render_result_table(rows)

    report_text = result.get("report", "")
    if report_text:
        with st.expander("\U0001f4c4  View Full Report", expanded=is_last and not rows):
            safe_r = html.escape(report_text).replace("$", "&#36;")
            st.markdown(f'<div class="report-pre">{safe_r}</div>', unsafe_allow_html=True)

    # Bubble up chart / anomaly / exec
    if result.get("figure_json"):
        try:
            fig = pio.from_json(json.dumps(result["figure_json"]))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        except Exception:
            pass

    render_anomaly_table(result)
    render_executive_summary(result)


# ── Render: anomaly + executive summary (shared helpers) ─────────────────────

def render_anomaly_table(result: dict) -> None:
    anomalies = result.get("anomalies")
    if not anomalies or result.get("anomaly_count", 0) == 0:
        return

    interp = result.get("interpretation", "")
    interp_html = (
        f'<div class="anomaly-interp">{html.escape(interp)}</div>' if interp else ""
    )

    cols = list(anomalies[0].keys())
    hdr  = "".join(f"<th>{html.escape(c.replace('_', ' ').title())}</th>" for c in cols)
    body = "".join(
        "<tr>" + "".join(
            f"<td>{html.escape(str(row.get(c, '')))}</td>" for c in cols
        ) + "</tr>"
        for row in anomalies
    )
    count = result["anomaly_count"]
    st.markdown(
        f'<div class="anomaly-section">'
        f'<div class="anomaly-header">\u26a0 {count} anomal{"y" if count == 1 else "ies"} detected</div>'
        f'{interp_html}'
        f'<table class="anomaly-table"><thead><tr>{hdr}</tr></thead>'
        f'<tbody>{body}</tbody></table>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_executive_summary(result: dict) -> None:
    headline = result.get("exec_headline", "")
    insights = result.get("exec_insights", [])
    action   = result.get("exec_action", "")
    risks    = result.get("exec_risks", [])

    if not headline and not insights:
        return

    hl_html = (
        f'<div class="exec-headline">{html.escape(headline)}</div>' if headline else ""
    )
    bl_html = ""
    if insights:
        items   = "".join(f"<li>{html.escape(b)}</li>" for b in insights)
        bl_html = (
            '<div class="exec-section-label">Key Insights</div>'
            f'<ul class="exec-insights">{items}</ul>'
        )
    ac_html = ""
    if action:
        ac_html = (
            '<div class="exec-action-wrap"><div class="exec-action">'
            '<div class="exec-action-label">Recommended Action</div>'
            f'<div class="exec-action-text">{html.escape(action)}</div>'
            '</div></div>'
        )

    st.markdown(
        '<div class="exec-card">'
        '<div class="exec-card-header">'
        '<span class="exec-card-label">Executive Summary</span>'
        '</div>'
        f'{hl_html}{bl_html}{ac_html}'
        '</div>',
        unsafe_allow_html=True,
    )

    if risks:
        all_cols  = list(risks[0].keys())
        data_cols = [c for c in all_cols if c != "risk_reason"]
        r_cols    = data_cols + (["risk_reason"] if "risk_reason" in all_cols else [])
        hdr = "".join(
            f'<th>{html.escape(c.replace("_", " ").title())}</th>' for c in r_cols
        )
        body = ""
        for row in risks:
            cells = ""
            for c in r_cols:
                cls = ' class="risk-reason"' if c == "risk_reason" else ""
                cells += f'<td{cls}>{html.escape(str(row.get(c, "")))}</td>'
            body += f"<tr>{cells}</tr>"
        st.markdown(
            f'<div class="risk-section">'
            f'<div class="risk-header">'
            f'\u26a1 {len(risks)} at-risk area{"s" if len(risks) != 1 else ""}'
            f'</div>'
            f'<table class="risk-table"><thead><tr>{hdr}</tr></thead>'
            f'<tbody>{body}</tbody></table>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ── Render: user bubble ──────────────────────────────────────────────────────

def render_user_bubble(text: str, ts: str = "") -> None:
    safe = html.escape(text)
    st.markdown(
        f'<div class="user-row"><div class="user-bubble">{safe}</div></div>'
        f'<div class="ts-right">{ts}</div>',
        unsafe_allow_html=True,
    )


def render_typing_indicator():
    ph = st.empty()
    ph.markdown("""
        <div class="typing-row">
            <div class="typing-bubble">
                <span class="typing-label">Analyzing data</span>
                <div class="typing-dots">
                    <span></span><span></span><span></span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    return ph


# ── Render: assistant bubble (main dispatcher) ───────────────────────────────

def render_assistant_bubble(result: dict, msg_idx: int, is_last: bool, ts: str = "") -> None:
    # Error state
    if result.get("status") == "error":
        msg = html.escape(result.get("message", "An unknown error occurred."))
        st.markdown(
            f'<div class="error-row"><div class="error-bubble">\u26a0&nbsp;&nbsp;{msg}</div></div>'
            f'<div class="ts-left">{ts}</div>',
            unsafe_allow_html=True,
        )
        return

    response_mode = result.get("response_mode", "default")
    finding       = result.get("finding", "")
    follow_ups    = result.get("follow_up_questions", [])
    orig_query    = result.get("_query", "")

    # Check if all agent results errored — no useful data at all
    has_rows = bool(
        result.get("result_rows")
        or result.get("supporting_data")
        or result.get("sections")
        or result.get("figure_json")
        or result.get("anomalies")
    )
    report_text = result.get("report", "")
    if not has_rows and not finding and report_text:
        # Likely an error wrapped in a report — show friendly message
        # Extract the first error reason from the report
        error_reason = ""
        for line in report_text.splitlines():
            stripped = line.strip().strip("─").strip()
            if stripped and "error" not in stripped.lower() and "status" not in stripped.lower():
                error_reason = stripped
                break
        if not error_reason:
            error_reason = "The query could not be processed."
        safe_reason = html.escape(error_reason)
        st.markdown(
            f'<div class="ts-left">{ts}</div>'
            f'<div style="background:#1a1a1a; border:1px solid rgba(245,158,11,0.3); '
            f'border-left:3px solid #f59e0b; border-radius:8px; padding:14px 18px; '
            f'margin:14px 15% 12px 0; font-size:14px; color:#a0a0a0; line-height:1.6;">'
            f'Sorry, I couldn\'t process that query. {safe_reason}</div>',
            unsafe_allow_html=True,
        )
        render_follow_ups(follow_ups, msg_idx)
        return

    # Timestamp
    st.markdown(f'<div class="ts-left">{ts}</div>', unsafe_allow_html=True)

    # "You asked:" quote
    render_you_asked(orig_query)

    # Finding sentence (always shown first)
    render_finding(finding)

    # Mode-specific content
    if response_mode == "direct":
        render_mode_direct(result)

    elif response_mode == "chart":
        render_mode_chart(result)

    elif response_mode == "comparison":
        render_mode_comparison(result)

    elif response_mode == "list":
        render_mode_list(result)

    elif response_mode == "summary":
        render_mode_summary(result)

    elif response_mode == "report":
        render_mode_report(result, is_last)

    elif response_mode == "anomaly":
        render_mode_anomaly(result)

    else:  # "default"
        render_mode_default(result, is_last)

    # Action row: PDF download + copy + pin + re-run
    report_text = result.get("report", "")
    c1, c2, c3, c4, _pad = st.columns([1.2, 1, 0.8, 1, 6])

    with c1:
        render_pdf_download(orig_query, result, msg_idx)

    with c2:
        st.markdown('<div class="action-ghost">', unsafe_allow_html=True)
        if st.button("\U0001f4cb Copy", key=f"copy_{msg_idx}", help="Copy finding to clipboard"):
            text_to_copy = finding or report_text or ""
            safe_js = (
                text_to_copy
                .replace("\\", "\\\\")
                .replace("`", "\\`")
                .replace("$", "\\$")
            )
            components.html(
                f"<script>"
                f"navigator.clipboard.writeText(`{safe_js}`)"
                f".catch(function(){{"
                f"  var t=document.createElement('textarea');"
                f"  t.value=`{safe_js}`;"
                f"  document.body.appendChild(t);"
                f"  t.select();"
                f"  document.execCommand('copy');"
                f"  document.body.removeChild(t);"
                f"}});"
                f"</script>",
                height=0,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with c3:
        if finding:
            st.markdown('<div class="action-ghost">', unsafe_allow_html=True)
            if st.button("\U0001f4cc Pin", key=f"pin_{msg_idx}", help="Pin this finding"):
                pinned = st.session_state.pinned
                if finding not in pinned:
                    pinned.insert(0, finding)
                    st.session_state.pinned = pinned[:10]
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    with c4:
        if orig_query:
            st.markdown('<div class="action-ghost">', unsafe_allow_html=True)
            if st.button("\U0001f504 Re-run", key=f"rerun_{msg_idx}"):
                st.session_state.input_value = orig_query
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    # Follow-up suggestions
    render_follow_ups(follow_ups, msg_idx)


# ── Main chat area ────────────────────────────────────────────────────────────

if not st.session_state.messages:
    st.markdown(f"""
        <div class="welcome-wrap">
            <div class="welcome-card">
                <div class="welcome-icon"><img src='data:image/png;base64,{_ICON_B64}' style='width:64px; height:64px;' /></div>
                <div class="welcome-title">OLAP Sales Assistant</div>
                <div class="welcome-sub">
                    Ask natural-language questions about your retail data
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.markdown(
        '<div class="welcome-hint">\u2190 Pick a quick action or type below</div>',
        unsafe_allow_html=True,
    )
else:
    last_idx = len(st.session_state.messages) - 1
    for i, msg in enumerate(st.session_state.messages):
        ts = msg.get("ts", "")
        if msg["role"] == "user":
            render_user_bubble(msg["content"], ts=ts)
        else:
            render_assistant_bubble(
                msg["content"], msg_idx=i, is_last=(i == last_idx), ts=ts
            )

# ── Input area ────────────────────────────────────────────────────────────────

# If a card, history item, or follow-up was clicked, inject JS to pre-populate.
if st.session_state.input_value:
    _tpl = (
        st.session_state.input_value
        .replace("\\", "\\\\")
        .replace("`", "\\`")
    )
    st.session_state.input_value = ""
    components.html(f"""
    <script>
    (function() {{
        function inject() {{
            var ta = window.parent.document.querySelector(
                '[data-testid="stChatInputTextArea"]'
            );
            if (!ta) {{ setTimeout(inject, 100); return; }}
            var setter = Object.getOwnPropertyDescriptor(
                window.HTMLTextAreaElement.prototype, 'value'
            ).set;
            setter.call(ta, `{_tpl}`);
            ta.dispatchEvent(new Event('input', {{ bubbles: true }}));
            ta.focus();
        }}
        inject();
    }})();
    </script>
    """, height=0)

user_input = st.chat_input("Ask about your sales data...")

if user_input:
    st.session_state.show_welcome = False
    st.session_state.last_query   = user_input
    _update_history(user_input)
    ts_now = _ts()

    # Display user bubble immediately before API call
    st.session_state.messages.append({"role": "user", "content": user_input, "ts": ts_now})
    render_user_bubble(user_input, ts=ts_now)

    # Show typing indicator while waiting
    typing_ph = render_typing_indicator()

    # Call API
    result = call_query_api(user_input, st.session_state.session_id)
    result["_query"] = user_input

    # Clear typing indicator
    typing_ph.empty()

    st.session_state.messages.append({
        "role": "assistant",
        "content": result,
        "ts": _ts(),
    })
    st.rerun()

st.markdown('<div class="kb-hint">Press Enter to send</div>', unsafe_allow_html=True)
