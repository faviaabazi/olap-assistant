"""
Streamlit chat interface for the OLAP Assistant.

Run with:
    streamlit run frontend/app.py
"""

from __future__ import annotations

import html
import json
import uuid
from datetime import datetime

import pandas as pd
import plotly.io as pio
import requests
import streamlit as st
import streamlit.components.v1 as components

# ── Constants ─────────────────────────────────────────────────────────────────

API_BASE = "http://localhost:8000"

PROMPT_TEMPLATES = [
    ("📊", "Chart",      "Show me a chart of revenue by ______"),
    ("📈", "Growth",     "Show year-over-year growth of ______ by region"),
    ("🔍", "Drill Down", "Drill down into ______ by country"),
    ("🏆", "Top 5",      "Top 5 ______ by revenue"),
    ("⚠️", "Anomalies",  "Detect profit anomalies by ______"),
    ("📋", "Summary",    "Give me an executive summary of ______"),
]

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="OLAP Assistant",
    page_icon="📊",
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
    content: "▸"; position: absolute; left: 0; color: #f59e0b; font-size: 11px; top: 7px;
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
    background: transparent !important;
    border: 1px solid #f59e0b !important;
    border-radius: 8px !important;
    color: #f59e0b !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    padding: 8px 16px !important;
    width: 100% !important;
    transition: background 0.2s !important;
}
.new-convo-btn > div > button:hover {
    background: rgba(245, 158, 11, 0.12) !important;
}

/* ── Quick action card buttons ── */
.card-btn > div > button {
    width: 100% !important;
    background: #222222 !important;
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
    transition: background 0.2s, border-color 0.2s !important;
}
.card-btn > div > button:hover {
    background: #2a2a2a !important;
    border-color: #f59e0b !important;
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
    color: #fafafa !important;
    background: transparent !important;
}

/* ── Response action ghost buttons ── */
.action-ghost > div > button,
.action-ghost [data-testid="stDownloadButton"] > button {
    background: none !important;
    border: none !important;
    color: #737373 !important;
    font-size: 12px !important;
    padding: 4px 8px !important;
    border-radius: 6px !important;
    transition: color 0.2s, background 0.2s !important;
    min-width: auto !important;
}
.action-ghost > div > button:hover,
.action-ghost [data-testid="stDownloadButton"] > button:hover {
    color: #f59e0b !important;
    background: rgba(245, 158, 11, 0.08) !important;
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

/* ── Comparison table ── */
.comparison-table {
    width: 100%; border-collapse: collapse; font-size: 13px; margin: 10px 0;
}
.comparison-table th {
    background: #2a2a2a; color: #f59e0b; padding: 8px 12px; text-align: left;
    border-bottom: 2px solid rgba(245,158,11,0.3);
    font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em;
    white-space: nowrap;
}
.comparison-table td {
    padding: 8px 12px; border-bottom: 1px solid #2a2a2a; color: #fafafa;
    white-space: nowrap;
}
.comparison-table tr:last-child td { border-bottom: none; }
.comparison-table tr:hover td { background: #2a2a2a; }
.comparison-table td.null-cell { color: #555555; }
.comparison-takeaway {
    font-size: 13px; color: #a0a0a0; font-style: italic;
    margin-top: 14px; padding: 8px 14px;
    border-left: 3px solid #f59e0b40; line-height: 1.65;
}

/* ── Simple table ── */
.simple-table {
    width: 100%; border-collapse: collapse; font-size: 13px; margin: 10px 0;
}
.simple-table th {
    background: #2a2a2a; color: #737373; padding: 8px 12px; text-align: left;
    border-bottom: 1px solid #333333;
    font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em;
    white-space: nowrap;
}
.simple-table td {
    padding: 8px 12px; border-bottom: 1px solid #2a2a2a; color: #fafafa;
    white-space: nowrap;
}
.simple-table tr.top-row td { color: #f59e0b; font-weight: 600; }
.simple-table tr:last-child td { border-bottom: none; }
.simple-table tr:hover td { background: #2a2a2a; }
.simple-summary {
    font-size: 13px; color: #a0a0a0; margin-top: 10px; line-height: 1.6;
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


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
        <div style='padding:24px 0 12px; text-align:center;'>
            <div style='font-size:48px; line-height:1;'>📊</div>
            <div style='font-size:20px; font-weight:700; color:#fafafa; margin:10px 0 4px;'>
                OLAP Assistant
            </div>
            <div style='font-size:11px; color:#737373; letter-spacing:0.08em;'>
                Retail Sales · 2022–2024
            </div>
        </div>
        <hr class='sidebar-divider' />
    """, unsafe_allow_html=True)

    # New conversation
    st.markdown('<div class="new-convo-btn">', unsafe_allow_html=True)
    if st.button("⊕  New Conversation", use_container_width=True, key="new_convo"):
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
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # Query history
    st.markdown('<span class="sidebar-section-label">RECENT</span>', unsafe_allow_html=True)
    if st.session_state.query_history:
        for qi, q in enumerate(st.session_state.query_history):
            label = f"🕐 {q[:35]}{'...' if len(q) > 35 else ''}"
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

    # Quick action cards — 2-column grid
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


# ── Render helpers ────────────────────────────────────────────────────────────

def _fmt_cell(val) -> str:
    """Format a table cell value: numbers get thousands separators, None → em-dash."""
    if val is None:
        return None  # caller handles null-cell class
    try:
        f = float(val)
        if f == int(f):
            return f"{int(f):,}"
        return f"{f:,.2f}"
    except (TypeError, ValueError):
        return html.escape(str(val))


def render_comparison(result: dict) -> None:
    """Render a comparison table: highlights % columns in green/red, shows takeaway."""
    rows = result.get("result_rows", [])
    if not rows:
        return

    cols = list(rows[0].keys())
    hdr  = "".join(
        f'<th>{html.escape(c.replace("_", " ").title())}</th>' for c in cols
    )
    body = ""
    for row in rows:
        cells = ""
        for c in cols:
            val = row.get(c)
            if c.endswith("_pct") and val is not None:
                try:
                    pct   = float(val)
                    color = "#10b981" if pct >= 0 else "#ef4444"
                    sign  = "+" if pct > 0 else ""
                    cells += (
                        f'<td style="color:{color}; font-weight:600;">'
                        f'{sign}{pct:.2f}%</td>'
                    )
                except (TypeError, ValueError):
                    cells += '<td class="null-cell">—</td>'
            elif val is None:
                cells += '<td class="null-cell">—</td>'
            else:
                display = _fmt_cell(val)
                cells += f"<td>{display}</td>"
        body += f"<tr>{cells}</tr>"

    st.markdown(
        f'<table class="comparison-table"><thead><tr>{hdr}</tr></thead>'
        f'<tbody>{body}</tbody></table>',
        unsafe_allow_html=True,
    )

    takeaway = result.get("comparison_takeaway", "")
    if takeaway:
        st.markdown(
            f'<div class="comparison-takeaway">{html.escape(takeaway)}</div>',
            unsafe_allow_html=True,
        )


def render_simple(result: dict) -> None:
    """Render a simple ranked/margin table with the top row highlighted in amber."""
    rows = result.get("result_rows", [])
    if not rows:
        return

    cols = list(rows[0].keys())
    hdr  = "".join(
        f'<th>{html.escape(c.replace("_", " ").title())}</th>' for c in cols
    )
    body = ""
    for i, row in enumerate(rows):
        row_class = ' class="top-row"' if i == 0 else ""
        cells = ""
        for c in cols:
            val = row.get(c)
            if val is None:
                cells += '<td class="null-cell">—</td>'
            else:
                display = _fmt_cell(val)
                cells += f"<td>{display}</td>"
        body += f"<tr{row_class}>{cells}</tr>"

    st.markdown(
        f'<table class="simple-table"><thead><tr>{hdr}</tr></thead>'
        f'<tbody>{body}</tbody></table>',
        unsafe_allow_html=True,
    )

    summary = result.get("simple_summary", "")
    if summary:
        st.markdown(
            f'<div class="simple-summary">{html.escape(summary)}</div>',
            unsafe_allow_html=True,
        )


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


def render_plotly_chart(result: dict) -> None:
    figure_json = result.get("figure_json")
    if not figure_json:
        return
    try:
        fig = pio.from_json(json.dumps(figure_json))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    except Exception:
        return

    raw = result.get("table") or result.get("rows")
    if raw:
        with st.expander("🗂  View Raw Data"):
            try:
                df = pd.DataFrame(raw)
                st.dataframe(df, use_container_width=True)
            except Exception:
                st.json(raw)


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
        f'<div class="anomaly-header">⚠ {count} anomal{"y" if count == 1 else "ies"} detected</div>'
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
        cols      = data_cols + (["risk_reason"] if "risk_reason" in all_cols else [])
        hdr = "".join(
            f'<th>{html.escape(c.replace("_", " ").title())}</th>' for c in cols
        )
        body = ""
        for row in risks:
            cells = ""
            for c in cols:
                cls = ' class="risk-reason"' if c == "risk_reason" else ""
                cells += f'<td{cls}>{html.escape(str(row.get(c, "")))}</td>'
            body += f"<tr>{cells}</tr>"
        st.markdown(
            f'<div class="risk-section">'
            f'<div class="risk-header">'
            f'⚡ {len(risks)} at-risk area{"s" if len(risks) != 1 else ""}'
            f'</div>'
            f'<table class="risk-table"><thead><tr>{hdr}</tr></thead>'
            f'<tbody>{body}</tbody></table>'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_assistant_bubble(result: dict, msg_idx: int, is_last: bool, ts: str = "") -> None:
    # Error state
    if result.get("status") == "error":
        msg = html.escape(result.get("message", "An unknown error occurred."))
        st.markdown(
            f'<div class="error-row"><div class="error-bubble">⚠&nbsp;&nbsp;{msg}</div></div>'
            f'<div class="ts-left">{ts}</div>',
            unsafe_allow_html=True,
        )
        return

    # Bubble header content
    response_mode = result.get("response_mode", "report")
    exec_headline = result.get("exec_headline", "")
    section_count = result.get("section_count", 0)
    report_text   = result.get("report", "")

    if exec_headline:
        excerpt = html.escape(exec_headline[:160]) + ("…" if len(exec_headline) > 160 else "")
        bubble_body = (
            f'<span style="font-weight:600; color:#f59e0b; font-size:15px;">'
            f'{excerpt}</span>'
        )
    elif response_mode == "comparison":
        n = len(result.get("result_rows", []))
        bubble_body = f'<strong style="color:#fafafa;">📊 {n}-row comparison</strong>'
    elif response_mode == "simple":
        n = len(result.get("result_rows", []))
        bubble_body = f'<strong style="color:#fafafa;">✓ {n} result{"s" if n != 1 else ""}</strong>'
    elif response_mode == "chart":
        bubble_body = '<strong style="color:#fafafa;">📊 Chart ready</strong>'
    else:
        s = "sections" if section_count != 1 else "section"
        bubble_body = f'<strong style="color:#fafafa;">{section_count} {s}</strong>'

    st.markdown(
        f'<div class="asst-row"><div class="asst-bubble">{bubble_body}</div></div>'
        f'<div class="ts-left">{ts}</div>',
        unsafe_allow_html=True,
    )

    # Content — dispatch on response mode
    if response_mode == "chart":
        render_plotly_chart(result)

    elif response_mode == "executive":
        render_executive_summary(result)

    elif response_mode == "comparison":
        render_comparison(result)

    elif response_mode == "simple":
        render_simple(result)

    else:  # "report" — full expander + optional chart / anomaly / exec summary
        if report_text:
            with st.expander("📄  View Full Report", expanded=is_last):
                safe_r = html.escape(report_text).replace("$", "&#36;")
                st.markdown(f'<div class="report-pre">{safe_r}</div>', unsafe_allow_html=True)
        render_plotly_chart(result)
        render_anomaly_table(result)
        render_executive_summary(result)

    # Response action buttons
    orig_query = result.get("_query", "")
    c1, c2, c3, _pad = st.columns([1, 1, 1, 7])

    with c1:
        st.markdown('<div class="action-ghost">', unsafe_allow_html=True)
        if st.button("📋 Copy", key=f"copy_{msg_idx}", help="Copy report to clipboard"):
            text_to_copy = report_text or exec_headline or ""
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

    with c2:
        if report_text:
            st.markdown('<div class="action-ghost">', unsafe_allow_html=True)
            st.download_button(
                "💾 Download",
                data=report_text,
                file_name="report.txt",
                mime="text/plain",
                key=f"dl_{msg_idx}",
            )
            st.markdown("</div>", unsafe_allow_html=True)

    with c3:
        if orig_query:
            st.markdown('<div class="action-ghost">', unsafe_allow_html=True)
            if st.button("🔄 Re-run", key=f"rerun_{msg_idx}"):
                st.session_state.input_value = orig_query
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)


# ── Main chat area ────────────────────────────────────────────────────────────

if not st.session_state.messages:
    st.markdown("""
        <div class="welcome-wrap">
            <div class="welcome-card">
                <div class="welcome-icon">📊</div>
                <div class="welcome-title">OLAP Sales Assistant</div>
                <div class="welcome-sub">
                    Ask natural-language questions about your retail data
                </div>
                <div class="feature-pills">
                    <span class="feature-pill">📅 2022–2024</span>
                    <span class="feature-pill">🌍 4 Regions</span>
                    <span class="feature-pill">📦 4 Categories</span>
                </div>
                <div class="welcome-hint">← Pick a quick action or type below</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
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

# If a card or history item was clicked, inject JS to pre-populate the textarea.
# This does NOT send the query — the user edits the text and presses Enter themselves.
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

    st.session_state.messages.append({"role": "user", "content": user_input, "ts": ts_now})

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
