"""
Streamlit chat interface for the OLAP Assistant.

Run with:
    streamlit run frontend/app.py
"""

from __future__ import annotations

import html
import json
import uuid

import plotly.io as pio
import requests
import streamlit as st

# ── Constants ─────────────────────────────────────────────────────────────────

API_BASE = "http://localhost:8000"

EXAMPLE_QUERIES = [
    "Top 5 countries by revenue",
    "YoY growth by region",
    "Drill down Europe by country",
    "Profit margins by category",
    "Compare Q3 vs Q4 2024",
    "Show me a chart of revenue by region",
    "Detect profit anomalies by country",
]

# ── Page configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="OLAP Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Chrome removal ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── App background ── */
.stApp { background-color: #0d0f16; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #12141e;
    border-right: 1px solid #1c1f30;
}
[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    background: #191c2a;
    color: #8892b0;
    border: 1px solid #222540;
    border-radius: 8px;
    padding: 8px 12px;
    text-align: left;
    font-size: 13px;
    transition: all 0.18s ease;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #1e2235;
    border-color: #6366f1;
    color: #e2e8f0;
    transform: translateX(3px);
}

/* New conversation button */
.new-convo-btn > div > button {
    background: linear-gradient(135deg, #6366f1 0%, #818cf8 100%) !important;
    color: #fff !important;
    border: none !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    border-radius: 10px !important;
    padding: 10px 16px !important;
    box-shadow: 0 4px 16px rgba(99, 102, 241, 0.35);
    transition: all 0.2s ease !important;
}
.new-convo-btn > div > button:hover {
    box-shadow: 0 6px 22px rgba(99, 102, 241, 0.5) !important;
    transform: translateY(-2px) !important;
}

/* ── User bubble (right-aligned) ── */
.user-row {
    display: flex;
    justify-content: flex-end;
    margin: 16px 0 4px 18%;
}
.user-bubble {
    background: linear-gradient(135deg, #6366f1 0%, #818cf8 100%);
    color: #fff;
    padding: 12px 18px;
    border-radius: 20px 20px 5px 20px;
    font-size: 14.5px;
    line-height: 1.6;
    box-shadow: 0 4px 16px rgba(99, 102, 241, 0.3);
    word-break: break-word;
}

/* ── Assistant bubble (left-aligned) ── */
.asst-row {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    margin: 4px 18% 2px 0;
}
.asst-avatar {
    font-size: 22px;
    flex-shrink: 0;
    padding-top: 3px;
}
.asst-bubble {
    background: #191c2a;
    color: #b8c4da;
    padding: 12px 18px;
    border-radius: 5px 20px 20px 20px;
    font-size: 14.5px;
    line-height: 1.6;
    border: 1px solid #222540;
    box-shadow: 0 3px 12px rgba(0, 0, 0, 0.4);
    word-break: break-word;
}

/* ── Error bubble ── */
.error-row {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    margin: 4px 18% 2px 0;
}
.error-bubble {
    background: rgba(239, 68, 68, 0.07);
    border: 1px solid rgba(239, 68, 68, 0.28);
    color: #fca5a5;
    padding: 12px 18px;
    border-radius: 5px 20px 20px 20px;
    font-size: 14px;
    line-height: 1.6;
    word-break: break-word;
}

/* ── Expander (report container) ── */
[data-testid="stExpander"] {
    background: #101320 !important;
    border: 1px solid #1e2235 !important;
    border-radius: 10px !important;
    margin: 6px 0 4px 42px;
}
[data-testid="stExpander"] summary {
    color: #6366f1 !important;
    font-size: 13px !important;
    font-weight: 600 !important;
}
[data-testid="stExpander"] summary:hover {
    color: #818cf8 !important;
}

/* ── Monospace report block ── */
.report-pre {
    font-family: 'JetBrains Mono', 'Cascadia Code', 'Fira Code', 'Courier New', monospace;
    font-size: 12px;
    line-height: 1.65;
    color: #a8b8d0;
    white-space: pre;
    overflow-x: auto;
    background: #090b12;
    padding: 20px 22px;
    border-radius: 7px;
    border: 1px solid #171a28;
    margin: 2px 0 6px 0;
}

/* ── Routing metadata ── */
.reasoning-line {
    font-size: 11.5px;
    color: #384060;
    font-style: italic;
    margin: 4px 0 5px 42px;
    line-height: 1.45;
}
.badge-row {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    margin: 0 0 14px 42px;
}
.badge {
    display: inline-flex;
    align-items: center;
    background: #13151f;
    border: 1px solid #222540;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 11px;
    color: #6366f1;
    font-family: 'Courier New', monospace;
    letter-spacing: 0.02em;
}

/* ── Welcome screen ── */
.welcome-wrap {
    display: flex;
    justify-content: center;
    padding-top: 72px;
}
.welcome-card {
    background: #191c2a;
    border: 1px solid #222540;
    border-radius: 20px;
    padding: 44px 52px;
    text-align: center;
    max-width: 580px;
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.55);
}
.welcome-icon { font-size: 52px; margin-bottom: 14px; }
.welcome-title {
    font-size: 26px;
    font-weight: 700;
    background: linear-gradient(135deg, #818cf8, #c4b5fd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 14px;
}
.welcome-body {
    font-size: 14.5px;
    color: #5a6480;
    line-height: 1.75;
}
.welcome-hint {
    margin-top: 22px;
    font-size: 13px;
    color: #2d3454;
    border-top: 1px solid #1e2235;
    padding-top: 18px;
}

/* ── Stats pills ── */
.stat-row {
    display: flex;
    gap: 10px;
    justify-content: center;
    margin-top: 18px;
    flex-wrap: wrap;
}
.stat-pill {
    background: #13151f;
    border: 1px solid #222540;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 12px;
    color: #818cf8;
}

/* ── Chart container ── */
.chart-wrap {
    margin: 6px 0 4px 42px;
    border: 1px solid #1e2235;
    border-radius: 10px;
    overflow: hidden;
}
.chart-reasoning {
    font-size: 11.5px;
    color: #384060;
    font-style: italic;
    margin: 4px 0 10px 42px;
}

/* ── Anomaly section ── */
.anomaly-section { margin: 8px 0 4px 42px; }
.anomaly-header {
    font-size: 12px;
    font-weight: 700;
    color: #ef4444;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    margin-bottom: 7px;
}
.anomaly-interp {
    font-size: 13px;
    color: #b8c4da;
    line-height: 1.65;
    background: rgba(239, 68, 68, 0.06);
    border: 1px solid rgba(239, 68, 68, 0.22);
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 10px;
}
.anomaly-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'JetBrains Mono', 'Cascadia Code', 'Courier New', monospace;
    font-size: 11.5px;
}
.anomaly-table th {
    background: rgba(239, 68, 68, 0.12);
    color: #ef4444;
    padding: 6px 10px;
    text-align: left;
    border-bottom: 1px solid rgba(239, 68, 68, 0.25);
    white-space: nowrap;
}
.anomaly-table td {
    background: rgba(239, 68, 68, 0.05);
    color: #fca5a5;
    padding: 5px 10px;
    border-bottom: 1px solid rgba(239, 68, 68, 0.1);
    white-space: nowrap;
}
.anomaly-table tr:last-child td { border-bottom: none; }

/* ── Chat input ── */
[data-testid="stChatInput"] > div {
    background: #191c2a !important;
    border: 1px solid #2a2d48 !important;
    border-radius: 14px !important;
    box-shadow: 0 -2px 20px rgba(0,0,0,0.3);
}
[data-testid="stChatInput"] textarea {
    color: #dde4f0 !important;
    font-size: 14.5px !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #3a3f5c !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state initialisation ──────────────────────────────────────────────

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    # Each entry: {"role": "user"|"assistant", "content": str|dict}
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
        <div style='padding: 22px 0 10px; text-align: center;'>
            <div style='font-size: 46px;'>📊</div>
            <div style='font-size: 21px; font-weight: 700; color: #e2e8f0; margin: 10px 0 5px;'>
                OLAP Assistant
            </div>
            <div style='font-size: 11px; color: #383d5c; letter-spacing: 0.1em; text-transform: uppercase;'>
                Retail Sales · 2022 – 2024
            </div>
        </div>
        <div style='border-top: 1px solid #1c1f30; margin: 14px 0 18px;'></div>
    """, unsafe_allow_html=True)

    # New conversation button
    st.markdown('<div class="new-convo-btn">', unsafe_allow_html=True)
    if st.button("⊕  New Conversation", use_container_width=True, key="new_convo"):
        try:
            requests.post(
                f"{API_BASE}/reset",
                json={"session_id": st.session_state.session_id},
                timeout=5,
            )
        except Exception:
            pass  # server may be offline; still reset locally
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    short_id = st.session_state.session_id[:8] + "…"
    st.markdown(
        f"<div style='font-size: 10.5px; color: #252840; font-family: monospace;"
        f"text-align: center; margin: 8px 0 20px;'>{short_id}</div>",
        unsafe_allow_html=True,
    )

    # Example queries
    st.markdown("""
        <div style='font-size: 11.5px; font-weight: 700; color: #3d4466;
        text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 9px;'>
            Quick Examples
        </div>
    """, unsafe_allow_html=True)

    for example in EXAMPLE_QUERIES:
        if st.button(f"› {example}", use_container_width=True, key=f"ex__{example}"):
            st.session_state.pending_query = example

    st.markdown("""
        <div style='border-top: 1px solid #1c1f30; margin: 20px 0 14px;'></div>
        <div style='font-size: 10.5px; color: #252840; text-align: center; line-height: 1.8;'>
            Claude · DuckDB · FastAPI · Streamlit
        </div>
    """, unsafe_allow_html=True)

# ── API helpers ───────────────────────────────────────────────────────────────

def call_query_api(query: str, session_id: str) -> dict:
    """POST /query and return parsed JSON, or an error dict on failure."""
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
        return {
            "status": "error",
            "message": "Request timed out (90 s). The query may be too complex.",
        }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}

# ── Render helpers ────────────────────────────────────────────────────────────

def render_plotly_chart(result: dict) -> None:
    """Render a Plotly figure from result["figure_json"] if present."""
    figure_json = result.get("figure_json")
    if not figure_json:
        return
    try:
        fig = pio.from_json(json.dumps(figure_json))
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)
    except Exception:
        return  # silently skip broken figures

    reasoning = result.get("chart_reasoning", "")
    if reasoning:
        st.markdown(
            f'<div class="chart-reasoning">'
            f'📈&nbsp;{html.escape(reasoning)}'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_anomaly_table(result: dict) -> None:
    """Render anomaly rows highlighted in red, plus the interpretation."""
    anomalies = result.get("anomalies")
    if not anomalies or result.get("anomaly_count", 0) == 0:
        return

    interpretation = result.get("interpretation", "")
    interp_html = (
        f'<div class="anomaly-interp">{html.escape(interpretation)}</div>'
        if interpretation else ""
    )

    # Build HTML table from anomaly rows
    cols = list(anomalies[0].keys())
    header_cells = "".join(
        f"<th>{html.escape(c.replace('_', ' ').title())}</th>" for c in cols
    )
    rows_html = ""
    for row in anomalies:
        cells = "".join(
            f"<td>{html.escape(str(row.get(c, '')))}</td>" for c in cols
        )
        rows_html += f"<tr>{cells}</tr>"

    table_html = (
        f'<table class="anomaly-table">'
        f"<thead><tr>{header_cells}</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        f"</table>"
    )

    count = result["anomaly_count"]
    st.markdown(
        f'<div class="anomaly-section">'
        f'<div class="anomaly-header">⚠ {count} anomal{"y" if count == 1 else "ies"} detected</div>'
        f"{interp_html}"
        f"{table_html}"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_user_bubble(text: str) -> None:
    safe = html.escape(text)
    st.markdown(
        f'<div class="user-row"><div class="user-bubble">{safe}</div></div>',
        unsafe_allow_html=True,
    )


def render_assistant_bubble(result: dict, is_last: bool) -> None:
    """Render one assistant turn: summary bubble + optional report expander + metadata."""
    # ── Error state ────────────────────────────────────────────────────────
    if result.get("status") == "error":
        msg = html.escape(result.get("message", "An unknown error occurred."))
        st.markdown(
            f"""
            <div class="error-row">
                <div class="asst-avatar">🤖</div>
                <div class="error-bubble">⚠&nbsp;&nbsp;{msg}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # ── Success state ──────────────────────────────────────────────────────
    report_text   = result.get("report", "")
    section_count = result.get("section_count", 0)
    routing       = result.get("_routing", {})
    steps         = routing.get("steps", [])
    reasoning     = routing.get("reasoning", "")

    # Build a one-line summary for the bubble
    step_chain = " → ".join(
        f"{s.get('agent','?')}.{s.get('method','?')}" for s in steps
    )
    sections_label = f"{section_count} section{'s' if section_count != 1 else ''}"
    chain_html = (
        f"&nbsp;·&nbsp;<code style='font-size:12px; color:#818cf8;'>{html.escape(step_chain)}</code>"
        if step_chain else ""
    )
    bubble_body = f"<strong>{sections_label}</strong>{chain_html}"

    st.markdown(
        f"""
        <div class="asst-row">
            <div class="asst-avatar">🤖</div>
            <div class="asst-bubble">{bubble_body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Full report in expander ────────────────────────────────────────────
    if report_text:
        with st.expander("📄  View Full Report", expanded=is_last):
            safe_report = html.escape(report_text).replace('$', '&#36;')
            st.markdown(
                f'<div class="report-pre">{safe_report}</div>',
                unsafe_allow_html=True,
            )

    # ── Plotly chart (visualization agent) ────────────────────────────────
    render_plotly_chart(result)

    # ── Anomaly table (anomaly detection agent) ────────────────────────────
    render_anomaly_table(result)

    # ── Routing metadata ───────────────────────────────────────────────────
    if reasoning:
        safe_r = html.escape(reasoning)
        st.markdown(
            f'<div class="reasoning-line">{safe_r}</div>',
            unsafe_allow_html=True,
        )
    if steps:
        badges = "".join(
            f'<span class="badge">'
            f'⚡&nbsp;{html.escape(s.get("agent",""))}.{html.escape(s.get("method",""))}'
            f'</span>'
            for s in steps
        )
        st.markdown(
            f'<div class="badge-row">{badges}</div>',
            unsafe_allow_html=True,
        )

# ── Main chat area ────────────────────────────────────────────────────────────

if not st.session_state.messages:
    st.markdown("""
        <div class="welcome-wrap">
            <div class="welcome-card">
                <div class="welcome-icon">📊</div>
                <div class="welcome-title">OLAP Sales Assistant</div>
                <div class="welcome-body">
                    Ask natural-language questions about retail sales data spanning
                    <strong style='color:#818cf8;'>January 2022 – December 2024</strong>.
                </div>
                <div class="stat-row">
                    <span class="stat-pill">🌍 20 countries</span>
                    <span class="stat-pill">🗂 4 regions</span>
                    <span class="stat-pill">📦 4 categories</span>
                    <span class="stat-pill">👥 3 segments</span>
                </div>
                <div class="welcome-hint">
                    ← Pick a quick example from the sidebar, or type a question below.
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
else:
    last_idx = len(st.session_state.messages) - 1
    for i, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            render_user_bubble(msg["content"])
        else:
            render_assistant_bubble(msg["content"], is_last=(i == last_idx))

# ── Chat input ────────────────────────────────────────────────────────────────

user_input = st.chat_input("Ask about your sales data…")

# Sidebar example buttons set pending_query; pick it up in the same render pass
if not user_input and st.session_state.pending_query:
    user_input = st.session_state.pending_query
    st.session_state.pending_query = None

if user_input:
    # Append user message immediately so it appears in the next render
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("🔍  Routing your query through the OLAP agents…"):
        result = call_query_api(user_input, st.session_state.session_id)

    st.session_state.messages.append({"role": "assistant", "content": result})
    st.rerun()
