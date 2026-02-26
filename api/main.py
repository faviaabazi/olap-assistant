"""
FastAPI entry point for the OLAP Assistant.

Endpoints
─────────
POST /query   — Run a natural-language query through the Orchestrator
POST /reset   — Clear conversation history for a session
GET  /health  — Liveness / DB connectivity check
"""

from __future__ import annotations

import os
import pathlib
import sys

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Path setup so `orchestrator` and `agents` are importable ──────────────────
_ROOT = pathlib.Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

load_dotenv(_ROOT / ".env")

from orchestrator.orchestrator import Orchestrator  # noqa: E402

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="OLAP Assistant API",
    description=(
        "Multi-agent OLAP assistant for retail sales data (Jan 2022 – Dec 2024). "
        "Send natural-language queries and receive structured analytical reports."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Session store: one Orchestrator instance per session_id ───────────────────

_DB_PATH = str(_ROOT / "olap.duckdb")
_sessions: dict[str, Orchestrator] = {}


def _get_orchestrator(session_id: str) -> Orchestrator:
    """Return the existing Orchestrator for *session_id*, creating one if needed."""
    if session_id not in _sessions:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        _sessions[session_id] = Orchestrator(api_key=api_key, db_path=_DB_PATH)
    return _sessions[session_id]


# ── Request / response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    session_id: str


class ResetRequest(BaseModel):
    session_id: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post(
    "/query",
    summary="Run an OLAP query",
    description=(
        "Submit a natural-language question about the retail sales dataset. "
        "The Orchestrator plans, executes agents, and returns a structured response."
    ),
    tags=["Query"],
)
def query(body: QueryRequest) -> dict:
    orchestrator = _get_orchestrator(body.session_id)
    try:
        return orchestrator.handle_query(body.query)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post(
    "/reset",
    summary="Reset a session",
    description=(
        "Clear the conversation history for the given session, "
        "allowing a fresh context for subsequent queries."
    ),
    tags=["Session"],
)
def reset(body: ResetRequest) -> dict:
    if body.session_id in _sessions:
        del _sessions[body.session_id]
    return {"status": "ok"}


@app.get(
    "/health",
    summary="Health check",
    description=(
        "Returns the API status and verifies that the DuckDB database "
        "is reachable by running a trivial query."
    ),
    tags=["Health"],
)
def health() -> dict:
    try:
        import duckdb
        con = duckdb.connect(_DB_PATH, read_only=True)
        con.execute("SELECT 1").fetchone()
        con.close()
        db_status = "connected"
    except Exception:
        db_status = "unavailable"

    return {"status": "ok", "database": db_status}
