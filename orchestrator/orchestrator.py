from __future__ import annotations

from typing import Any, Dict, List, Optional

import anthropic
import duckdb

from orchestrator.planner import Planner
from orchestrator.conversation_state import ConversationState
from orchestrator.response_engine import generate_response

from agents.cube_operations import CubeOperationsAgent
from agents.dimension_navigator import DimensionNavigatorAgent
from agents.kpi_calculator import KPICalculatorAgent
from agents.optional.anomaly_detection import AnomalyDetectionAgent
from agents.optional.executive_summary import ExecutiveSummaryAgent
from agents.optional.visualization import VisualizationAgent


def _sanitize_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ensure every numeric field is int or float; replace None with 0."""
    clean: List[Dict[str, Any]] = []
    for row in rows:
        out: Dict[str, Any] = {}
        for k, v in row.items():
            if v is None:
                out[k] = 0
            elif isinstance(v, (int, float, str, bool)):
                out[k] = v
            else:
                # Decimal or other numeric-like → float
                try:
                    out[k] = float(v)
                except (TypeError, ValueError):
                    out[k] = str(v)
        clean.append(out)
    return clean


_AGENT_MAP = {
    "navigator": DimensionNavigatorAgent,
    "cube": CubeOperationsAgent,
    "kpi": KPICalculatorAgent,
    "anomaly": AnomalyDetectionAgent,
    "executive_summary": ExecutiveSummaryAgent,
    "visualization": VisualizationAgent,
}


class Orchestrator:
    """Central execution controller. Routes queries through Planner → Agents → Response."""

    def __init__(self, api_key: str, db_path: str = "olap.duckdb") -> None:
        self.client = anthropic.Anthropic(api_key=api_key)
        self.con = duckdb.connect(db_path, read_only=True)
        self.planner = Planner(api_key=api_key)
        self.state = ConversationState()

        self.agents: Dict[str, Any] = {
            name: cls(self.client, self.con)
            for name, cls in _AGENT_MAP.items()
        }

    # ── Step execution ───────────────────────────────────────────────────────

    def execute_steps(
        self,
        steps: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        prev_rows: List[Dict[str, Any]] = []

        for step in steps:
            agent_name = step["agent"]
            method_name = step["method"]
            params = dict(step.get("params", {}))

            agent = self.agents.get(agent_name)
            if agent is None:
                results.append({
                    "status": "error",
                    "message": f"Unknown agent: {agent_name}",
                    "result": [],
                })
                continue

            # Visualization and executive_summary receive prior results
            if agent_name == "visualization" and prev_rows:
                params.setdefault("data", prev_rows)
            elif agent_name == "executive_summary":
                params.setdefault("results", results)

            method = getattr(agent, method_name, None)
            if method is None:
                results.append({
                    "status": "error",
                    "message": f"Unknown method: {agent_name}.{method_name}",
                    "result": [],
                })
                continue

            # Agents with run() take a single dict; others take kwargs
            if method_name == "run":
                result = method(params)
            else:
                result = method(**params)

            results.append(result)

            # Track rows for downstream agents
            rows = result.get("result") or []
            if isinstance(rows, list) and rows:
                prev_rows = rows

        return results

    # ── Query handling ───────────────────────────────────────────────────────

    def handle_query(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        history = conversation_history or self.state.get_history()

        steps = self.planner.create_plan(query, history)
        results = self.execute_steps(steps)

        # Collect all rows from successful steps
        all_rows: List[Dict[str, Any]] = []
        for r in results:
            if r.get("status") != "ok":
                continue
            rows = r.get("result") or []
            if isinstance(rows, list):
                all_rows = rows

        all_rows = _sanitize_rows(all_rows)
        finding = generate_response(self.client, query, all_rows)

        self.state.add_turn(query, finding, all_rows)

        return {
            "query": query,
            "steps": steps,
            "rows": all_rows,
            "finding": finding,
        }
