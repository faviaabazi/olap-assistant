from __future__ import annotations

from typing import Any, Dict, List, Optional

from orchestrator.conversation_state import ConversationState
from orchestrator.planner import Planner


class Orchestrator:
    """
    Thin wrapper around Planner.
    All routing, execution, and response-building is handled by Planner.run().
    Orchestrator exists only to hold shared state and expose handle_query().
    """

    def __init__(self, api_key: str, db_path: str = "olap.duckdb") -> None:
        self.planner = Planner(db_path=db_path)
        self.state = ConversationState()

    def handle_query(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Route *query* through Planner.run() and return its structured response.

        The response always contains:
            status, response_mode, finding, follow_up_questions, _routing
        Plus mode-specific fields (result_rows, rows, sections, report, etc.)
        """
        # Sync external history into planner if provided
        if conversation_history:
            self.planner.conversation_history = list(conversation_history)

        result = self.planner.run(query)

        # Keep ConversationState in sync for callers that use it
        rows = result.get("result_rows") or result.get("rows") or []
        finding = result.get("finding", "")
        self.state.add_turn(query, finding, rows)

        return result

    def reset(self) -> None:
        """Clear conversation history in both planner and state."""
        self.planner.reset()
        self.state.clear()
