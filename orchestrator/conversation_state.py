from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List, Optional


def _to_float(v: Any) -> Optional[float]:
    """Return *v* as float, or None if not numeric."""
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, Decimal):
        return float(v)
    return None


class ConversationState:
    """Stores conversation history and extracted numeric metrics for follow-up reasoning."""

    def __init__(self) -> None:
        self._history: List[Dict[str, Any]] = []
        self._metrics: Dict[str, float] = {}

    def add_turn(
        self,
        query: str,
        finding: str,
        rows: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self._history.append({
            "query": query,
            "finding": finding,
            "rows": rows or [],
        })

        if rows and len(rows) == 1:
            for key, val in rows[0].items():
                num = _to_float(val)
                if num is not None:
                    self._metrics[key] = num

    def get_last_metric(self, name: str) -> Optional[float]:
        return self._metrics.get(name)

    def get_history(self) -> List[Dict[str, Any]]:
        return list(self._history)
