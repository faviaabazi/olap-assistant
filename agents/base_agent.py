"""
BaseAgent — shared interface and utilities for all OLAP agents.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import duckdb


class BaseAgent(ABC):
    """
    Abstract base for every OLAP agent.

    Parameters
    ----------
    client:
        An instantiated ``anthropic.Anthropic`` client.
    con:
        An open ``duckdb.DuckDBPyConnection`` pointed at olap.duckdb.
    """

    def __init__(self, client: Any, con: duckdb.DuckDBPyConnection) -> None:
        self.client = client
        self.con = con

    # ── public interface ───────────────────────────────────────────────────────

    @abstractmethod
    def run(self, query: Any) -> dict:
        """
        Process *query* and return a result dict.

        Every concrete agent must implement this method.  The returned dict
        must follow the standard schema:
            {
                "status":  "ok" | "error",
                "result":  list[dict],   # row dicts — always a list
                "message": str,          # concise factual summary
            }
        """

    # ── helpers ────────────────────────────────────────────────────────────────

    def execute_sql(self, sql: str) -> list[dict]:
        """
        Run *sql* against the connected DuckDB database and return each row
        as a plain ``dict`` keyed by column name.

        Parameters
        ----------
        sql:
            A single SELECT statement (or any DQL/DML that returns rows).

        Returns
        -------
        list[dict]
            One dict per row.  Empty list if the query produces no rows.
        """
        rel = self.con.execute(sql)
        columns = [desc[0] for desc in rel.description]
        return [dict(zip(columns, row)) for row in rel.fetchall()]
