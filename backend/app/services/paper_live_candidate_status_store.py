from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any


def ensure_candidate_status_tables(conn: Any) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_live_candidate_status (
          candidate_key TEXT NOT NULL,
          symbol TEXT NOT NULL,
          duration TEXT NOT NULL,
          status TEXT NOT NULL,
          reason TEXT NOT NULL,
          details_json TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          PRIMARY KEY(candidate_key, symbol, duration)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_live_candidate_status_history (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          candidate_key TEXT NOT NULL,
          symbol TEXT NOT NULL,
          duration TEXT NOT NULL,
          old_status TEXT,
          new_status TEXT NOT NULL,
          reason TEXT NOT NULL,
          details_json TEXT NOT NULL,
          changed_at TEXT NOT NULL
        )
        """
    )


def write_candidate_status(conn: Any, symbol: str, duration: str, candidate: dict[str, Any]) -> None:
    ensure_candidate_status_tables(conn)
    previous = _current_status(conn, symbol, duration, candidate["candidateKey"])
    if previous is None or previous["status"] != candidate["status"]:
        _write_history(conn, symbol, duration, candidate, previous)
    conn.execute(
        """
        INSERT INTO paper_live_candidate_status(
          candidate_key, symbol, duration, status, reason, details_json, updated_at
        )
        VALUES(?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(candidate_key, symbol, duration) DO UPDATE SET
          status = excluded.status, reason = excluded.reason,
          details_json = excluded.details_json, updated_at = excluded.updated_at
        """,
        _status_values(symbol, duration, candidate),
    )


def recent_status_changes(conn: Any, symbol: str, duration: str, *, limit: int = 50) -> list[dict[str, Any]]:
    ensure_candidate_status_tables(conn)
    rows = conn.execute(
        """
        SELECT candidate_key, symbol, duration, old_status, new_status,
               reason, details_json, changed_at
        FROM paper_live_candidate_status_history
        WHERE symbol = ? AND duration = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (symbol.strip().upper(), duration, int(limit)),
    ).fetchall()
    return [_history_payload(dict(row)) for row in rows]


def _current_status(conn: Any, symbol: str, duration: str, candidate_key: str) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT status, reason
        FROM paper_live_candidate_status
        WHERE candidate_key = ? AND symbol = ? AND duration = ?
        """,
        (candidate_key, symbol.strip().upper(), duration),
    ).fetchone()
    return None if row is None else dict(row)


def _write_history(conn: Any, symbol: str, duration: str, candidate: dict[str, Any], previous: dict[str, Any] | None) -> None:
    conn.execute(
        """
        INSERT INTO paper_live_candidate_status_history(
          candidate_key, symbol, duration, old_status, new_status, reason, details_json, changed_at
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            candidate["candidateKey"],
            symbol.strip().upper(),
            duration,
            None if previous is None else previous["status"],
            candidate["status"],
            candidate["reason"],
            json.dumps(candidate, ensure_ascii=False),
            _utc_now(),
        ),
    )


def _status_values(symbol: str, duration: str, candidate: dict[str, Any]) -> tuple[Any, ...]:
    return (
        candidate["candidateKey"],
        symbol.strip().upper(),
        duration,
        candidate["status"],
        candidate["reason"],
        json.dumps(candidate, ensure_ascii=False),
        _utc_now(),
    )


def _history_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidateKey": row["candidate_key"],
        "symbol": row["symbol"],
        "duration": row["duration"],
        "oldStatus": row["old_status"],
        "newStatus": row["new_status"],
        "reason": row["reason"],
        "details": json.loads(row["details_json"]),
        "changedAt": row["changed_at"],
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
