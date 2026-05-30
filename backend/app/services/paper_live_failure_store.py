from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from app.db.session import get_conn, run_db_write_with_retry
from app.services.paper_live_json_fields import parse_details_json


def ensure_prediction_failure_table(conn: Any) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_live_prediction_failures (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          candidate_key TEXT NOT NULL,
          strategy_key TEXT NOT NULL,
          symbol TEXT NOT NULL,
          duration TEXT NOT NULL,
          stage TEXT NOT NULL,
          reason TEXT NOT NULL,
          details_json TEXT NOT NULL,
          created_at TEXT NOT NULL
        )
        """
    )


def log_prediction_failure(
    *,
    candidate_key: str,
    strategy_key: str,
    symbol: str,
    duration: str,
    stage: str,
    reason: str,
    details: dict[str, Any] | None = None,
) -> None:
    run_db_write_with_retry(
        lambda: _insert_prediction_failure(
            candidate_key=candidate_key,
            strategy_key=strategy_key,
            symbol=symbol,
            duration=duration,
            stage=stage,
            reason=reason,
            details=details or {},
        )
    )


def recent_prediction_failures(conn: Any, symbol: str, duration: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT candidate_key, strategy_key, stage, reason, details_json, created_at
        FROM paper_live_prediction_failures
        WHERE symbol = ? AND duration = ?
        ORDER BY id DESC
        LIMIT 50
        """,
        (symbol.strip().upper(), duration),
    ).fetchall()
    return [_failure_payload(row) for row in rows]


def _insert_prediction_failure(
    *,
    candidate_key: str,
    strategy_key: str,
    symbol: str,
    duration: str,
    stage: str,
    reason: str,
    details: dict[str, Any],
) -> None:
    conn = get_conn()
    try:
        ensure_prediction_failure_table(conn)
        conn.execute(
            """
            INSERT INTO paper_live_prediction_failures(
              candidate_key, strategy_key, symbol, duration, stage, reason, details_json, created_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                candidate_key,
                strategy_key,
                symbol.strip().upper(),
                duration,
                stage,
                reason,
                json.dumps(details, ensure_ascii=False),
                _utc_now(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _failure_payload(row: Any) -> dict[str, Any]:
    data = dict(row)
    details = parse_details_json(data.get("details_json"))
    payload = {
        "candidateKey": data["candidate_key"],
        "strategyKey": data["strategy_key"],
        "stage": data["stage"],
        "reason": data["reason"],
        "details": details.value,
        "createdAt": data["created_at"],
    }
    if details.error:
        payload["detailsParseError"] = details.error
    return payload


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
