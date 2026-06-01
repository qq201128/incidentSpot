from __future__ import annotations

import json
from typing import Any

from app.db.session import get_conn, run_db_write_with_retry

FEATURE_SNAPSHOT_TABLE = "factor_combo_feature_snapshots"


def load_factor_combo_feature_snapshots(symbol: str, duration: str) -> list[dict[str, Any]]:
    conn = get_conn()
    try:
        if not _table_exists(conn):
            return []
        rows = conn.execute(
            f"""
            SELECT entry_open_time, payload
            FROM {FEATURE_SNAPSHOT_TABLE}
            WHERE symbol = ? AND duration = ?
            ORDER BY entry_open_time
            """,
            (symbol.strip().upper(), duration),
        ).fetchall()
    finally:
        conn.close()
    return [_snapshot_from_row(row) for row in rows]


def save_factor_combo_feature_snapshot(
    symbol: str,
    duration: str,
    entry_open_time: int,
    ranking: dict[str, Any],
) -> None:
    snapshot = {"entryOpenTime": int(entry_open_time), "ranking": ranking.get("ranking") or []}
    save_factor_combo_feature_snapshots(symbol, duration, (snapshot,))


def save_factor_combo_feature_snapshots(
    symbol: str,
    duration: str,
    snapshots: tuple[dict[str, Any], ...] | list[dict[str, Any]],
) -> None:
    rows = [_snapshot_values(symbol, duration, snapshot) for snapshot in snapshots]
    if not rows:
        raise ValueError("factor combo feature snapshot batch must not be empty")

    def _persist() -> None:
        conn = get_conn()
        try:
            _ensure_table(conn)
            conn.executemany(
                f"""
                INSERT INTO {FEATURE_SNAPSHOT_TABLE}(symbol, duration, entry_open_time, payload)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(symbol, duration, entry_open_time) DO UPDATE SET payload = excluded.payload
                """,
                rows,
            )
            conn.commit()
        finally:
            conn.close()

    run_db_write_with_retry(_persist)


def _snapshot_values(symbol: str, duration: str, snapshot: dict[str, Any]) -> tuple[str, str, int, str]:
    entry_open_time = int(snapshot["entryOpenTime"])
    payload = {"entryOpenTime": entry_open_time, "ranking": snapshot.get("ranking") or []}
    if snapshot.get("previousTopFactorName"):
        payload["previousTopFactorName"] = str(snapshot["previousTopFactorName"])
    return (
        symbol.strip().upper(),
        duration,
        entry_open_time,
        json.dumps(payload, ensure_ascii=False),
    )


def _snapshot_from_row(row: Any) -> dict[str, Any]:
    payload = json.loads(row["payload"])
    if not isinstance(payload, dict):
        raise ValueError("factor combo feature snapshot payload must be an object")
    payload.setdefault("entryOpenTime", int(row["entry_open_time"]))
    return payload


def _table_exists(conn: Any) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (FEATURE_SNAPSHOT_TABLE,),
    ).fetchone()
    return row is not None


def _ensure_table(conn: Any) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {FEATURE_SNAPSHOT_TABLE} (
          symbol TEXT NOT NULL,
          duration TEXT NOT NULL,
          entry_open_time INTEGER NOT NULL,
          payload TEXT NOT NULL,
          PRIMARY KEY(symbol, duration, entry_open_time)
        )
        """
    )
