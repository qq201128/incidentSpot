from __future__ import annotations

from typing import Any

from app.api.event_response import event_response
from app.services.rule_config import SUPPORTED_RULE_DURATIONS
from app.services.strategy_registry import FACTOR_COMBO_STRATEGY_KEY

DEFAULT_POSITION_LIMIT = 80


def factor_combo_positions_payload(
    conn: Any,
    *,
    symbol: str,
    duration: str,
    factor_name: str | None,
    limit: int = DEFAULT_POSITION_LIMIT,
) -> dict[str, Any]:
    if duration not in SUPPORTED_RULE_DURATIONS:
        raise ValueError(f"unsupported duration: {duration}")
    rows = _event_rows(conn, symbol.upper(), duration, limit)
    events = [event_response(conn, row) for row in rows]
    return {
        "strategyKey": FACTOR_COMBO_STRATEGY_KEY,
        "symbol": symbol.upper(),
        "duration": duration,
        "factorName": factor_name,
        "total": len(events),
        "openCount": sum(1 for item in events if item["status"] == "OPEN"),
        "settledCount": sum(1 for item in events if item["status"] == "SETTLED"),
        "currentFactorCount": _current_factor_count(events, factor_name),
        "totalPnl": round(sum(float(item.get("totalPnl") or 0.0) for item in events), 6),
        "events": events,
    }


def _event_rows(conn: Any, symbol: str, duration: str, limit: int) -> list[Any]:
    return conn.execute(
        """
        SELECT *
        FROM events
        WHERE strategy_key = ?
          AND symbol = ?
          AND event_interval = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (FACTOR_COMBO_STRATEGY_KEY, symbol, duration, int(limit)),
    ).fetchall()


def _current_factor_count(events: list[dict], factor_name: str | None) -> int:
    if not factor_name:
        return 0
    return sum(1 for item in events if item.get("aiHighWinrateRule") == factor_name)
