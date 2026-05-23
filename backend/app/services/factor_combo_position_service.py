from __future__ import annotations

from typing import Any

from app.api.event_response import event_response
from app.services.factor_combo_simulation_keys import factor_combo_event_strategy_filter
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
    normalized_factor_name = _normalized_factor_name(factor_name)
    rows = _event_rows(
        conn,
        symbol=symbol.upper(),
        duration=duration,
        factor_name=normalized_factor_name,
        limit=limit,
    )
    events = [event_response(conn, row) for row in rows]
    return {
        "strategyKey": FACTOR_COMBO_STRATEGY_KEY,
        "symbol": symbol.upper(),
        "duration": duration,
        "factorName": normalized_factor_name,
        "total": len(events),
        "openCount": sum(1 for item in events if item["status"] == "OPEN"),
        "settledCount": sum(1 for item in events if item["status"] == "SETTLED"),
        "currentFactorCount": _current_factor_count(events, normalized_factor_name),
        "totalPnl": round(sum(float(item.get("totalPnl") or 0.0) for item in events), 6),
        "events": events,
    }


def _event_rows(
    conn: Any,
    *,
    symbol: str,
    duration: str,
    factor_name: str | None,
    limit: int,
) -> list[Any]:
    strategy_clause, strategy_params = factor_combo_event_strategy_filter()
    factor_clause = " AND ai_high_winrate_rule = ?" if factor_name else ""
    factor_params = (factor_name,) if factor_name else ()
    params = (*strategy_params, symbol, duration, *factor_params, int(limit))
    return conn.execute(
        f"""
        SELECT *
        FROM events
        WHERE {strategy_clause}
          AND symbol = ?
          AND event_interval = ?
          {factor_clause}
        ORDER BY id DESC
        LIMIT ?
        """,
        params,
    ).fetchall()


def _current_factor_count(events: list[dict], factor_name: str | None) -> int:
    if not factor_name:
        return 0
    return sum(1 for item in events if item.get("aiHighWinrateRule") == factor_name)


def _normalized_factor_name(factor_name: str | None) -> str | None:
    if factor_name is None:
        return None
    normalized = factor_name.strip()
    return normalized or None
