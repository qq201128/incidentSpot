from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.db.session import get_conn
from app.services.auto_trade_settings_payloads import table_exists
from app.services.factor_candidate_signal_keys import is_factor_candidate_signal_key
from app.services.factor_combo_simulation_keys import is_batch_combo_simulation_strategy


@dataclass(frozen=True)
class RuntimeAvailability:
    conn: Any
    events_available: bool
    failures_available: bool


def attach_slot_and_runtime_state(candidates: list[dict[str, Any]], symbol: str, duration: str) -> None:
    slots = _slot_rows(symbol, duration)
    _attach_slot_state(candidates, slots, symbol, duration)
    _attach_runtime_state(candidates, symbol, duration)


def runtime_statuses_for_slots(slots: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    conn = get_conn()
    try:
        events_available = _runtime_tables_available(conn)
        failures_available = table_exists(conn, "paper_live_prediction_failures")
        availability = RuntimeAvailability(conn, events_available, failures_available)
        statuses = [_runtime_status(slot, availability) for slot in slots]
        return {_status_key(status): status for status in statuses if _has_runtime_state(status)}
    finally:
        conn.close()


def _attach_slot_state(
    candidates: list[dict[str, Any]],
    slots: dict[str, Any],
    symbol: str,
    duration: str,
) -> None:
    for item in candidates:
        slot = slots.get(item.get("strategyKey"))
        item["slot"] = _slot_payload(slot)
        if slot and bool(slot["enabled"]):
            item["gateStatus"] = "enabled"
    candidates.extend(_orphan_slots(candidates, slots, symbol, duration))


def _attach_runtime_state(candidates: list[dict[str, Any]], symbol: str, duration: str) -> None:
    conn = get_conn()
    try:
        events_available = _runtime_tables_available(conn)
        failures_available = table_exists(conn, "paper_live_prediction_failures")
        for item in candidates:
            key = item.get("strategyKey")
            item["latestEvent"] = _latest_event(conn, key, symbol, duration) if events_available and key else None
            item["latestFailure"] = _latest_failure(conn, key, symbol, duration) if failures_available and key else None
    finally:
        conn.close()


def _runtime_status(
    slot: dict[str, Any],
    availability: RuntimeAvailability,
) -> dict[str, Any]:
    strategy_key = str(slot["strategyKey"])
    symbol = str(slot["symbol"]).upper()
    duration = str(slot["duration"])
    latest_event = _latest_event(availability.conn, strategy_key, symbol, duration) if availability.events_available else None
    latest_failure = _latest_failure(
        availability.conn, strategy_key, symbol, duration,
    ) if availability.failures_available else None
    return {
        "strategyKey": strategy_key,
        "candidateType": _candidate_type_from_key(strategy_key),
        "factorName": None,
        "symbol": symbol,
        "duration": duration,
        "source": "auto_trade_strategies",
        "gatePassed": None,
        "gateStatus": "enabled" if bool(slot.get("enabled")) else "not_enabled",
        "rejectionReason": None,
        "metrics": {},
        "slot": _payload_slot(slot),
        "latestEvent": latest_event,
        "latestFailure": latest_failure,
    }


def _slot_rows(symbol: str, duration: str) -> dict[str, Any]:
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT * FROM auto_trade_strategies WHERE symbol = ? AND duration = ?",
            (symbol, duration),
        ).fetchall()
        return {str(row["strategy_key"]): dict(row) for row in rows}
    finally:
        conn.close()


def _latest_event(conn: Any, strategy_key: str, symbol: str, duration: str) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT e.id, e.status, e.result, e.start_time, e.end_time, e.ai_high_winrate_rule,
               o.id AS order_id, o.status AS order_status, o.external_status, o.qty,
               s.id AS settlement_id, s.pnl, s.settled_at
        FROM events e
        LEFT JOIN orders o ON o.event_id = e.id
        LEFT JOIN settlements s ON s.event_id = e.id
        WHERE e.strategy_key = ? AND e.symbol = ? AND e.event_interval = ?
        ORDER BY e.id DESC
        LIMIT 1
        """,
        (strategy_key, symbol, duration),
    ).fetchone()
    return None if row is None else _event_payload(row)


def _latest_failure(conn: Any, strategy_key: str, symbol: str, duration: str) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT stage, reason, created_at
        FROM paper_live_prediction_failures
        WHERE strategy_key = ? AND symbol = ? AND duration = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (strategy_key, symbol, duration),
    ).fetchone()
    return None if row is None else {"stage": row["stage"], "reason": row["reason"], "createdAt": row["created_at"]}


def _event_payload(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "status": row["status"],
        "result": row["result"],
        "startTime": row["start_time"],
        "endTime": row["end_time"],
        "factorName": row["ai_high_winrate_rule"],
        "orderId": row["order_id"],
        "orderStatus": row["order_status"],
        "externalStatus": row["external_status"],
        "qty": row["qty"],
        "settlementId": row["settlement_id"],
        "settlementPnl": row["pnl"],
        "settledAt": row["settled_at"],
    }


def _slot_payload(slot: dict[str, Any] | None) -> dict[str, Any] | None:
    if slot is None:
        return None
    return {"enabled": bool(slot["enabled"]), "qty": float(slot["qty"]), "updatedAt": slot["updated_at"]}


def _payload_slot(slot: dict[str, Any]) -> dict[str, Any]:
    return {
        "enabled": bool(slot.get("enabled")),
        "qty": float(slot.get("qty") or 0),
        "updatedAt": slot.get("updatedAt"),
    }


def _orphan_slots(
    candidates: list[dict[str, Any]],
    slots: dict[str, Any],
    symbol: str,
    duration: str,
) -> list[dict[str, Any]]:
    seen = {item.get("strategyKey") for item in candidates}
    return [_orphan_slot_payload(row, symbol, duration) for key, row in slots.items() if _is_dynamic_key(key) and key not in seen]


def _orphan_slot_payload(row: dict[str, Any], symbol: str, duration: str) -> dict[str, Any]:
    return {
        "strategyKey": row["strategy_key"],
        "candidateType": _candidate_type_from_key(row["strategy_key"]),
        "factorName": None,
        "symbol": symbol,
        "duration": duration,
        "source": "auto_trade_strategies",
        "gatePassed": None,
        "gateStatus": "enabled" if bool(row["enabled"]) else "not_enabled",
        "rejectionReason": "candidate_cache_unavailable",
        "metrics": {},
        "slot": _slot_payload(row),
    }


def _runtime_tables_available(conn: Any) -> bool:
    return all(table_exists(conn, name) for name in ("events", "orders", "settlements"))


def _has_runtime_state(status: dict[str, Any]) -> bool:
    return status["latestEvent"] is not None or status["latestFailure"] is not None


def _status_key(status: dict[str, Any]) -> tuple[str, str, str]:
    return (str(status["strategyKey"]), str(status["symbol"]).upper(), str(status["duration"]))


def _is_dynamic_key(strategy_key: str) -> bool:
    return is_factor_candidate_signal_key(strategy_key) or is_batch_combo_simulation_strategy(strategy_key)


def _candidate_type_from_key(strategy_key: str) -> str:
    return "single_factor" if is_factor_candidate_signal_key(strategy_key) else "factor_combo"
