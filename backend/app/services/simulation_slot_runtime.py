from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.db.session import get_conn
from app.services.auto_trade_settings_payloads import table_exists
from app.services.factor_candidate_signal_keys import is_factor_candidate_signal_key
from app.services.factor_combo_simulation_keys import is_batch_combo_simulation_strategy

SlotKey = tuple[str, str, str]


@dataclass(frozen=True)
class RuntimeLookups:
    events: dict[SlotKey, dict[str, Any]]
    failures: dict[SlotKey, dict[str, Any]]


def attach_slot_and_runtime_state(candidates: list[dict[str, Any]], symbol: str, duration: str) -> None:
    slots = _slot_rows(symbol, duration)
    _attach_slot_state(candidates, slots, symbol, duration)
    _attach_runtime_state(candidates, symbol, duration)


def runtime_statuses_for_slots(slots: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    slot_keys = _slot_keys(slots)
    conn = get_conn()
    try:
        lookups = _runtime_lookups(conn, slot_keys)
        statuses = [_runtime_status(slot, lookups) for slot in slots]
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
    slot_keys = _candidate_slot_keys(candidates, symbol, duration)
    conn = get_conn()
    try:
        lookups = _runtime_lookups(conn, slot_keys)
        for item in candidates:
            key = item.get("strategyKey")
            slot_key = (str(key), symbol, duration) if key else None
            item["latestEvent"] = lookups.events.get(slot_key) if slot_key else None
            item["latestFailure"] = lookups.failures.get(slot_key) if slot_key else None
    finally:
        conn.close()


def _runtime_status(
    slot: dict[str, Any],
    lookups: RuntimeLookups,
) -> dict[str, Any]:
    strategy_key = str(slot["strategyKey"])
    symbol = str(slot["symbol"]).upper()
    duration = str(slot["duration"])
    slot_key = (strategy_key, symbol, duration)
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
        "latestEvent": lookups.events.get(slot_key),
        "latestFailure": lookups.failures.get(slot_key),
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


def _runtime_lookups(conn: Any, slot_keys: tuple[SlotKey, ...]) -> RuntimeLookups:
    if not slot_keys:
        return RuntimeLookups(events={}, failures={})
    events = _latest_events(conn, slot_keys) if _runtime_tables_available(conn) else {}
    failures = _latest_failures(conn, slot_keys) if table_exists(conn, "paper_live_prediction_failures") else {}
    return RuntimeLookups(events=events, failures=failures)


def _latest_events(conn: Any, slot_keys: tuple[SlotKey, ...]) -> dict[SlotKey, dict[str, Any]]:
    rows = conn.execute(
        """
        WITH requested(strategy_key, symbol, duration) AS (
          VALUES {values}
        ),
        ranked AS (
          SELECT r.strategy_key AS request_strategy_key,
                 r.symbol AS request_symbol,
                 r.duration AS request_duration,
                 e.id, e.status, e.result, e.start_time, e.end_time, e.ai_high_winrate_rule,
                 o.id AS order_id, o.status AS order_status, o.external_status, o.qty,
                 s.id AS settlement_id, s.pnl, s.settled_at,
                 ROW_NUMBER() OVER (
                   PARTITION BY r.strategy_key, r.symbol, r.duration
                   ORDER BY e.id DESC
                 ) AS row_rank
          FROM requested r
          JOIN events e
            ON e.strategy_key = r.strategy_key
           AND e.symbol = r.symbol
           AND e.event_interval = r.duration
          LEFT JOIN orders o ON o.event_id = e.id
          LEFT JOIN settlements s ON s.event_id = e.id
        )
        SELECT *
        FROM ranked
        WHERE row_rank = 1
        """.format(values=_value_placeholders(slot_keys)),
        _flatten_slot_keys(slot_keys),
    ).fetchall()
    return {_runtime_row_key(row): _event_payload(row) for row in rows}


def _latest_failures(conn: Any, slot_keys: tuple[SlotKey, ...]) -> dict[SlotKey, dict[str, Any]]:
    rows = conn.execute(
        """
        WITH requested(strategy_key, symbol, duration) AS (
          VALUES {values}
        ),
        ranked AS (
          SELECT r.strategy_key AS request_strategy_key,
                 r.symbol AS request_symbol,
                 r.duration AS request_duration,
                 f.stage, f.reason, f.created_at,
                 ROW_NUMBER() OVER (
                   PARTITION BY r.strategy_key, r.symbol, r.duration
                   ORDER BY f.id DESC
                 ) AS row_rank
          FROM requested r
          JOIN paper_live_prediction_failures f
            ON f.strategy_key = r.strategy_key
           AND f.symbol = r.symbol
           AND f.duration = r.duration
        )
        SELECT *
        FROM ranked
        WHERE row_rank = 1
        """.format(values=_value_placeholders(slot_keys)),
        _flatten_slot_keys(slot_keys),
    ).fetchall()
    return {
        _runtime_row_key(row): {
            "stage": row["stage"],
            "reason": row["reason"],
            "createdAt": row["created_at"],
        }
        for row in rows
    }


def _value_placeholders(slot_keys: tuple[SlotKey, ...]) -> str:
    return ", ".join("(?, ?, ?)" for _slot_key in slot_keys)


def _flatten_slot_keys(slot_keys: tuple[SlotKey, ...]) -> tuple[str, ...]:
    return tuple(value for slot_key in slot_keys for value in slot_key)


def _slot_keys(slots: list[dict[str, Any]]) -> tuple[SlotKey, ...]:
    return tuple(sorted(
        (str(slot["strategyKey"]), str(slot["symbol"]).upper(), str(slot["duration"]))
        for slot in slots
    ))


def _candidate_slot_keys(candidates: list[dict[str, Any]], symbol: str, duration: str) -> tuple[SlotKey, ...]:
    return tuple(sorted(
        (str(item["strategyKey"]), symbol, duration)
        for item in candidates
        if item.get("strategyKey")
    ))


def _runtime_row_key(row: Any) -> SlotKey:
    return (
        str(row["request_strategy_key"]),
        str(row["request_symbol"]).upper(),
        str(row["request_duration"]),
    )


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
