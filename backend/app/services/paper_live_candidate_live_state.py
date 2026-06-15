from __future__ import annotations

import sqlite3
from typing import Any

from app.db.session import get_conn
from app.services.kline_timing import rule_interval_ms_for_duration

LIVE_OVERVIEW_LIMIT = 100


def live_state_by_strategy(conn: Any, symbol: str, duration: str) -> dict[str, dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT strategy_key, enabled, live_trading_enabled, qty, updated_at
        FROM auto_trade_strategies
        WHERE symbol = ? AND duration = ?
        """,
        (symbol.strip().upper(), duration),
    ).fetchall()
    return {str(row["strategy_key"]): _state_payload(row) for row in rows}


def live_trading_overview() -> dict[str, Any]:
    conn = get_conn()
    try:
        rows = conn.execute(
            """
            SELECT strategy_key, symbol, duration, enabled, live_trading_enabled, qty, updated_at
            FROM auto_trade_strategies
            WHERE live_trading_enabled = 1
            ORDER BY symbol, duration, updated_at DESC
            LIMIT ?
            """,
            (LIVE_OVERVIEW_LIMIT,),
        ).fetchall()
        runtime = _runtime_status_by_slot()
        items = [_overview_item(conn, dict(row), runtime) for row in rows]
    finally:
        conn.close()
    return {
        "version": "paper_live_live_trading_overview_v1",
        "activeCount": len(items),
        "limit": LIVE_OVERVIEW_LIMIT,
        "groups": _overview_groups(items),
        "items": items,
    }


def _state_payload(row: Any) -> dict[str, Any]:
    return {
        "autoTradeEnabled": bool(row["enabled"]),
        "liveTradingEnabled": bool(row["live_trading_enabled"]),
        "qty": float(row["qty"]),
        "updatedAt": row["updated_at"],
    }


def _overview_item(conn: Any, row: dict[str, Any], runtime: dict[tuple[str, str, str], dict[str, Any]]) -> dict[str, Any]:
    candidate = _latest_candidate_for_strategy(conn, row)
    last_settled = _latest_settled_for_strategy(conn, row)
    last_event = _latest_settled_event_for_strategy(conn, row)
    runtime_status = runtime.get(_slot_key(row), {})
    latest_prediction = runtime_status.get("latestPrediction") or {}
    return {
        "symbol": row["symbol"],
        "duration": row["duration"],
        "strategyKey": row["strategy_key"],
        "candidateKey": _candidate_key(candidate, row["strategy_key"]),
        "candidateName": _candidate_name(candidate, row["strategy_key"]),
        "candidateType": _candidate_type(candidate),
        "modelFamily": candidate.get("model_family"),
        "modelVersion": candidate.get("model_version"),
        "factorName": candidate.get("high_winrate_rule"),
        "qty": float(row["qty"]),
        "autoTradeEnabled": bool(row["enabled"]),
        "liveTradingEnabled": bool(row["live_trading_enabled"]),
        "runtimeReason": runtime_status.get("reason") or "unknown",
        "latestPredictionAt": latest_prediction.get("createdAt"),
        "latestPredictionFresh": latest_prediction.get("fresh"),
        "latestPredictionAgeMs": latest_prediction.get("ageMs"),
        "lastSettledOpenTime": last_event.get("start_time") or last_settled.get("open_time"),
        "lastSettledExitOpenTime": _settled_exit_open_time(last_settled, row["duration"]),
        "lastSettledEventEndTime": last_event.get("end_time"),
        "lastSettledAt": last_settled.get("settled_at"),
        "lastSettledPredictionCorrect": _settled_prediction_correct(last_settled, last_event),
        "lastSettledEntryPrice": last_event.get("strike_value") or last_settled.get("entry_price"),
        "lastSettledExitPrice": last_event.get("settlement_price") or last_settled.get("exit_price"),
        "lastSettledPredictionOpenTime": last_settled.get("open_time"),
        "updatedAt": row["updated_at"],
    }


def _latest_candidate_for_strategy(conn: Any, row: dict[str, Any]) -> dict[str, Any]:
    found = conn.execute(
        """
        SELECT signal_key, high_winrate_rule, model_family, model_version
        FROM predictions
        WHERE symbol = ? AND duration = ? AND strategy_key = ?
        ORDER BY open_time DESC
        LIMIT 1
        """,
        (row["symbol"], row["duration"], row["strategy_key"]),
    ).fetchone()
    return dict(found) if found is not None else {}


def _latest_settled_for_strategy(conn: Any, row: dict[str, Any]) -> dict[str, Any]:
    found = conn.execute(
        """
        SELECT open_time, entry_price, exit_price, prediction_correct, settled_at
        FROM predictions
        WHERE symbol = ? AND duration = ? AND strategy_key = ? AND settled_at IS NOT NULL
        ORDER BY open_time DESC
        LIMIT 1
        """,
        (row["symbol"], row["duration"], row["strategy_key"]),
    ).fetchone()
    return dict(found) if found is not None else {}


def _latest_settled_event_for_strategy(conn: Any, row: dict[str, Any]) -> dict[str, Any]:
    try:
        found = conn.execute(
            """
            SELECT id, start_time, end_time, strike_value, settlement_price,
                   ai_prediction_correct, prediction_open_time
            FROM events
            WHERE symbol = ? AND event_interval = ? AND strategy_key = ? AND status = 'SETTLED'
            ORDER BY start_time DESC, id DESC
            LIMIT 1
            """,
            (row["symbol"], row["duration"], row["strategy_key"]),
        ).fetchone()
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc).lower():
            return {}
        raise
    return dict(found) if found is not None else {}


def _settled_prediction_correct(row: dict[str, Any], event: dict[str, Any]) -> bool | None:
    value = event.get("ai_prediction_correct")
    if value is None:
        value = row.get("prediction_correct")
    if value is None:
        return None
    return bool(value)


def _settled_exit_open_time(row: dict[str, Any], duration: str) -> int | None:
    open_time = row.get("open_time")
    if open_time is None:
        return None
    return int(open_time) + rule_interval_ms_for_duration(duration)


def _candidate_key(candidate: dict[str, Any], fallback: str) -> str:
    if candidate.get("model_family") and candidate.get("model_version"):
        return str(candidate["model_version"])
    return str(candidate.get("signal_key") or candidate.get("high_winrate_rule") or fallback)


def _candidate_name(candidate: dict[str, Any], fallback: str) -> str:
    if candidate.get("model_family") and candidate.get("model_version"):
        return f"{candidate['model_family']} · {candidate['model_version']}"
    return str(candidate.get("high_winrate_rule") or candidate.get("signal_key") or fallback)


def _candidate_type(candidate: dict[str, Any]) -> str:
    family = candidate.get("model_family")
    if family and family not in {"factor", "factor_combo"}:
        return "model"
    return "factor_combo" if family == "factor_combo" else "factor"


def _overview_groups(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in items:
        grouped.setdefault((str(item["symbol"]), str(item["duration"])), []).append(item)
    return [
        {
            "symbol": symbol,
            "duration": duration,
            "activeCount": len(candidates),
            "candidates": candidates,
        }
        for (symbol, duration), candidates in grouped.items()
    ]


def _runtime_status_by_slot() -> dict[tuple[str, str, str], dict[str, Any]]:
    from app.services.auto_trade_status import get_auto_trade_status

    status = get_auto_trade_status()
    rows = status.get("strategies") or []
    return {
        (
            str(item.get("settings", {}).get("strategyKey")),
            str(item.get("settings", {}).get("symbol")).upper(),
            str(item.get("settings", {}).get("duration")),
        ): item
        for item in rows
        if isinstance(item, dict) and isinstance(item.get("settings"), dict)
    }


def _slot_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row["strategy_key"]), str(row["symbol"]).upper(), str(row["duration"]))
