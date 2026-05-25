from __future__ import annotations

from typing import Any

from app.db.session import get_conn
from app.services.factor_learning_common import round_metric, utc_now
from app.services.model_family_config import MODEL_FAMILIES, model_family_strategy_key


def model_shadow_simulation_report(symbol: str, duration: str) -> dict[str, Any]:
    sym = symbol.strip().upper()
    keys = _model_shadow_keys(duration)
    conn = get_conn()
    try:
        predictions = _prediction_rows(conn, sym, duration, keys)
        events = _event_rows(conn, sym, duration, keys)
    finally:
        conn.close()
    families = _family_rows(keys, predictions, events)
    summary = _summary(families)
    return {
        "source": "model_shadow",
        "symbol": sym,
        "duration": duration,
        "updatedAt": utc_now(),
        "summary": summary,
        "families": families,
        "watchlist": [row for row in families if row["status"] != "simulation_active"],
    }


def _model_shadow_keys(duration: str) -> list[str]:
    return [model_family_strategy_key(family, duration) for family in MODEL_FAMILIES]


def _prediction_rows(conn: Any, symbol: str, duration: str, keys: list[str]) -> dict[str, dict[str, Any]]:
    placeholders = ",".join("?" for _key in keys)
    rows = conn.execute(
        f"""
        SELECT strategy_key, COUNT(*) AS total,
               SUM(CASE WHEN trade_quality_passed = 1 THEN 1 ELSE 0 END) AS passed,
               SUM(CASE WHEN settled_at IS NOT NULL THEN 1 ELSE 0 END) AS settled,
               SUM(CASE WHEN prediction_correct = 1 THEN 1 ELSE 0 END) AS wins,
               MAX(created_at) AS latest_prediction_at
        FROM predictions
        WHERE strategy_key IN ({placeholders}) AND symbol = ? AND duration = ?
        GROUP BY strategy_key
        """,
        (*keys, symbol, duration),
    ).fetchall()
    return {str(row["strategy_key"]): dict(row) for row in rows}


def _event_rows(conn: Any, symbol: str, duration: str, keys: list[str]) -> dict[str, dict[str, Any]]:
    placeholders = ",".join("?" for _key in keys)
    rows = conn.execute(
        f"""
        SELECT strategy_key, COUNT(*) AS total,
               SUM(CASE WHEN status = 'OPEN' THEN 1 ELSE 0 END) AS open_count,
               MAX(id) AS latest_event_id
        FROM events
        WHERE strategy_key IN ({placeholders}) AND symbol = ? AND event_interval = ?
        GROUP BY strategy_key
        """,
        (*keys, symbol, duration),
    ).fetchall()
    return {str(row["strategy_key"]): dict(row) for row in rows}


def _family_rows(keys: list[str], predictions: dict[str, dict[str, Any]], events: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [_family_row(key, predictions.get(key) or {}, events.get(key) or {}) for key in keys]


def _family_row(key: str, prediction: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    total = int(prediction.get("total") or 0)
    passed = int(prediction.get("passed") or 0)
    event_count = int(event.get("total") or 0)
    return {
        "strategyKey": key,
        "status": _status(total, passed, event_count),
        "predictionCount": total,
        "qualityPassedCount": passed,
        "settledPredictionCount": int(prediction.get("settled") or 0),
        "predictionWinRate": _win_rate(prediction),
        "simulationEventCount": event_count,
        "openEventCount": int(event.get("open_count") or 0),
        "latestPredictionAt": prediction.get("latest_prediction_at"),
        "latestEventId": event.get("latest_event_id"),
    }


def _status(total: int, passed: int, event_count: int) -> str:
    if total <= 0:
        return "waiting_model_prediction"
    if passed <= 0:
        return "quality_gate_blocked"
    if event_count <= 0:
        return "passed_prediction_without_simulation_event"
    return "simulation_active"


def _win_rate(prediction: dict[str, Any]) -> float | None:
    settled = int(prediction.get("settled") or 0)
    if settled <= 0:
        return None
    return round_metric(int(prediction.get("wins") or 0) / settled, 4)


def _summary(families: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "familyCount": len(families),
        "predictionCount": sum(int(row["predictionCount"]) for row in families),
        "qualityPassedCount": sum(int(row["qualityPassedCount"]) for row in families),
        "simulationEventCount": sum(int(row["simulationEventCount"]) for row in families),
        "activeFamilyCount": sum(1 for row in families if row["status"] == "simulation_active"),
        "blockedFamilyCount": sum(1 for row in families if row["status"] == "quality_gate_blocked"),
    }
