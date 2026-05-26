from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

STAGE_FEATURE_CONSTRUCTION = "feature_construction"
STAGE_PREDICTION_GENERATION = "prediction_generation"
STAGE_LABEL_CONSTRUCTION = "label_construction"
STAGE_SETTLEMENT_UPDATE = "settlement_update"
STATUS_PASSED = "passed"
STATUS_FAILED = "failed"
STATUS_PENDING = "pending"
REASON_INVALID_DATA_LEAKAGE = "invalid_data_leakage"


def ensure_stage_log_table(conn: Any) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_live_prediction_stage_log (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          signal_key TEXT NOT NULL,
          strategy_key TEXT NOT NULL,
          symbol TEXT NOT NULL,
          duration TEXT NOT NULL,
          open_time INTEGER,
          stage TEXT NOT NULL,
          status TEXT NOT NULL,
          reason TEXT,
          details_json TEXT NOT NULL,
          created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_paper_live_stage_log_lookup
        ON paper_live_prediction_stage_log(symbol, duration, signal_key, open_time, stage)
        """
    )


def log_prediction_stage(
    conn: Any,
    *,
    signal_key: str,
    strategy_key: str,
    symbol: str,
    duration: str,
    open_time: int | None,
    stage: str,
    status: str,
    reason: str | None,
    details: dict[str, Any],
) -> None:
    ensure_stage_log_table(conn)
    conn.execute(
        """
        INSERT INTO paper_live_prediction_stage_log(
          signal_key, strategy_key, symbol, duration, open_time, stage,
          status, reason, details_json, created_at
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            signal_key,
            strategy_key,
            symbol.strip().upper(),
            duration,
            open_time,
            stage,
            status,
            reason,
            json.dumps(details, ensure_ascii=False, sort_keys=True),
            _utc_now(),
        ),
    )


def log_prediction_generation_stages(conn: Any, result: dict[str, Any]) -> None:
    context = _prediction_context(result)
    feature_status, feature_reason = _feature_stage_decision(result)
    log_prediction_stage(
        conn,
        **context,
        stage=STAGE_FEATURE_CONSTRUCTION,
        status=feature_status,
        reason=feature_reason,
        details=_prediction_details(result),
    )
    prediction_status = _prediction_stage_status(result)
    log_prediction_stage(
        conn,
        **context,
        stage=STAGE_PREDICTION_GENERATION,
        status=prediction_status[0],
        reason=prediction_status[1],
        details=_prediction_details(result),
    )


def log_settlement_success(
    conn: Any,
    row: dict[str, Any],
    *,
    entry_price: float,
    exit_open_time: int,
    exit_price: float,
    actual_return: float,
    prediction_correct: bool,
) -> None:
    context = _row_context(row)
    details = _settlement_details(entry_price, exit_open_time, exit_price, actual_return)
    log_prediction_stage(
        conn,
        **context,
        stage=STAGE_LABEL_CONSTRUCTION,
        status=STATUS_PASSED,
        reason="target_period_closed",
        details={**details, "predictionCorrect": bool(prediction_correct)},
    )
    log_prediction_stage(
        conn,
        **context,
        stage=STAGE_SETTLEMENT_UPDATE,
        status=STATUS_PASSED,
        reason="settled_with_real_kline",
        details={**details, "predictionCorrect": bool(prediction_correct)},
    )


def log_settlement_pending(
    conn: Any,
    row: dict[str, Any],
    *,
    entry_price: float | None,
    exit_open_time: int,
    exit_price: float | None,
) -> None:
    reason = _missing_price_reason(entry_price, exit_price)
    log_prediction_stage(
        conn,
        **_row_context(row),
        stage=STAGE_SETTLEMENT_UPDATE,
        status=STATUS_PENDING,
        reason=reason,
        details={
            "entryPrice": entry_price,
            "exitOpenTime": exit_open_time,
            "settlementPrice": exit_price,
        },
    )


def recent_stage_logs(conn: Any, symbol: str, duration: str, *, limit: int = 50) -> list[dict[str, Any]]:
    ensure_stage_log_table(conn)
    rows = conn.execute(
        """
        SELECT signal_key, strategy_key, symbol, duration, open_time, stage,
               status, reason, details_json, created_at
        FROM paper_live_prediction_stage_log
        WHERE symbol = ? AND duration = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (symbol.strip().upper(), duration, int(limit)),
    ).fetchall()
    return [_stage_log_payload(dict(row)) for row in rows]


def _prediction_context(result: dict[str, Any]) -> dict[str, Any]:
    strategy_key = str(result.get("strategy_key") or result.get("signal_key"))
    return {
        "signal_key": str(result.get("signal_key") or strategy_key),
        "strategy_key": strategy_key,
        "symbol": str(result["symbol"]),
        "duration": str(result["duration"]),
        "open_time": int(result["open_time"]),
    }


def _row_context(row: dict[str, Any]) -> dict[str, Any]:
    signal_key = row.get("signal_key") or row.get("strategy_key")
    return {
        "signal_key": str(signal_key),
        "strategy_key": str(row.get("strategy_key") or signal_key),
        "symbol": str(row["symbol"]),
        "duration": str(row["duration"]),
        "open_time": int(row["open_time"]),
    }


def _feature_stage_decision(result: dict[str, Any]) -> tuple[str, str]:
    freshness = result.get("data_freshness_status")
    missing = result.get("missing_feature_status")
    if freshness == REASON_INVALID_DATA_LEAKAGE:
        return STATUS_FAILED, REASON_INVALID_DATA_LEAKAGE
    if not missing:
        return STATUS_FAILED, "missing_feature_status_not_recorded"
    if missing != "complete":
        return STATUS_FAILED, str(missing)
    if not freshness:
        return STATUS_FAILED, "data_freshness_status_not_recorded"
    return STATUS_PASSED, "features_available_before_entry"


def _prediction_stage_status(result: dict[str, Any]) -> tuple[str, str]:
    if result.get("data_freshness_status") == REASON_INVALID_DATA_LEAKAGE:
        return STATUS_FAILED, REASON_INVALID_DATA_LEAKAGE
    return STATUS_PASSED, "prediction_written_at_entry_window"


def _prediction_details(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "entryOpenTime": result.get("open_time"),
        "modelVersion": result.get("model_version"),
        "modelFamily": result.get("model_family"),
        "dataFreshnessStatus": result.get("data_freshness_status"),
        "missingFeatureStatus": result.get("missing_feature_status"),
        "sourceOpenTime": _source_open_time(result),
        "futureDataLeakageReason": result.get("future_data_leakage_reason"),
    }


def _settlement_details(
    entry_price: float,
    exit_open_time: int,
    exit_price: float,
    actual_return: float,
) -> dict[str, Any]:
    return {
        "entryPrice": entry_price,
        "exitOpenTime": exit_open_time,
        "settlementPrice": exit_price,
        "actualReturn": actual_return,
    }


def _missing_price_reason(entry_price: float | None, exit_price: float | None) -> str:
    if entry_price is None and exit_price is None:
        return "entry_and_settlement_price_missing"
    if entry_price is None:
        return "entry_price_missing"
    return "settlement_price_missing"


def _source_open_time(result: dict[str, Any]) -> int | None:
    for key in ("sourceOpenTime", "source_open_time", "featureOpenTime", "feature_open_time"):
        value = result.get(key)
        if value is not None:
            return int(value)
    return None


def _stage_log_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "signalKey": row["signal_key"],
        "strategyKey": row["strategy_key"],
        "symbol": row["symbol"],
        "duration": row["duration"],
        "entryOpenTime": row["open_time"],
        "stage": row["stage"],
        "status": row["status"],
        "reason": row["reason"],
        "details": json.loads(row["details_json"]),
        "createdAt": row["created_at"],
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
