from __future__ import annotations

from datetime import datetime, timezone

from app.db.session import get_conn
from app.services.paper_live_stage_log import log_prediction_generation_stages
from app.services.prediction_policy import trade_policy_payload
from app.services.strategy_registry import DEFAULT_STRATEGY_KEY


INSERT_PREDICTION_SQL = """INSERT INTO predictions(
  signal_key, strategy_key, symbol, duration, open_time, direction, probability_up, confidence,
  certainty_label, trade_quality_score, trade_quality_passed, trade_quality_gate,
  high_winrate_gate, high_winrate_rule, high_winrate_gate_passed,
  high_winrate_gate_value, high_winrate_gate_min, entry_price, expected_return,
  model_version, model_family, validation_win_rate, feature_window, model_duration, model_trained_at,
  data_freshness_status, missing_feature_status, created_at
)
VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""


def save_prediction(result: dict, *, allow_existing: bool = False) -> bool:
    payload = _paper_live_boundary_payload(result)
    conn = get_conn()
    try:
        if not allow_existing and prediction_exists_conn(conn, payload):
            return False
        conn.execute(INSERT_PREDICTION_SQL, _prediction_values(payload))
        log_prediction_generation_stages(conn, payload)
        conn.commit()
        return True
    finally:
        conn.close()


def prediction_exists(
    *,
    strategy_key: str | None = None,
    signal_key: str | None = None,
    symbol: str,
    duration: str,
    open_time: int,
) -> bool:
    conn = get_conn()
    try:
        request = {
            "signal_key": signal_key,
            "strategy_key": strategy_key,
            "symbol": symbol,
            "duration": duration,
            "open_time": open_time,
        }
        return prediction_exists_conn(conn, request)
    finally:
        conn.close()


def prediction_exists_conn(conn, request: dict) -> bool:
    row = conn.execute(
        """
        SELECT 1 FROM predictions
        WHERE signal_key = ? AND symbol = ? AND duration = ? AND open_time = ?
        LIMIT 1
        """,
        (
            _signal_key(request),
            request["symbol"].upper(),
            request["duration"],
            int(request["open_time"]),
        ),
    ).fetchone()
    return row is not None


def get_latest_prediction(
    symbol: str,
    duration: str,
    *,
    strategy_key: str = DEFAULT_STRATEGY_KEY,
    signal_key: str | None = None,
) -> dict:
    conn = get_conn()
    row = conn.execute(
        """
        SELECT
          signal_key, strategy_key, symbol, duration, open_time, direction, probability_up, confidence, certainty_label,
          trade_quality_score, trade_quality_passed, trade_quality_gate,
          high_winrate_gate, high_winrate_rule, high_winrate_gate_passed,
          high_winrate_gate_value, high_winrate_gate_min, entry_price, exit_price,
          actual_return, prediction_correct, settled_at, expected_return,
          model_version, model_family, validation_win_rate, feature_window, model_duration, model_trained_at,
          data_freshness_status, missing_feature_status, created_at
        FROM predictions
        WHERE signal_key = ? AND symbol = ? AND duration = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (signal_key or strategy_key, symbol.upper(), duration),
    ).fetchone()
    conn.close()
    if row is None:
        raise ValueError(f"no cached prediction for {symbol.upper()} {duration}")
    return prediction_response(dict(row))


def prediction_response(result: dict) -> dict:
    generated_at = result["created_at"] if "created_at" in result else _utc_now_iso()
    generated_ms = _parse_iso_ms(generated_at)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    return {
        "symbol": result["symbol"],
        "signalKey": _signal_key(result),
        "strategyKey": _strategy_key(result),
        "duration": result["duration"],
        "direction": result["direction"],
        "probabilityUp": result["probability_up"],
        "confidence": result["confidence"],
        "certaintyLabel": result["certainty_label"],
        "threshold": result.get("threshold"),
        "openTime": result.get("open_time"),
        "entryOpenTime": result.get("open_time"),
        "tradeQualityScore": result.get("trade_quality_score"),
        "tradeQualityPassed": _as_bool(result.get("trade_quality_passed")),
        "tradeQualityGate": result.get("trade_quality_gate"),
        "highWinrateGate": result.get("high_winrate_gate"),
        "highWinrateRule": result.get("high_winrate_rule"),
        "highWinrateGatePassed": _as_bool(result.get("high_winrate_gate_passed")),
        "highWinrateGateValue": result.get("high_winrate_gate_value"),
        "highWinrateGateMin": result.get("high_winrate_gate_min"),
        "entryPrice": result.get("entry_price"),
        "exitPrice": result.get("exit_price"),
        "settlementPrice": result.get("exit_price"),
        "actualReturn": result.get("actual_return"),
        "predictionCorrect": _as_bool(result.get("prediction_correct")),
        "expectedReturn": result.get("expected_return"),
        "modelVersion": result.get("model_version"),
        "modelFamily": result.get("model_family"),
        "validationWinRate": result.get("validation_win_rate"),
        "featureWindow": result.get("feature_window"),
        "modelDuration": result.get("model_duration"),
        "modelTrainedAt": result.get("model_trained_at"),
        "dataFreshnessStatus": result.get("data_freshness_status"),
        "missingFeatureStatus": result.get("missing_feature_status"),
        "settledAt": result.get("settled_at"),
        "generatedAt": generated_at,
        "predictionCreatedAt": generated_at,
        "ageMs": max(now_ms - generated_ms, 0),
        "signalSource": result.get("signal_source"),
        "ruleScore": result.get("rule_score"),
        "ruleReasons": result.get("rule_reasons"),
        "orderbook": result.get("orderbook"),
        "timeframeVotes": result.get("timeframe_votes"),
        **trade_policy_payload(result["duration"], strategy_key=_strategy_key(result)),
    }


def _prediction_values(result: dict) -> tuple:
    return (
        _signal_key(result), _strategy_key(result), result["symbol"], result["duration"], result["open_time"],
        result["direction"], result["probability_up"], result["confidence"],
        result["certainty_label"], result.get("trade_quality_score"),
        int(bool(result.get("trade_quality_passed"))), result.get("trade_quality_gate"),
        result.get("high_winrate_gate"), result.get("high_winrate_rule"),
        int(bool(result.get("high_winrate_gate_passed"))), result.get("high_winrate_gate_value"),
        result.get("high_winrate_gate_min"), result.get("entry_price"), result.get("expected_return"),
        result.get("model_version"), result.get("model_family"), result.get("validation_win_rate"),
        result.get("feature_window"), result.get("model_duration"), result.get("model_trained_at"),
        result.get("data_freshness_status"), result.get("missing_feature_status"), _utc_now_iso(),
    )


def _paper_live_boundary_payload(result: dict) -> dict:
    source_open_time = _source_open_time(result)
    if source_open_time is None or source_open_time <= int(result["open_time"]):
        return result
    return {
        **result,
        "data_freshness_status": "invalid_data_leakage",
        "missing_feature_status": result.get("missing_feature_status") or "complete",
        "future_data_leakage_reason": "source_open_time_after_entry_open_time",
        "future_data_leakage_source_open_time": source_open_time,
    }


def _source_open_time(result: dict) -> int | None:
    for key in ("sourceOpenTime", "source_open_time", "featureOpenTime", "feature_open_time"):
        value = result.get(key)
        if value is not None:
            return int(value)
    return None


def _signal_key(result: dict) -> str:
    return str(result.get("signal_key") or result.get("strategy_key") or DEFAULT_STRATEGY_KEY)


def _strategy_key(result: dict) -> str:
    return str(result.get("strategy_key") or DEFAULT_STRATEGY_KEY)


def _as_bool(value) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _parse_iso_ms(value: str) -> int:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
