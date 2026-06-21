from __future__ import annotations

from typing import Any

from app.db.session import get_conn
from app.services.auto_trade_execution import create_trade_from_prediction
from app.services.auto_trade_types import AutoTradeSettings
from app.services.candidate_regime_admission import CandidateRegimeAdmission, evaluate_candidate_regime_admission
from app.services.paper_live_failure_store import log_prediction_failure
from app.services.position_guard import has_open_position
from app.services.prediction_cache_service import EXECUTION_BLOCKED, EXECUTION_EVENT_CREATED, mark_prediction_execution
from app.services.rule_config import DURATION_TO_MINUTES

CANDIDATE_REGIME_ADMISSION_STAGE = "candidate_regime_admission"
MARKET_REGIME_GATE_STAGE = CANDIDATE_REGIME_ADMISSION_STAGE


def create_batch_combo_simulation_trade(
    parent: AutoTradeSettings,
    prediction: dict[str, Any],
) -> dict[str, Any] | None:
    settings = _batch_settings(parent, prediction)
    if _live_trading_enabled(settings):
        _mark_execution(prediction, EXECUTION_BLOCKED, "live_trading_enabled")
        return None
    if _has_open_position(settings):
        _mark_execution(prediction, EXECUTION_BLOCKED, "open_position_exists")
        return None
    admission = evaluate_candidate_regime_admission(prediction)
    if not admission.allowed:
        _log_candidate_regime_skip(settings, prediction, admission)
        _mark_execution(prediction, EXECUTION_BLOCKED, admission.reason)
        return None
    result = create_trade_from_prediction(settings, prediction, regime_decision=admission)
    _mark_execution(prediction, EXECUTION_EVENT_CREATED, admission.reason, event_id=result.get("eventId"))
    return result


def _batch_settings(parent: AutoTradeSettings, prediction: dict[str, Any]) -> AutoTradeSettings:
    duration = str(prediction["duration"])
    return AutoTradeSettings(
        strategy_key=str(prediction["strategy_key"]),
        enabled=True,
        symbol=str(prediction["symbol"]).strip().upper(),
        duration=duration,
        duration_minutes=int(DURATION_TO_MINUTES[duration]),
        qty=float(parent.qty),
        live_trading_enabled=False,
    )


def _has_open_position(settings: AutoTradeSettings) -> bool:
    conn = get_conn()
    try:
        return has_open_position(
            conn,
            settings.symbol,
            settings.strategy_key,
            event_interval=settings.duration,
            require_market_regime_gate_passed=True,
        )
    finally:
        conn.close()


def _live_trading_enabled(settings: AutoTradeSettings) -> bool:
    conn = get_conn()
    try:
        row = conn.execute(
            """
            SELECT live_trading_enabled
            FROM auto_trade_strategies
            WHERE strategy_key = ? AND symbol = ? AND duration = ?
            """,
            (settings.strategy_key, settings.symbol, settings.duration),
        ).fetchone()
        return bool(row and row["live_trading_enabled"])
    finally:
        conn.close()


def _log_candidate_regime_skip(
    settings: AutoTradeSettings,
    prediction: dict[str, Any],
    decision: CandidateRegimeAdmission,
) -> None:
    log_prediction_failure(
        candidate_key=_candidate_key(prediction),
        strategy_key=settings.strategy_key,
        symbol=settings.symbol,
        duration=settings.duration,
        stage=CANDIDATE_REGIME_ADMISSION_STAGE,
        reason=decision.reason,
        details={
            "mode": decision.mode,
            "openTime": int(prediction["open_time"]),
            "direction": str(prediction["direction"]),
            "regime": decision.regime,
            "sampleCount": decision.sample_count,
            "metrics": decision.metrics,
        },
    )


def _candidate_key(prediction: dict[str, Any]) -> str:
    for key in ("signal_key", "model_version", "high_winrate_rule", "strategy_key"):
        value = prediction.get(key)
        if value:
            return str(value)
    return str(prediction["strategy_key"])


def _mark_execution(prediction: dict[str, Any], status: str, reason: str, *, event_id: Any = None) -> None:
    prediction_id = prediction.get("id")
    if prediction_id is None:
        return
    mark_prediction_execution(
        int(prediction_id),
        status=status,
        reason=reason,
        event_id=None if event_id is None else int(event_id),
    )
