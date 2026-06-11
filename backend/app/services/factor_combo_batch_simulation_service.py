from __future__ import annotations

from typing import Any

from app.db.session import get_conn
from app.services.auto_trade_execution import create_trade_from_prediction
from app.services.auto_trade_types import AutoTradeSettings
from app.services.market_regime_trade_gate import MarketRegimeTradeDecision, evaluate_market_regime_trade_gate
from app.services.paper_live_failure_store import log_prediction_failure
from app.services.position_guard import has_open_position
from app.services.rule_config import DURATION_TO_MINUTES

MARKET_REGIME_GATE_STAGE = "market_regime_trade_gate"


def create_batch_combo_simulation_trade(
    parent: AutoTradeSettings,
    prediction: dict[str, Any],
) -> dict[str, Any] | None:
    settings = _batch_settings(parent, prediction)
    if _has_open_position(settings):
        return None
    regime_decision = evaluate_market_regime_trade_gate(
        symbol=settings.symbol,
        duration=settings.duration,
        open_time=int(prediction["open_time"]),
        direction=str(prediction["direction"]),
    )
    if not regime_decision.allowed:
        _log_market_regime_skip(settings, prediction, regime_decision)
        return None
    return create_trade_from_prediction(settings, prediction)


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


def _log_market_regime_skip(
    settings: AutoTradeSettings,
    prediction: dict[str, Any],
    decision: MarketRegimeTradeDecision,
) -> None:
    log_prediction_failure(
        candidate_key=_candidate_key(prediction),
        strategy_key=settings.strategy_key,
        symbol=settings.symbol,
        duration=settings.duration,
        stage=MARKET_REGIME_GATE_STAGE,
        reason=decision.reason,
        details={
            "mode": decision.mode,
            "openTime": int(prediction["open_time"]),
            "direction": str(prediction["direction"]),
            "regime": decision.regime,
        },
    )


def _candidate_key(prediction: dict[str, Any]) -> str:
    for key in ("signal_key", "model_version", "high_winrate_rule", "strategy_key"):
        value = prediction.get(key)
        if value:
            return str(value)
    return str(prediction["strategy_key"])
