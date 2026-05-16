from __future__ import annotations

from typing import Any

from app.db.session import get_conn
from app.services.auto_trade_execution import create_trade_from_prediction
from app.services.auto_trade_types import AutoTradeSettings
from app.services.position_guard import has_open_position
from app.services.rule_config import DURATION_TO_MINUTES


def create_batch_combo_simulation_trade(
    parent: AutoTradeSettings,
    prediction: dict[str, Any],
) -> dict[str, Any] | None:
    if not prediction.get("trade_quality_passed"):
        return None
    settings = _batch_settings(parent, prediction)
    if _has_open_position(settings):
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
        )
    finally:
        conn.close()
