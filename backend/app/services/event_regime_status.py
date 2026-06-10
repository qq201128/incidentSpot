from __future__ import annotations

from typing import Any

from app.services.event_regime_detector import EventRegimeDataError, detect_event_regime


def market_regime_status(symbol: str, duration: str, open_time: int) -> dict[str, Any]:
    try:
        regime = detect_event_regime(symbol, duration, int(open_time))
    except EventRegimeDataError as exc:
        return {"ready": False, "reason": str(exc)}
    return {
        "ready": True,
        "trendState": regime.trend_state,
        "volatilityState": regime.volatility_state,
        "regimeLabel": regime.regime_label,
        "confidence": regime.confidence,
        "reasonCodes": list(regime.reason_codes),
        "metrics": regime.metrics,
    }
