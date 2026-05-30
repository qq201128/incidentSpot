from __future__ import annotations

import json
from typing import Any

from app.services.prediction_cache_service import prediction_response


def test_prediction_response_exposes_metadata_json_parse_errors() -> None:
    payload = prediction_response(_prediction_row("{broken", json.dumps({"score": 0.72})))

    assert payload["walkForwardResult"] == "{broken"
    assert payload["recentRollingResult"] == {"score": 0.72}
    assert payload["metadataParseErrors"][0]["field"] == "walkForwardResult"
    assert payload["metadataParseErrors"][0]["exceptionType"] == "JSONDecodeError"
    assert "Expecting property name" in payload["metadataParseErrors"][0]["error"]


def _prediction_row(walk_forward: Any, recent_rolling: Any) -> dict[str, Any]:
    return {
        "symbol": "BTCUSDT",
        "duration": "10m",
        "open_time": 1_780_000_000_000,
        "direction": "UP",
        "probability_up": 0.68,
        "confidence": 0.68,
        "certainty_label": "medium",
        "walk_forward_result": walk_forward,
        "recent_rolling_result": recent_rolling,
        "created_at": "2026-05-26T00:00:00+00:00",
    }
