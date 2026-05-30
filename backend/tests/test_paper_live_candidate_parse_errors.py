from __future__ import annotations

import json

from app.services import paper_live_candidate_service as service


def test_candidate_payload_exposes_metadata_json_parse_errors() -> None:
    candidate = {
        "signal_key": "factor_alpha",
        "strategy_key": "factor_alpha",
        "high_winrate_rule": "alpha",
        "high_winrate_gate_value": 0.7,
        "high_winrate_gate_min": 0.62,
        "model_family": None,
        "model_version": None,
        "validation_win_rate": None,
        "feature_window": None,
        "oos_win_rate": 0.61,
        "walk_forward_result": "{broken",
        "recent_rolling_result": json.dumps({"winRate": 0.64}),
        "data_freshness_status": "fresh",
        "missing_feature_status": "complete",
        "latest_created_at": "2026-05-26T00:00:00+00:00",
        "first_created_at": "2026-05-26T00:00:00+00:00",
        "prediction_count": 1,
    }

    payload = service._candidate_payload(candidate, [])

    assert payload["walkForwardResult"] == "{broken"
    assert payload["recentRollingResult"] == {"winRate": 0.64}
    assert payload["performanceComparison"]["walkForwardResult"] == "{broken"
    assert payload["performanceComparison"]["recentRollingResult"] == {"winRate": 0.64}
    assert payload["metadataParseErrors"][0]["field"] == "walkForwardResult"
    assert payload["metadataParseErrors"][0]["exceptionType"] == "JSONDecodeError"
    assert "Expecting property name" in payload["metadataParseErrors"][0]["error"]
