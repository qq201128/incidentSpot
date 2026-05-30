from __future__ import annotations

from app.services import (
    paper_live_candidate_status_store,
    paper_live_failure_store,
    paper_live_stage_log,
)

BROKEN_DETAILS = "{broken"


def test_status_change_payload_exposes_details_parse_error() -> None:
    payload = paper_live_candidate_status_store._history_payload(
        {
            "candidate_key": "factor_alpha",
            "symbol": "BTCUSDT",
            "duration": "10m",
            "old_status": "paper_collecting",
            "new_status": "paper_failed",
            "reason": "paper_live_win_rate_below_target",
            "details_json": BROKEN_DETAILS,
            "changed_at": "2026-05-26T00:00:00+00:00",
        }
    )

    assert payload["details"] == BROKEN_DETAILS
    assert payload["detailsParseError"]["field"] == "details"
    assert payload["detailsParseError"]["exceptionType"] == "JSONDecodeError"


def test_prediction_failure_payload_exposes_details_parse_error() -> None:
    payload = paper_live_failure_store._failure_payload(
        {
            "candidate_key": "factor_alpha",
            "strategy_key": "factor_alpha",
            "stage": "prediction_generation",
            "reason": "missing_feature",
            "details_json": BROKEN_DETAILS,
            "created_at": "2026-05-26T00:00:00+00:00",
        }
    )

    assert payload["details"] == BROKEN_DETAILS
    assert payload["detailsParseError"]["exceptionType"] == "JSONDecodeError"


def test_stage_log_payload_exposes_details_parse_error() -> None:
    payload = paper_live_stage_log._stage_log_payload(
        {
            "signal_key": "factor_alpha",
            "strategy_key": "factor_alpha",
            "symbol": "BTCUSDT",
            "duration": "10m",
            "open_time": 123,
            "stage": "feature_construction",
            "status": "failed",
            "reason": "missing_feature",
            "details_json": BROKEN_DETAILS,
            "created_at": "2026-05-26T00:00:00+00:00",
        }
    )

    assert payload["details"] == BROKEN_DETAILS
    assert payload["detailsParseError"]["exceptionType"] == "JSONDecodeError"
