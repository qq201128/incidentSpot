from __future__ import annotations

import asyncio

from app.main import health
from app.services.background_loop_status import (
    background_loop_statuses,
    record_loop_failure,
    record_loop_start,
    record_loop_success,
    reset_background_loop_statuses,
)


def test_background_loop_status_records_failure_details() -> None:
    reset_background_loop_statuses()

    record_loop_start("factor_ranking", {"intervalSeconds": 60})
    record_loop_failure("factor_ranking", RuntimeError("rank failed"), {"symbol": "BTCUSDT"})

    status = background_loop_statuses()["factor_ranking"]
    assert status["status"] == "failed"
    assert status["lastError"] == "rank failed"
    assert status["lastExceptionType"] == "RuntimeError"
    assert status["lastFailureDetails"] == {"symbol": "BTCUSDT"}


def test_background_loop_status_records_success_without_losing_failure() -> None:
    reset_background_loop_statuses()

    record_loop_failure("market_context", RuntimeError("market failed"))
    record_loop_success("market_context", {"symbolCount": 2})

    status = background_loop_statuses()["market_context"]
    assert status["status"] == "passed"
    assert status["lastError"] == "market failed"
    assert status["lastSuccessDetails"] == {"symbolCount": 2}


def test_health_exposes_background_loop_status() -> None:
    reset_background_loop_statuses()
    record_loop_failure("auto_trade", RuntimeError("trade failed"))

    payload = asyncio.run(health())

    assert payload["background"]["auto_trade"]["status"] == "failed"
    assert payload["background"]["auto_trade"]["lastError"] == "trade failed"
