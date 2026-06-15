from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from app.services.background_loop_status import record_loop_failure, record_loop_start, record_loop_stopped, record_loop_success
from app.services.background_threads import run_blocking_daemon
from app.services.settlement_service import scan_due_open_events, settle_event

logger = logging.getLogger(__name__)
LOOP_NAME = "auto_settlement"


@dataclass(frozen=True)
class SettlementScanResult:
    due_ids: list[int]
    settled_count: int
    failed_ids: list[int]
    invalid_events: list[dict]


async def auto_settlement_loop(stop_event: asyncio.Event, poll_seconds: int = 3) -> None:
    record_loop_start(LOOP_NAME, {"pollSeconds": poll_seconds})
    if stop_event.is_set():
        record_loop_stopped(LOOP_NAME, "stop_before_first_scan")
        return
    while not stop_event.is_set():
        try:
            await run_blocking_daemon(_run_settlement_scan_once)
        except Exception as exc:
            record_loop_failure(LOOP_NAME, exc, {"stage": "scan"})
            logger.exception("auto settlement scan failed")
        if await _wait_for_next_poll(stop_event, poll_seconds):
            record_loop_stopped(LOOP_NAME, "stop_between_scans")
            return


def _run_settlement_scan_once() -> None:
    scan = scan_due_open_events()
    _settle_due_events(scan.due_ids, scan.invalid_events)


def _settle_due_events(due_ids: list[int], invalid_events: list[dict]) -> None:
    settled_count = 0
    failed_ids = []
    for event_id in due_ids:
        if _settle_one_event(event_id):
            settled_count += 1
        else:
            failed_ids.append(event_id)
    _record_settlement_scan_result(
        SettlementScanResult(
            due_ids=due_ids,
            settled_count=settled_count,
            failed_ids=failed_ids,
            invalid_events=invalid_events,
        )
    )


def _settle_one_event(event_id: int) -> bool:
    try:
        result = settle_event(event_id)
        logger.info("auto settled event=%s result=%s", event_id, result.get("result"))
        return True
    except Exception as exc:
        record_loop_failure(LOOP_NAME, exc, {"eventId": event_id})
        logger.exception("auto settlement failed for event=%s", event_id)
        return False


def _record_settlement_scan_result(result: SettlementScanResult) -> None:
    details = {
        "dueCount": len(result.due_ids),
        "settledCount": result.settled_count,
        "failedEventIds": result.failed_ids,
        "invalidEvents": result.invalid_events,
    }
    if result.failed_ids or result.invalid_events:
        record_loop_failure(LOOP_NAME, RuntimeError("auto settlement failed for events"), details)
        return
    record_loop_success(LOOP_NAME, details)


async def _wait_for_next_poll(stop_event: asyncio.Event, poll_seconds: int) -> bool:
    try:
        await asyncio.wait_for(stop_event.wait(), timeout=poll_seconds)
        return True
    except TimeoutError:
        return False
