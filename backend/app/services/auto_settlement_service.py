from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from app.services.background_loop_status import record_loop_failure, record_loop_start, record_loop_stopped, record_loop_success
from app.services.background_threads import run_blocking_daemon
from app.services.settlement_service import scan_due_open_events
from app.services.batch_settlement_service import batch_settle_due_events

logger = logging.getLogger(__name__)
LOOP_NAME = "auto_settlement"


@dataclass(frozen=True)
class SettlementScanResult:
    due_ids: list[int]
    settled_count: int
    failed_ids: list[int]
    invalid_events: list[dict]
    price_fetch_count: int = 0  # 新增：API调用次数统计


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
    _settle_due_events_batch(scan.due_ids, scan.invalid_events)


def _settle_due_events_batch(due_ids: list[int], invalid_events: list[dict]) -> None:
    """使用批量结算服务处理到期事件"""
    if not due_ids:
        _record_settlement_scan_result(
            SettlementScanResult(
                due_ids=[],
                settled_count=0,
                failed_ids=[],
                invalid_events=invalid_events,
                price_fetch_count=0,
            )
        )
        return

    try:
        result = batch_settle_due_events(due_ids)
        failed_ids = [evt["eventId"] for evt in result.failed_events]

        logger.info(
            f"Batch settlement completed: {result.settled_count}/{result.total_events} settled, "
            f"API calls: {result.price_fetch_count} (avg {result.total_events / max(result.price_fetch_count, 1):.1f} events/call)"
        )

        _record_settlement_scan_result(
            SettlementScanResult(
                due_ids=due_ids,
                settled_count=result.settled_count,
                failed_ids=failed_ids,
                invalid_events=invalid_events,
                price_fetch_count=result.price_fetch_count,
            )
        )

    except Exception as exc:
        logger.exception("Batch settlement failed, falling back to single-event settlement")
        # 降级：使用原有逻辑逐个结算
        _settle_due_events_fallback(due_ids, invalid_events)

def _settle_due_events_fallback(due_ids: list[int], invalid_events: list[dict]) -> None:
    """降级方案：逐个结算事件"""
    from app.services.settlement_service import settle_event

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
            price_fetch_count=len(due_ids),  # 降级时每个事件一次调用
        )
    )


def _settle_one_event(event_id: int) -> bool:
    from app.services.settlement_service import settle_event

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
        "priceFetchCount": result.price_fetch_count,
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
