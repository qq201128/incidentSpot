from __future__ import annotations

import asyncio
import logging

from app.services.settlement_service import get_due_open_event_ids, settle_event

logger = logging.getLogger(__name__)


async def auto_settlement_loop(stop_event: asyncio.Event, poll_seconds: int = 5) -> None:
    while not stop_event.is_set():
        try:
            due_ids = get_due_open_event_ids()
            for event_id in due_ids:
                try:
                    result = settle_event(event_id)
                    logger.info("auto settled event=%s result=%s", event_id, result.get("result"))
                except Exception:
                    logger.exception("auto settlement failed for event=%s", event_id)
        except Exception:
            logger.exception("auto settlement scan failed")

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=poll_seconds)
        except TimeoutError:
            continue
