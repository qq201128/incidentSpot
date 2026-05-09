from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from app.services.live_order_settings import FIXED_PAYOUT_RATIO

PROJECT_ROOT_PARENT_INDEX = 3
PROJECT_ROOT = Path(__file__).resolve().parents[PROJECT_ROOT_PARENT_INDEX]
LOG_PATH = PROJECT_ROOT / "runtime" / "live-order-failures.log"
LOGGER_NAME = "incident_spot.live_order_failure"
LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"


def log_live_order_failure(ctx: Any, exc: Exception) -> None:
    details = _failure_details(ctx, exc)
    _failure_logger().error(
        "live order failed %s",
        json.dumps(details, ensure_ascii=False, default=str),
        exc_info=(type(exc), exc, exc.__traceback__),
    )

def _failure_logger() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        return logger
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(handler)
    return logger


def _failure_details(ctx: Any, exc: Exception) -> dict[str, Any]:
    return {
        "symbol": ctx.symbol,
        "eventInterval": ctx.event_interval,
        "side": ctx.side,
        "qty": ctx.payload.order.qty,
        "payoutRatio": FIXED_PAYOUT_RATIO,
        "strikeValue": ctx.entry_price,
        "eventTitle": ctx.payload.event.title,
        "eventEndTime": ctx.payload.event.endTime,
        "errorType": type(exc).__name__,
        "error": str(exc),
    }
