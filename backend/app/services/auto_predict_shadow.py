from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from app.services.auto_trade_types import AutoTradeSettings


async def backfill_shadow_predictions(settings_list: list[AutoTradeSettings], deps: dict[str, Any]) -> None:
    targets = deps["ready_targets"](settings_list)
    if not targets:
        return
    summaries = await asyncio.gather(
        *(
            asyncio.to_thread(deps["backfill"], family, symbol, duration, deps["current_entry"](duration))
            for family, symbol, duration in targets
        ),
        return_exceptions=True,
    )
    for (family, symbol, duration), summary in zip(targets, summaries):
        handle_backfill_summary(family, symbol, duration, summary, deps["logger"])


def handle_backfill_summary(family: str, symbol: str, duration: str, summary: object, logger: logging.Logger) -> None:
    if isinstance(summary, Exception):
        logger.error(
            "model family shadow backfill failed family=%s symbol=%s duration=%s",
            family,
            symbol,
            duration,
            exc_info=(type(summary), summary, summary.__traceback__),
        )
        return
    if summary["savedCount"]:
        logger.info("predict: model family shadow backfill summary=%s", summary)


def ready_shadow_backfill_targets(settings_list: list[AutoTradeSettings], deps: dict[str, Any]) -> list[tuple[str, str, str]]:
    ready = []
    for family, symbol, duration in deps["unique_targets"](settings_list):
        status = deps["lstm_status"](symbol, duration) if family == "lstm" else deps["family_status"](family, symbol, duration)
        if status.get("shadowPredictionReady"):
            ready.append((family, symbol, duration))
    return ready


def log_model_family_shadow_skip(logger: logging.Logger, settings: AutoTradeSettings, family: str, status: dict, *, role: str) -> None:
    logger.info(
        "predict: %s shadow skipped role=%s for %s %s reason=%s dependency=%s status=%s artifacts=%s combo=%s",
        family,
        role,
        settings.symbol,
        settings.duration,
        status.get("shadowPredictionBlockedReason"),
        status.get("dependencyStatus"),
        status.get("status"),
        status.get("artifactsReady"),
        status.get("comboSnapshotReason"),
    )
