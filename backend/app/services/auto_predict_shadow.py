from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Callable

from app.services.auto_trade_types import AutoTradeSettings


@dataclass(frozen=True)
class ShadowBackfillTarget:
    family: str
    symbol: str
    duration: str


@dataclass(frozen=True)
class ShadowBackfillFailure:
    family: str
    symbol: str
    duration: str
    exception: Exception


class ShadowBackfillBatchError(RuntimeError):
    def __init__(self, failures: list[ShadowBackfillFailure]) -> None:
        self.failures = tuple(failures)
        self.details = [_failure_detail(failure) for failure in failures]
        targets = ", ".join(f"{item.family}:{item.symbol}:{item.duration}" for item in failures)
        super().__init__(f"model family shadow backfill failed for: {targets}")


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
    failures = []
    for target_tuple, summary in zip(targets, summaries):
        failure = handle_backfill_summary(_backfill_target(target_tuple), summary, deps["logger"])
        if failure is not None:
            failures.append(failure)
    if failures:
        raise ShadowBackfillBatchError(failures)


def handle_backfill_summary(
    target: ShadowBackfillTarget,
    summary: object,
    logger: logging.Logger,
) -> ShadowBackfillFailure | None:
    if isinstance(summary, Exception):
        logger.error(
            "model family shadow backfill failed family=%s symbol=%s duration=%s",
            target.family,
            target.symbol,
            target.duration,
            exc_info=(type(summary), summary, summary.__traceback__),
        )
        return ShadowBackfillFailure(target.family, target.symbol, target.duration, summary)
    if summary["savedCount"]:
        logger.info("predict: model family shadow backfill summary=%s", summary)
    return None


def ready_shadow_backfill_targets(settings_list: list[AutoTradeSettings], deps: dict[str, Any]) -> list[tuple[str, str, str]]:
    ready = []
    for family, symbol, duration in deps["unique_targets"](settings_list):
        status = deps["lstm_status"](symbol, duration) if family == "lstm" else deps["family_status"](family, symbol, duration)
        if status.get("shadowPredictionReady"):
            ready.append((family, symbol, duration))
    return ready


def log_model_family_shadow_skip(
    logger: logging.Logger,
    *,
    settings: AutoTradeSettings,
    family: str,
    status: dict,
    role: str,
) -> None:
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


def _backfill_target(target: tuple[str, str, str]) -> ShadowBackfillTarget:
    family, symbol, duration = target
    return ShadowBackfillTarget(family, symbol, duration)


def _failure_detail(failure: ShadowBackfillFailure) -> dict[str, str]:
    return {
        "family": failure.family,
        "symbol": failure.symbol,
        "duration": failure.duration,
        "error": str(failure.exception),
        "exceptionType": type(failure.exception).__name__,
    }
