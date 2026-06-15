from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any

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
    exception: BaseException


class ShadowBackfillBatchError(RuntimeError):
    def __init__(self, failures: list[ShadowBackfillFailure]) -> None:
        self.failures = tuple(failures)
        self.details = [_failure_detail(failure) for failure in failures]
        targets = ", ".join(f"{item.family}:{item.symbol}:{item.duration}" for item in failures)
        super().__init__(f"model family shadow backfill failed for: {targets}")


DEFAULT_BACKFILL_TIMEOUT_SECONDS = 20.0
DEFAULT_BACKFILL_ENTRY_LIMIT = 24


async def backfill_shadow_predictions(settings_list: list[AutoTradeSettings], deps: dict[str, Any]) -> None:
    targets = deps["ready_targets"](settings_list)
    if not targets:
        return
    timeout_seconds = _backfill_timeout_seconds()
    entry_limit = _backfill_entry_limit()
    summaries = await asyncio.gather(
        *(
            _backfill_one(
                deps,
                family,
                symbol,
                duration,
                timeout_seconds=timeout_seconds,
                entry_limit=entry_limit,
            )
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


async def _backfill_one(
    deps: dict[str, Any],
    family: str,
    symbol: str,
    duration: str,
    *,
    timeout_seconds: float,
    entry_limit: int,
) -> dict[str, Any]:
    return await asyncio.wait_for(
        asyncio.to_thread(
            deps["backfill"],
            family,
            symbol,
            duration,
            deps["current_entry"](duration),
            max_entries=entry_limit,
        ),
        timeout=timeout_seconds,
    )


def handle_backfill_summary(
    target: ShadowBackfillTarget,
    summary: object,
    logger: logging.Logger,
) -> ShadowBackfillFailure | None:
    if isinstance(summary, BaseException):
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


def _backfill_timeout_seconds() -> float:
    raw = os.getenv("MODEL_SHADOW_BACKFILL_TIMEOUT_SECONDS", str(DEFAULT_BACKFILL_TIMEOUT_SECONDS))
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"MODEL_SHADOW_BACKFILL_TIMEOUT_SECONDS must be numeric: {raw!r}") from exc
    return max(0.1, value)


def _backfill_entry_limit() -> int:
    raw = os.getenv("MODEL_SHADOW_BACKFILL_ENTRY_LIMIT", str(DEFAULT_BACKFILL_ENTRY_LIMIT))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"MODEL_SHADOW_BACKFILL_ENTRY_LIMIT must be an integer: {raw!r}") from exc
    return max(1, value)
