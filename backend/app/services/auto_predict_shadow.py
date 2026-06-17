from __future__ import annotations

import asyncio
import logging
import os
import time
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
DEFAULT_BACKFILL_CONCURRENCY = 4


async def backfill_shadow_predictions(settings_list: list[AutoTradeSettings], deps: dict[str, Any]) -> None:
    targets = deps["ready_targets"](settings_list)
    if not targets:
        return
    save_backfill = deps.get("save_backfill")
    timeout_seconds = _backfill_timeout_seconds()
    entry_limit = _backfill_entry_limit()
    concurrency = _backfill_concurrency()
    current_entry_only = bool(deps.get("current_entry_only"))
    cycle_context = deps.get("cycle_context")
    summaries = await _backfill_targets(
        targets,
        deps,
        timeout_seconds=timeout_seconds,
        entry_limit=entry_limit,
        current_entry_only=current_entry_only,
        cycle_context=cycle_context,
        concurrency=concurrency,
    )
    failures = []
    for target_tuple, summary in zip(targets, summaries):
        target = _backfill_target(target_tuple)
        try:
            saved_summary = _save_backfill_summary(summary, save_backfill)
        except BaseException as exc:
            saved_summary = exc
        failure = handle_backfill_summary(target, saved_summary, deps["logger"])
        if failure is not None:
            failures.append(failure)
    if failures:
        raise ShadowBackfillBatchError(failures)


async def _backfill_targets(
    targets: list[tuple[str, str, str]],
    deps: dict[str, Any],
    *,
    timeout_seconds: float,
    entry_limit: int,
    current_entry_only: bool,
    cycle_context: Any | None,
    concurrency: int,
) -> list[object]:
    limiter = asyncio.Semaphore(max(1, int(concurrency)))

    async def run_one(target: tuple[str, str, str]) -> object:
        async with limiter:
            family, symbol, duration = target
            return await _backfill_one(
                deps,
                family,
                symbol,
                duration,
                timeout_seconds=timeout_seconds,
                entry_limit=entry_limit,
                current_entry_only=current_entry_only,
                cycle_context=cycle_context,
            )

    return await asyncio.gather(*(run_one(target) for target in targets), return_exceptions=True)


async def _backfill_one(
    deps: dict[str, Any],
    family: str,
    symbol: str,
    duration: str,
    *,
    timeout_seconds: float,
    entry_limit: int,
    current_entry_only: bool,
    cycle_context: Any | None,
) -> object:
    started = time.perf_counter()
    summary = await asyncio.to_thread(
        deps.get("build_backfill") or deps["backfill"],
        family,
        symbol,
        duration,
        deps["current_entry"](duration),
        max_entries=entry_limit,
        current_entry_only=current_entry_only,
        cycle_context=cycle_context,
    )
    elapsed = time.perf_counter() - started
    if elapsed > timeout_seconds:
        raise TimeoutError(
            f"model family shadow backfill exceeded {timeout_seconds:.1f}s: "
            f"{family}:{symbol}:{duration} took {elapsed:.3f}s"
        )
    return summary



def _save_backfill_summary(summary: object, save_backfill) -> object:
    if isinstance(summary, BaseException) or save_backfill is None:
        return summary
    return save_backfill(summary)


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


def _backfill_concurrency() -> int:
    raw = os.getenv("MODEL_SHADOW_BACKFILL_CONCURRENCY", str(DEFAULT_BACKFILL_CONCURRENCY))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"MODEL_SHADOW_BACKFILL_CONCURRENCY must be an integer: {raw!r}") from exc
    return max(1, value)
