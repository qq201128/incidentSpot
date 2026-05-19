from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from app.services.factor_learning_common import utc_now
from app.services.factor_learning_memory_store import (
    FACTOR_LEARNING_VERSION,
    load_factor_learning_memory,
    save_factor_learning_memory,
)
from app.services.rule_config import SUPPORTED_RULE_DURATIONS


def mark_factor_learning_refresh_queued(symbol: str, duration: str, *, run_agent: bool) -> dict[str, Any]:
    return _save_refresh_status(symbol, duration, "queued", run_agent=run_agent)


def mark_factor_learning_refresh_running(symbol: str, duration: str, *, run_agent: bool) -> dict[str, Any]:
    return _save_refresh_status(symbol, duration, "running", run_agent=run_agent)


def mark_factor_learning_refresh_failed(
    symbol: str,
    duration: str,
    error: str,
    *,
    run_agent: bool,
) -> dict[str, Any]:
    return _save_refresh_status(symbol, duration, "failed", error, run_agent=run_agent)


def mark_factor_learning_refresh_completed(memory: dict[str, Any], *, run_agent: bool) -> dict[str, Any]:
    updated = deepcopy(memory)
    updated["refreshTask"] = _refresh_task_payload("completed", run_agent)
    return _save_memory_payload(updated)


def _save_refresh_status(
    symbol: str,
    duration: str,
    status: str,
    error: str | None = None,
    *,
    run_agent: bool,
) -> dict[str, Any]:
    _validate_duration(duration)
    sym = symbol.strip().upper()
    memory = load_factor_learning_memory(sym, duration) or _queued_memory(sym, duration)
    updated = deepcopy(memory)
    updated["refreshTask"] = _refresh_task_payload(status, run_agent, error)
    return _save_memory_payload(updated)


def _refresh_task_payload(status: str, run_agent: bool, error: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"status": status, "runAgent": run_agent, "updatedAt": utc_now()}
    if error:
        payload["error"] = error
    return payload


def _queued_memory(symbol: str, duration: str) -> dict[str, Any]:
    return {
        "version": FACTOR_LEARNING_VERSION,
        "symbol": symbol,
        "duration": duration,
        "updatedAt": utc_now(),
        "source": {"status": "queued"},
    }


def _save_memory_payload(memory: dict[str, Any]) -> dict[str, Any]:
    payload = deepcopy(memory)
    payload.pop("memoryPath", None)
    path = save_factor_learning_memory(payload)
    return {**payload, "memoryPath": _path_payload(path)}


def _validate_duration(duration: str) -> None:
    if duration not in SUPPORTED_RULE_DURATIONS:
        raise ValueError(f"unsupported duration: {duration}")


def _path_payload(path: Path) -> str:
    return str(path)
