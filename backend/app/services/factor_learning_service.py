from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from app.db.session import get_conn
from app.services.factor_combination_cache_service import get_cached_combination_ranking
from app.services.factor_learning_common import utc_now
from app.services.factor_frame_service import load_factor_frame
from app.services.factor_learning_core import build_factor_learning_memory
from app.services.factor_learning_llm_agent import (
    AGENT_NAME,
    AGENT_PROVIDER,
    attach_llm_agent_review,
)
from app.services.factor_learning_memory_store import (
    load_factor_learning_memory,
    save_factor_learning_memory,
)
from app.services.forward_validation_service import settle_due_predictions
from app.services.rule_config import SUPPORTED_RULE_DURATIONS
from app.services.strategy_registry import FACTOR_COMBO_STRATEGY_KEY


def get_factor_learning_memory(symbol: str, duration: str) -> dict[str, Any] | None:
    _validate_duration(duration)
    return load_factor_learning_memory(symbol, duration)


def refresh_factor_learning_memory(
    symbol: str,
    duration: str,
    ranking_report: dict[str, Any] | None = None,
    *,
    run_llm_agent: bool = False,
) -> dict[str, Any]:
    _validate_duration(duration)
    sym = symbol.strip().upper()
    report = ranking_report or _cached_ranking_or_raise(sym, duration)
    settlement = settle_due_predictions(sym, duration)
    frame = load_factor_frame(sym)
    predictions = _settled_factor_combo_predictions(sym, duration)
    memory = build_factor_learning_memory(
        frame,
        report,
        predictions,
        symbol=sym,
        duration=duration,
        settlement_sweep=settlement,
    )
    if run_llm_agent:
        return _attach_agent_review_and_save(memory)
    return _save_memory_payload(memory)


def mark_factor_learning_agent_pending(memory: dict[str, Any]) -> dict[str, Any]:
    return _save_factor_learning_agent_status(memory, "pending")


def run_factor_learning_llm_agent(symbol: str, duration: str) -> dict[str, Any]:
    _validate_duration(duration)
    memory = load_factor_learning_memory(symbol, duration)
    if memory is None:
        raise ValueError(f"factor learning memory not found for {symbol.upper()} {duration}")
    return _attach_agent_review_and_save(memory)


def _attach_agent_review_and_save(memory: dict[str, Any]) -> dict[str, Any]:
    try:
        return _save_memory_payload(attach_llm_agent_review(memory))
    except Exception as exc:
        _save_factor_learning_agent_status(memory, "failed", str(exc))
        raise


def _save_factor_learning_agent_status(
    memory: dict[str, Any],
    status: str,
    error: str | None = None,
) -> dict[str, Any]:
    updated = deepcopy(memory)
    updated["llmAgent"] = _agent_status_payload(status, error)
    return _save_memory_payload(updated)


def _agent_status_payload(status: str, error: str | None) -> dict[str, Any]:
    payload = {
        "agent": AGENT_NAME,
        "provider": AGENT_PROVIDER,
        "status": status,
        "updatedAt": utc_now(),
    }
    if error:
        payload["error"] = error
    return payload


def _save_memory_payload(memory: dict[str, Any]) -> dict[str, Any]:
    payload = deepcopy(memory)
    payload.pop("memoryPath", None)
    path = save_factor_learning_memory(payload)
    return {**payload, "memoryPath": _path_payload(path)}


def _cached_ranking_or_raise(symbol: str, duration: str) -> dict[str, Any]:
    cached = get_cached_combination_ranking(symbol, duration)
    if cached is None:
        raise ValueError(f"no cached factor combination ranking for {symbol} {duration}")
    return cached


def _settled_factor_combo_predictions(symbol: str, duration: str) -> list[dict[str, Any]]:
    conn = get_conn()
    try:
        rows = conn.execute(
            """
            SELECT open_time, direction, confidence, trade_quality_score,
                   actual_return, prediction_correct, high_winrate_rule
            FROM predictions
            WHERE strategy_key = ? AND symbol = ? AND duration = ?
              AND settled_at IS NOT NULL
            ORDER BY open_time
            """,
            (FACTOR_COMBO_STRATEGY_KEY, symbol.upper(), duration),
        ).fetchall()
    finally:
        conn.close()
    return [dict(row) for row in rows]


def _validate_duration(duration: str) -> None:
    if duration not in SUPPORTED_RULE_DURATIONS:
        raise ValueError(f"unsupported duration: {duration}")


def _path_payload(path: Path) -> str:
    return str(path)
