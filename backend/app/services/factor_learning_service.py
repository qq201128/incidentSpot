from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

from app.db.session import get_conn
from app.services.agent_mined_factor_library import (
    agent_mined_factor_library_summary,
    process_agent_factor_candidates,
)
from app.services.factor_cache_metadata import cache_is_usable
from app.services.factor_combination_cache_service import get_cached_combination_ranking
from app.services.factor_combination_cache_service import save_cached_combination_ranking
from app.services.factor_combination_service import CombinationSearchConfig
from app.services.factor_combination_service import run_factor_combination_ranking_on_frame
from app.services.factor_combo_monitor_service import factor_combo_monitor_report
from app.services.factor_combo_simulation_keys import (
    BATCH_COMBO_KEY_PREFIX,
    BATCH_HIGH_WINRATE_KEY_PREFIX,
    factor_combo_simulation_strategy_keys,
    high_winrate_factor_combo_simulation_strategy_keys,
)
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
from app.services.factor_mined_candidates import materialize_mined_factor_frame
from app.services.factor_mined_library import mined_factor_library_summary
from app.services.forward_validation_service import settle_due_predictions
from app.services.lstm_config import lstm_shadow_strategy_key
from app.services.lstm_shadow_learning import lstm_shadow_learning_summary
from app.services.rule_config import SUPPORTED_RULE_DURATIONS


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
    base_frame = load_factor_frame(sym, duration)
    report = ranking_report or _current_ranking_report(sym, duration, base_frame)
    settlement = settle_due_predictions(sym, duration)
    mined_frame = materialize_mined_factor_frame(base_frame, symbol=sym, duration=duration)
    predictions = _settled_factor_combo_predictions(sym, duration)
    memory = build_factor_learning_memory(
        mined_frame.frame,
        report,
        predictions,
        symbol=sym,
        duration=duration,
        settlement_sweep=settlement,
        mined_frame_failures=list(mined_frame.failures),
        mined_library=mined_factor_library_summary(sym, duration),
        agent_mined_library=agent_mined_factor_library_summary(sym, duration),
        monitoring_report=factor_combo_monitor_report(sym, duration),
        lstm_shadow=lstm_shadow_learning_summary(sym, duration),
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
        reviewed = attach_llm_agent_review(memory)
        frame = load_factor_frame(str(reviewed["symbol"]), str(reviewed["duration"]))
        promoted = process_agent_factor_candidates(reviewed, frame)
        return _save_memory_payload(promoted)
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


def _current_ranking_report(symbol: str, duration: str, frame: pd.DataFrame) -> dict[str, Any]:
    cached = _usable_cached_ranking(symbol, duration)
    if cached is not None:
        return {**cached, "learningRefreshSource": "cache"}
    report = run_factor_combination_ranking_on_frame(
        frame,
        symbol=symbol,
        duration=duration,
        config=CombinationSearchConfig(),
    )
    save_cached_combination_ranking(report)
    return {**report, "learningRefreshSource": "rebuilt_cache"}


def _usable_cached_ranking(symbol: str, duration: str) -> dict[str, Any] | None:
    cached = get_cached_combination_ranking(symbol, duration)
    if cached is None:
        return None
    if not cache_is_usable(cached):
        return None
    return cached


def _settled_factor_combo_predictions(symbol: str, duration: str) -> list[dict[str, Any]]:
    conn = get_conn()
    try:
        rows = conn.execute(
            """
            SELECT open_time, direction, confidence, trade_quality_score,
                   actual_return, prediction_correct, high_winrate_rule, strategy_key
            FROM predictions
            WHERE (
                strategy_key IN ({placeholders})
                OR strategy_key LIKE ?
                OR strategy_key LIKE ?
            )
              AND symbol = ? AND duration = ?
              AND settled_at IS NOT NULL
            ORDER BY open_time
            """.format(placeholders=_strategy_placeholders()),
            (
                *_fixed_simulation_strategy_keys(duration),
                f"{BATCH_COMBO_KEY_PREFIX}%",
                f"{BATCH_HIGH_WINRATE_KEY_PREFIX}%",
                symbol.upper(),
                duration,
            ),
        ).fetchall()
    finally:
        conn.close()
    return [dict(row) for row in rows]


def _fixed_simulation_strategy_keys(duration: str) -> tuple[str, ...]:
    return (
        *factor_combo_simulation_strategy_keys(),
        *high_winrate_factor_combo_simulation_strategy_keys(),
        lstm_shadow_strategy_key(duration),
    )


def _strategy_placeholders() -> str:
    return ",".join("?" for _key in _fixed_simulation_strategy_keys_for_placeholders())


def _fixed_simulation_strategy_keys_for_placeholders() -> tuple[str, ...]:
    return (
        *factor_combo_simulation_strategy_keys(),
        *high_winrate_factor_combo_simulation_strategy_keys(),
        "factor_lstm_shadow_placeholder",
    )


def _validate_duration(duration: str) -> None:
    if duration not in SUPPORTED_RULE_DURATIONS:
        raise ValueError(f"unsupported duration: {duration}")


def _path_payload(path: Path) -> str:
    return str(path)
