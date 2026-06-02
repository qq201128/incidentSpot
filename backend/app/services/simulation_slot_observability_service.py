from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.services.factor_backtest_gate import backtest_gate_thresholds
from app.services.factor_candidate_signal_keys import is_factor_candidate_signal_key
from app.services.factor_combo_simulation_keys import is_batch_combo_simulation_strategy
from app.services.rule_config import SUPPORTED_RULE_DURATIONS
from app.services.simulation_slot_candidates import simulation_candidate_rows
from app.services.simulation_slot_runtime import attach_slot_and_runtime_state, runtime_statuses_for_slots


def simulation_slots_report(symbol: str, duration: str) -> dict[str, Any]:
    sym = symbol.strip().upper()
    if duration not in SUPPORTED_RULE_DURATIONS:
        raise ValueError(f"unsupported duration: {duration}")
    candidates = simulation_candidate_rows(sym, duration)
    attach_slot_and_runtime_state(candidates, sym, duration)
    return {
        "symbol": sym,
        "duration": duration,
        "updatedAt": _utc_now(),
        "thresholds": backtest_gate_thresholds(),
        "singleFactorSlots": _passed_count(candidates, "single_factor"),
        "comboFactorSlots": _passed_count(candidates, "factor_combo"),
        "enabledSlots": sum(1 for row in candidates if _slot_enabled(row)),
        "rejectedCount": sum(1 for row in candidates if row["gateStatus"] == "rejected"),
        "items": candidates,
    }


def simulation_statuses_for_slots(slots: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    reports = [
        simulation_slots_report(symbol, duration)
        for _key, symbol, duration in _dynamic_slot_keys(slots)
    ]
    runtime = runtime_statuses_for_slots(slots)
    candidates = _status_map([item for report in reports for item in report.get("items", [])])
    return {**runtime, **candidates}


def _dynamic_slot_keys(slots: list[dict[str, Any]]) -> set[tuple[str, str, str]]:
    return {
        (slot["strategyKey"], str(slot["symbol"]).upper(), str(slot["duration"]))
        for slot in slots
        if _is_dynamic_key(str(slot.get("strategyKey") or ""))
    }


def _status_map(items: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {
        (str(item["strategyKey"]), str(item["symbol"]).upper(), str(item["duration"])): item
        for item in items
        if item.get("strategyKey")
    }


def _passed_count(candidates: list[dict[str, Any]], candidate_type: str) -> int:
    return sum(1 for row in candidates if row["candidateType"] == candidate_type and row.get("gatePassed") is True)


def _slot_enabled(row: dict[str, Any]) -> bool:
    slot = row.get("slot")
    return isinstance(slot, dict) and slot.get("enabled") is True


def _is_dynamic_key(strategy_key: str) -> bool:
    return is_factor_candidate_signal_key(strategy_key) or is_batch_combo_simulation_strategy(strategy_key)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
