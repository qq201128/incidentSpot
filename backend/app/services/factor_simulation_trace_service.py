from __future__ import annotations

from typing import Any

from app.db.session import get_conn
from app.services.batch_combo_event_demotion import evaluate_batch_combo_event_demotion
from app.services.factor_backtest_gate import backtest_gate_thresholds
from app.services.factor_cache_metadata import cache_is_usable
from app.services.factor_candidate_event_demotion import evaluate_factor_candidate_event_demotion
from app.services.factor_candidate_signal_keys import (
    factor_candidate_signal_key,
    is_factor_candidate_signal_key,
)
from app.services.factor_catalog import get_factor_payload_by_name
from app.services.factor_combination_cache_service import get_cached_combination_ranking
from app.services.factor_combo_simulation_keys import (
    is_batch_combo_simulation_strategy,
    simulation_strategy_key_for_factor_name,
)
from app.services.factor_ranking_cache_service import get_cached_ranking
from app.services.high_winrate_combo_cache_service import get_cached_high_winrate_combo_ranking
from app.services.rule_config import SUPPORTED_RULE_DURATIONS
from app.services.simulation_slot_observability_service import simulation_slots_report


def factor_simulation_trace(symbol: str, duration: str, *, factor_name: str | None, strategy_key: str | None) -> dict:
    sym = symbol.strip().upper()
    dur = str(duration)
    if dur not in SUPPORTED_RULE_DURATIONS:
        raise ValueError(f"unsupported duration: {dur}")
    identity = _identity(factor_name, strategy_key)
    slot_report = simulation_slots_report(sym, dur)
    slot = _matching_slot(slot_report.get("items") or [], identity)
    factor = _factor_name(identity, slot)
    kind = _candidate_type(identity, slot, factor)
    strategy = _strategy_key(identity, slot, factor, kind)
    ranking = _ranking_snapshot(sym, dur, factor, kind)
    prediction = _latest_prediction(sym, dur, strategy)
    observation = _observation(sym, dur, strategy, kind)
    issues = _issues(factor, ranking, slot, prediction, observation)
    found = not _not_found(factor, ranking, slot)
    return {
        "symbol": sym,
        "duration": dur,
        "kind": kind,
        "factorName": factor,
        "strategyKey": strategy,
        "found": found,
        "status": "not_found" if not found else _status(issues, slot),
        "issues": issues,
        "source": _source_payload(slot, ranking),
        "ranking": ranking,
        "gate": _gate_payload(slot),
        "simulationSlot": _slot_payload(slot),
        "latestPrediction": prediction,
        "latestEvent": None if slot is None else slot.get("latestEvent"),
        "order": _order_payload(None if slot is None else slot.get("latestEvent")),
        "settlement": _settlement_payload(None if slot is None else slot.get("latestEvent")),
        "observation": observation,
        "thresholds": slot_report.get("thresholds") or backtest_gate_thresholds(),
        "updatedAt": slot_report.get("updatedAt"),
    }


def _identity(factor_name: str | None, strategy_key: str | None) -> dict[str, str | None]:
    factor = factor_name.strip() if factor_name else None
    strategy = strategy_key.strip() if strategy_key else None
    if not factor and not strategy:
        raise ValueError("factorName or strategyKey is required")
    return {"factorName": factor or None, "strategyKey": strategy or None}


def _matching_slot(items: list[dict[str, Any]], identity: dict[str, str | None]) -> dict[str, Any] | None:
    for item in items:
        if identity["strategyKey"] and item.get("strategyKey") == identity["strategyKey"]:
            return item
        if identity["factorName"] and item.get("factorName") == identity["factorName"]:
            return item
    return None


def _factor_name(identity: dict[str, str | None], slot: dict[str, Any] | None) -> str | None:
    if identity["factorName"]:
        return identity["factorName"]
    value = None if slot is None else slot.get("factorName")
    return str(value) if value else None


def _candidate_type(identity: dict[str, str | None], slot: dict[str, Any] | None, factor: str | None) -> str:
    if slot and slot.get("candidateType"):
        return str(slot["candidateType"])
    if factor and _is_combo_name(factor):
        return "factor_combo"
    if is_batch_combo_simulation_strategy(identity["strategyKey"]):
        return "factor_combo"
    return "single_factor"


def _strategy_key(identity: dict[str, str | None], slot: dict[str, Any] | None, factor: str | None, kind: str) -> str | None:
    if identity["strategyKey"]:
        return identity["strategyKey"]
    if slot and slot.get("strategyKey"):
        return str(slot["strategyKey"])
    if not factor:
        return None
    return simulation_strategy_key_for_factor_name(factor) if kind == "factor_combo" else factor_candidate_signal_key(factor)


def _ranking_snapshot(symbol: str, duration: str, factor: str | None, kind: str) -> dict[str, Any]:
    if not factor:
        return _missing_ranking(kind, "factor_name_unavailable")
    if kind == "factor_combo":
        return _combo_ranking_snapshot(symbol, duration, factor)
    return _single_ranking_snapshot(symbol, duration, factor)


def _single_ranking_snapshot(symbol: str, duration: str, factor: str) -> dict[str, Any]:
    cache = get_cached_ranking(symbol, duration)
    row, rank = _ranked_row(cache, factor)
    if row is None:
        return _ranking_status("factor_ranking_cache", cache, "factor_not_in_cache")
    return _ranking_payload("factor_ranking_cache", cache, row, rank)


def _combo_ranking_snapshot(symbol: str, duration: str, factor: str) -> dict[str, Any]:
    for source, cache in _combo_caches(symbol, duration):
        row, rank = _ranked_row(cache, factor)
        if row is not None:
            return _ranking_payload(source, cache, row, rank)
    cache = get_cached_combination_ranking(symbol, duration)
    return _ranking_status("factor_combo_ranking_cache", cache, "factor_not_in_cache")


def _combo_caches(symbol: str, duration: str) -> tuple[tuple[str, dict[str, Any] | None], ...]:
    return (
        ("factor_combo_ranking_cache", get_cached_combination_ranking(symbol, duration)),
        ("high_winrate_combo_ranking_cache", get_cached_high_winrate_combo_ranking(symbol, duration)),
    )


def _ranked_row(cache: dict[str, Any] | None, factor: str) -> tuple[dict[str, Any] | None, int | None]:
    for index, row in enumerate((cache or {}).get("ranking") or [], start=1):
        if isinstance(row, dict) and (row.get("factorName") == factor or row.get("name") == factor):
            return dict(row), index
    return None, None


def _ranking_payload(source: str, cache: dict[str, Any] | None, row: dict[str, Any], rank: int | None) -> dict[str, Any]:
    return {"source": source, "status": "available", "rank": rank, "metrics": _metrics(row), "row": row, "cacheStatus": (cache or {}).get("cacheStatus"), "updatedAt": (cache or {}).get("updatedAt")}


def _ranking_status(source: str, cache: dict[str, Any] | None, reason: str) -> dict[str, Any]:
    if cache is None:
        return _missing_ranking(source, "cache_missing")
    if not cache_is_usable(cache):
        return _missing_ranking(source, f"cache_unusable:{(cache.get('cacheStatus') or {}).get('reason')}")
    return _missing_ranking(source, reason)


def _missing_ranking(source: str, reason: str) -> dict[str, Any]:
    return {"source": source, "status": "unavailable", "reason": reason, "rank": None, "metrics": {}}


def _metrics(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "winRate": row.get("winRate") if row.get("winRate") is not None else row.get("backtestWinRate"),
        "profitFactor": row.get("profitFactor"),
        "totalPeriods": row.get("totalPeriods") or row.get("trades"),
        "factorScore": row.get("factorScore") or row.get("score"),
        "oosWinRate": _walk_forward(row).get("oosWinRate") or row.get("oosWinRate"),
    }


def _latest_prediction(symbol: str, duration: str, strategy_key: str | None) -> dict[str, Any] | None:
    if not strategy_key:
        return None
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT * FROM predictions WHERE strategy_key = ? AND symbol = ? AND duration = ? ORDER BY id DESC LIMIT 1",
            (strategy_key, symbol, duration),
        ).fetchone()
        return None if row is None else _prediction_payload(dict(row))
    finally:
        conn.close()


def _prediction_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "strategyKey": row["strategy_key"],
        "openTime": row["open_time"],
        "direction": row["direction"],
        "probabilityUp": row["probability_up"],
        "confidence": row["confidence"],
        "tradeQualityPassed": _as_bool(row["trade_quality_passed"]),
        "highWinrateRule": row["high_winrate_rule"],
        "createdAt": row["created_at"],
        "settledAt": row["settled_at"],
        "actualReturn": row["actual_return"],
        "predictionCorrect": _as_bool(row["prediction_correct"]),
    }


def _observation(symbol: str, duration: str, strategy_key: str | None, kind: str) -> dict[str, Any]:
    if not strategy_key:
        return {"status": "not_observed", "reason": "strategy_key_unavailable", "metrics": {}}
    section = _demotion_section(symbol, duration, kind)
    row = next((item for item in section.get("evaluations", []) if item.get("strategyKey") == strategy_key), None)
    if row is None:
        return {"status": "not_observed", "reason": "no_simulation_events", "metrics": {"sampleCount": 0}}
    return {"status": row.get("status"), "reason": row.get("reason"), "metrics": row.get("metrics") or {}, "row": row}


def _demotion_section(symbol: str, duration: str, kind: str) -> dict[str, Any]:
    if kind == "factor_combo":
        return evaluate_batch_combo_event_demotion(symbol, duration)
    return evaluate_factor_candidate_event_demotion(symbol, duration)


def _issues(factor: str | None, ranking: dict[str, Any], slot: dict[str, Any] | None, prediction: Any, observation: dict) -> list[str]:
    issues = []
    if factor is None:
        issues.append("factor_not_resolved_from_input")
    if ranking.get("status") != "available":
        issues.append(str(ranking.get("reason") or "ranking_unavailable"))
    if slot is None:
        issues.append("simulation_slot_not_synced")
    elif slot.get("latestEvent") is None:
        issues.append("no_simulation_events")
    elif slot["latestEvent"].get("settlementPnl") is None:
        issues.append("latest_event_not_settled")
    if prediction is None:
        issues.append("no_prediction_records")
    if observation.get("status") == "demoted":
        issues.append("demotion_watchlist")
    return issues


def _status(issues: list[str], slot: dict[str, Any] | None) -> str:
    if "factor_not_resolved_from_input" in issues or "factor_not_in_cache" in issues:
        return "not_found"
    if slot and slot.get("gateStatus") == "rejected":
        return "gate_rejected"
    if "demotion_watchlist" in issues:
        return "watchlist"
    if slot and slot.get("gateStatus") == "enabled":
        return "simulation_enabled"
    return "not_enabled"


def _not_found(factor: str | None, ranking: dict[str, Any], slot: dict[str, Any] | None) -> bool:
    exists = factor and (get_factor_payload_by_name(factor) is not None or ranking.get("status") == "available" or slot)
    return not bool(exists)


def _source_payload(slot: dict[str, Any] | None, ranking: dict[str, Any]) -> dict[str, Any]:
    return {"candidateSource": None if slot is None else slot.get("source"), "rankingSource": ranking.get("source")}


def _gate_payload(slot: dict[str, Any] | None) -> dict[str, Any]:
    if slot is None:
        return {"status": "unknown", "reason": "simulation_slot_not_synced"}
    return {"status": slot.get("gateStatus"), "passed": slot.get("gatePassed"), "reason": slot.get("rejectionReason"), "metrics": slot.get("metrics") or {}}


def _slot_payload(slot: dict[str, Any] | None) -> dict[str, Any]:
    if slot is None:
        return {"synced": False, "enabled": False, "reason": "simulation_slot_not_synced"}
    payload = slot.get("slot") if isinstance(slot.get("slot"), dict) else {}
    return {"synced": True, "enabled": bool(payload.get("enabled")), "qty": payload.get("qty"), "updatedAt": payload.get("updatedAt")}


def _order_payload(event: dict[str, Any] | None) -> dict[str, Any]:
    if not event:
        return {"status": "none", "reason": "no_simulation_events"}
    return {"id": event.get("orderId"), "status": event.get("orderStatus"), "externalStatus": event.get("externalStatus"), "qty": event.get("qty")}


def _settlement_payload(event: dict[str, Any] | None) -> dict[str, Any]:
    if not event:
        return {"status": "none", "reason": "no_simulation_events", "pnl": None}
    if event.get("settlementPnl") is None:
        return {"status": "unsettled", "reason": "latest_event_not_settled", "pnl": None}
    return {"status": "settled", "pnl": event.get("settlementPnl"), "settledAt": event.get("settledAt")}


def _walk_forward(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("walkForward") if isinstance(row.get("walkForward"), dict) else {}


def _is_combo_name(factor_name: str) -> bool:
    return factor_name.startswith("combo__") or factor_name.startswith("goal_combo__")


def _as_bool(value: Any) -> bool | None:
    return None if value is None else bool(value)
