from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.db.session import get_conn, run_db_write_with_retry
from app.services.high_winrate_strategy_metrics import high_winrate_decision, high_winrate_metrics
from app.services.live_readiness_gate import live_readiness_gate
from app.services.paper_live_failure_store import (
    ensure_prediction_failure_table,
    log_prediction_failure,
    recent_prediction_failures,
)
from app.services.paper_live_candidate_live_state import live_state_by_strategy
from app.services.paper_live_candidate_ranking import (
    ValidationMetadata,
    candidate_rank_key,
    focus_pool,
    performance_comparison,
)
from app.services.paper_live_candidate_answers import candidate_pool_answers
from app.services.paper_live_candidate_status_store import (
    ensure_candidate_status_tables,
    recent_status_changes,
    write_candidate_status,
)
from app.services.paper_live_json_fields import parse_details_json, parse_json_field
from app.services.paper_live_stage_log import ensure_stage_log_table, recent_stage_logs

STATUS_COLLECTING = "paper_collecting"
STATUS_STABLE = "paper_stable"
STATUS_FAILED = "paper_failed"
STATUS_BACKTEST = "backtest_candidate"
STATUS_LEAKAGE = "invalid_data_leakage"
OBSERVATION_POOL_LIMIT = 150
FAILED_LIST_LIMIT = 40
RECENT_SETTLED_ROW_LIMIT = 100
LIVE_READINESS_METRIC_KEYS = (
    "consecutiveLosses",
    "sampleCount",
    "winRate",
    "profitFactor",
    "avgReturn",
    "paperStability",
)


@dataclass(frozen=True)
class ReportPayloadInput:
    symbol: str
    duration: str
    ranked: list[dict[str, Any]]
    failures: list[dict[str, Any]]
    stage_logs: list[dict[str, Any]]
    status_changes: list[dict[str, Any]]


def refresh_paper_live_candidate_states(symbol: str, duration: str) -> dict[str, Any]:
    return run_db_write_with_retry(lambda: _refresh_states(symbol, duration))


def paper_live_candidate_report(symbol: str, duration: str) -> dict[str, Any]:
    return _candidate_report(symbol, duration, include_all_candidates=False)


def paper_live_candidate_full_report(symbol: str, duration: str) -> dict[str, Any]:
    return _candidate_report(symbol, duration, include_all_candidates=True)


def _candidate_report(symbol: str, duration: str, *, include_all_candidates: bool) -> dict[str, Any]:
    conn = get_conn()
    try:
        if not include_all_candidates:
            cached_report = _status_snapshot_report(conn, symbol, duration)
            if cached_report is not None:
                return cached_report
        report = _report_from_conn(conn, symbol, duration, include_all_candidates=include_all_candidates)
    finally:
        conn.close()
    return report


def _refresh_states(symbol: str, duration: str) -> dict[str, Any]:
    conn = get_conn()
    try:
        _ensure_report_write_tables(conn)
        report = _report_from_conn(conn, symbol, duration, include_all_candidates=True)
        for candidate in report["allCandidates"]:
            write_candidate_status(conn, symbol, duration, candidate=candidate)
        report["statusChanges"] = recent_status_changes(conn, symbol, duration)
        conn.commit()
        return report
    finally:
        conn.close()


def _ensure_report_write_tables(conn: Any) -> None:
    ensure_prediction_failure_table(conn)
    ensure_candidate_status_tables(conn)
    ensure_stage_log_table(conn)


def _report_from_conn(conn: Any, symbol: str, duration: str, *, include_all_candidates: bool) -> dict[str, Any]:
    rows = _candidate_rows(conn, symbol, duration)
    settled_by_identity = _settled_rows_by_identity(conn, symbol, duration)
    live_states = live_state_by_strategy(conn, symbol, duration)
    candidates = [
        _candidate_payload(
            item,
            settled_by_identity.get(_candidate_identity_key(item), []),
            live_states.get(str(item["strategy_key"])),
        )
        for item in rows
    ]
    failures = recent_prediction_failures(conn, symbol, duration)
    stage_logs = recent_stage_logs(conn, symbol, duration)
    status_changes = recent_status_changes(conn, symbol, duration)
    ranked = sorted(candidates, key=candidate_rank_key, reverse=True)
    return _report_payload(
        ReportPayloadInput(symbol, duration, ranked, failures, stage_logs, status_changes),
        include_all_candidates=include_all_candidates,
    )


def _status_snapshot_report(conn: Any, symbol: str, duration: str) -> dict[str, Any] | None:
    rows = _status_snapshot_rows(conn, symbol, duration)
    if not rows:
        return None
    live_states = live_state_by_strategy(conn, symbol, duration)
    candidates = [
        _candidate_from_status_snapshot(row, live_states)
        for row in rows
    ]
    ranked = sorted(candidates, key=candidate_rank_key, reverse=True)
    failures = recent_prediction_failures(conn, symbol, duration)
    stage_logs = recent_stage_logs(conn, symbol, duration)
    status_changes = recent_status_changes(conn, symbol, duration)
    report = _report_payload(
        ReportPayloadInput(symbol, duration, ranked, failures, stage_logs, status_changes),
        include_all_candidates=False,
    )
    return {
        **report,
        "payloadSource": "candidate_status_snapshot",
        "updatedAt": max(str(row.get("updated_at") or "") for row in rows) or report["updatedAt"],
    }


def _status_snapshot_rows(conn: Any, symbol: str, duration: str) -> list[dict[str, Any]]:
    try:
        rows = conn.execute(
            """
            SELECT candidate_key, status, reason, details_json, updated_at
            FROM paper_live_candidate_status
            WHERE symbol = ? AND duration = ?
            ORDER BY updated_at DESC
            """,
            (symbol.strip().upper(), duration),
        ).fetchall()
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc).lower():
            return []
        raise
    return [dict(row) for row in rows]


def _candidate_from_status_snapshot(
    row: dict[str, Any],
    live_states: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    parsed = parse_details_json(row.get("details_json"))
    candidate = dict(parsed.value) if isinstance(parsed.value, dict) else {}
    strategy_key = str(candidate.get("strategyKey") or "")
    candidate.setdefault("candidateKey", row.get("candidate_key"))
    candidate.setdefault("status", row.get("status"))
    candidate.setdefault("paperLiveStatus", row.get("status"))
    candidate.setdefault("reason", row.get("reason"))
    candidate.setdefault("metrics", {})
    live_state = live_states.get(strategy_key)
    if live_state is not None:
        candidate = {
            **candidate,
            "autoTradeEnabled": live_state["autoTradeEnabled"],
            "liveTradingEnabled": live_state["liveTradingEnabled"],
            "liveTradingUpdatedAt": live_state["updatedAt"],
        }
        metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
        if _has_live_readiness_metrics(metrics):
            candidate["liveReadiness"] = live_readiness_gate(
                metrics,
                candidate.get("status") or candidate.get("paperLiveStatus"),
                status_reason=candidate.get("reason"),
                real_trading_enabled=bool(live_state["liveTradingEnabled"]),
            )
    if parsed.error:
        candidate["metadataParseErrors"] = [parsed.error]
    return candidate


def _candidate_rows(conn: Any, symbol: str, duration: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT signal_key, strategy_key, high_winrate_rule, high_winrate_gate_value,
               high_winrate_gate_min, model_family, model_version, validation_win_rate, feature_window,
               model_duration, model_trained_at, data_freshness_status,
               missing_feature_status, oos_win_rate, walk_forward_result,
               recent_rolling_result, MIN(created_at) AS first_created_at,
               MAX(created_at) AS latest_created_at, COUNT(*) AS prediction_count,
               SUM(CASE WHEN settled_at IS NOT NULL THEN 1 ELSE 0 END) AS settled_sample_count,
               SUM(CASE WHEN settled_at IS NOT NULL AND prediction_correct THEN 1 ELSE 0 END) AS settled_win_count,
               SUM(CASE WHEN settled_at IS NOT NULL AND actual_return IS NOT NULL THEN 1 ELSE 0 END) AS settled_return_count,
               SUM(CASE WHEN settled_at IS NOT NULL AND actual_return > 0 THEN actual_return ELSE 0 END) AS settled_gain_sum,
               SUM(CASE WHEN settled_at IS NOT NULL AND actual_return < 0 THEN -actual_return ELSE 0 END) AS settled_loss_sum,
               SUM(CASE WHEN settled_at IS NOT NULL AND actual_return IS NOT NULL THEN actual_return ELSE 0 END) AS settled_return_sum,
               SUM(CASE WHEN settled_at IS NOT NULL AND data_freshness_status = ? THEN 1 ELSE 0 END) AS settled_leakage_count
        FROM predictions
        WHERE symbol = ? AND duration = ?
        GROUP BY signal_key, strategy_key, COALESCE(high_winrate_rule, model_version, signal_key)
        """,
        (STATUS_LEAKAGE, symbol.strip().upper(), duration),
    ).fetchall()
    return [dict(row) for row in rows]


def _settled_rows_by_identity(conn: Any, symbol: str, duration: str) -> dict[tuple[str, str], list[dict[str, Any]]]:
    rows = conn.execute(
        """
        WITH ranked AS (
            SELECT signal_key, COALESCE(high_winrate_rule, model_version, signal_key) AS lifecycle_identity,
                   open_time, direction, entry_price, exit_price, actual_return,
                   prediction_correct, high_winrate_rule, data_freshness_status,
                   missing_feature_status, settled_at,
                   ROW_NUMBER() OVER (
                       PARTITION BY signal_key, COALESCE(high_winrate_rule, model_version, signal_key)
                       ORDER BY open_time DESC
                   ) AS row_number
            FROM predictions
            WHERE symbol = ? AND duration = ? AND settled_at IS NOT NULL
        )
        SELECT signal_key, lifecycle_identity, open_time, direction, entry_price, exit_price,
               actual_return, prediction_correct, high_winrate_rule, data_freshness_status,
               missing_feature_status, settled_at
        FROM ranked
        WHERE row_number <= ?
        ORDER BY open_time DESC
        """,
        (symbol.strip().upper(), duration, RECENT_SETTLED_ROW_LIMIT),
    ).fetchall()
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        item = dict(row)
        key = (str(item.pop("signal_key")), str(item.pop("lifecycle_identity")))
        grouped.setdefault(key, []).append(item)
    return grouped


def _candidate_payload(
    candidate: dict[str, Any],
    rows: list[dict[str, Any]],
    live_state: dict[str, Any] | None,
) -> dict[str, Any]:
    metrics = _candidate_metrics(candidate, rows)
    decision = _candidate_decision(candidate, metrics, rows)
    walk_forward = parse_json_field("walkForwardResult", candidate.get("walk_forward_result"))
    recent_rolling = parse_json_field("recentRollingResult", candidate.get("recent_rolling_result"))
    live_enabled = bool(live_state and live_state["liveTradingEnabled"])
    payload = {
        "candidateKey": _candidate_key(candidate),
        "strategyKey": candidate["strategy_key"],
        "candidateType": _candidate_type(candidate),
        "factorName": candidate.get("high_winrate_rule"),
        "modelFamily": candidate.get("model_family"),
        "modelVersion": candidate.get("model_version"),
        "featureWindow": candidate.get("feature_window"),
        "minConfidence": candidate.get("high_winrate_gate_min"),
        "validationWinRate": candidate.get("validation_win_rate"),
        "backtestWinRate": candidate.get("high_winrate_gate_value"),
        "oosWinRate": candidate.get("oos_win_rate"),
        "walkForwardResult": walk_forward.value,
        "recentRollingResult": recent_rolling.value,
        "paperLiveWinRate": metrics.get("winRate"),
        "paperLiveSampleCount": metrics.get("sampleCount"),
        "paperLiveStatus": decision["status"],
        "autoTradeEnabled": bool(live_state and live_state["autoTradeEnabled"]),
        "liveTradingEnabled": live_enabled,
        "liveTradingUpdatedAt": live_state.get("updatedAt") if live_state else None,
        "liveReadiness": live_readiness_gate(metrics, decision["status"], status_reason=decision["reason"], real_trading_enabled=live_enabled),
        "status": decision["status"],
        "reason": decision["reason"],
        "metrics": metrics,
        "dataFreshnessStatus": _latest_status(candidate, rows, "data_freshness_status"),
        "missingFeatureStatus": _latest_status(candidate, rows, "missing_feature_status"),
        "predictionCreatedAt": candidate.get("latest_created_at"),
        "firstPredictionCreatedAt": candidate.get("first_created_at"),
        "predictionCount": candidate.get("prediction_count"),
    }
    payload["performanceComparison"] = {
        **performance_comparison(
            candidate,
            metrics,
            ValidationMetadata(walk_forward.value, recent_rolling.value),
        ),
        "paperLiveStatus": decision["status"],
    }
    parse_errors = [item.error for item in (walk_forward, recent_rolling) if item.error]
    if parse_errors:
        payload["metadataParseErrors"] = parse_errors
    return payload


def _candidate_decision(
    candidate: dict[str, Any],
    metrics: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, str]:
    leakage = _data_leakage_reason(candidate, rows)
    if leakage:
        return {"status": STATUS_LEAKAGE, "reason": leakage}
    return high_winrate_decision(metrics)


def _report_payload(data: ReportPayloadInput, *, include_all_candidates: bool) -> dict[str, Any]:
    focused = focus_pool(data.ranked, OBSERVATION_POOL_LIMIT)
    failed = [
        row
        for row in data.ranked
        if row["status"] in {STATUS_FAILED, STATUS_LEAKAGE}
    ][:FAILED_LIST_LIMIT]
    avoid_next_search = _avoid_next_search(data.ranked, data.failures)
    all_candidates = data.ranked if include_all_candidates else _dashboard_all_candidates(data.ranked, focused, failed)
    candidates = focused if include_all_candidates else _dashboard_candidate_rows(focused)
    failed_rows = failed if include_all_candidates else _dashboard_candidate_rows(failed)
    stable_rows = [row for row in candidates if row["status"] == STATUS_STABLE]
    collecting_rows = [row for row in candidates if row["status"] == STATUS_COLLECTING]
    return {
        "version": "paper_live_candidate_pool_v1",
        "payloadMode": "full" if include_all_candidates else "dashboard_slim",
        "symbol": data.symbol.strip().upper(),
        "duration": data.duration,
        "updatedAt": _utc_now(),
        "realTradingEnabled": any(bool(row.get("liveTradingEnabled")) for row in data.ranked),
        "observationPoolLimit": OBSERVATION_POOL_LIMIT,
        "allCandidateCount": len(data.ranked),
        "settledCandidateCount": _settled_candidate_count(data.ranked),
        "candidateModelFamilies": _candidate_model_families(data.ranked),
        "allCandidates": all_candidates if include_all_candidates else _dashboard_candidate_rows(all_candidates),
        "candidates": candidates,
        "rankingPolicy": _ranking_policy(),
        "collecting": collecting_rows,
        "stable": stable_rows,
        "failed": failed_rows,
        "predictionFailures": data.failures if include_all_candidates else _dashboard_failures(data.failures),
        "stageLogs": data.stage_logs if include_all_candidates else _dashboard_stage_logs(data.stage_logs),
        "statusChanges": data.status_changes if include_all_candidates else _dashboard_status_changes(data.status_changes),
        "avoidNextSearch": avoid_next_search,
        "answers": candidate_pool_answers(focused, data.ranked, data.failures, avoid_next_search) if include_all_candidates else {},
    }


def _ranking_policy() -> list[str]:
    return [
        "paper-live lifecycle stability",
        "OOS performance",
        "walk-forward performance",
        "recent rolling stability",
        "profitFactor",
        "avgReturn",
        "settled sample count",
    ]


def _dashboard_all_candidates(
    ranked: list[dict[str, Any]],
    focused: list[dict[str, Any]],
    failed: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    live_rows = [row for row in ranked if row.get("liveTradingEnabled")]
    return _dedupe_candidates([*live_rows, *focused, *failed])


def _dashboard_candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_dashboard_candidate_row(row) for row in rows]


def _dashboard_candidate_row(row: dict[str, Any]) -> dict[str, Any]:
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    return {
        "candidateKey": row.get("candidateKey"),
        "strategyKey": row.get("strategyKey"),
        "candidateType": row.get("candidateType"),
        "factorName": row.get("factorName"),
        "modelFamily": row.get("modelFamily"),
        "modelVersion": row.get("modelVersion"),
        "featureWindow": row.get("featureWindow"),
        "minConfidence": row.get("minConfidence"),
        "validationWinRate": row.get("validationWinRate"),
        "backtestWinRate": row.get("backtestWinRate"),
        "oosWinRate": row.get("oosWinRate"),
        "paperLiveWinRate": row.get("paperLiveWinRate"),
        "paperLiveSampleCount": row.get("paperLiveSampleCount"),
        "paperLiveStatus": row.get("paperLiveStatus"),
        "autoTradeEnabled": row.get("autoTradeEnabled"),
        "liveTradingEnabled": row.get("liveTradingEnabled"),
        "liveTradingUpdatedAt": row.get("liveTradingUpdatedAt"),
        "status": row.get("status"),
        "reason": row.get("reason"),
        "metrics": {
            "sampleCount": metrics.get("sampleCount"),
            "winRate": metrics.get("winRate"),
            "profitFactor": metrics.get("profitFactor"),
            "avgReturn": metrics.get("avgReturn"),
            "maxConsecutiveLosses": metrics.get("maxConsecutiveLosses"),
            "paperStability": metrics.get("paperStability"),
            "paperLiveWindows": metrics.get("paperLiveWindows"),
        },
        "dataFreshnessStatus": row.get("dataFreshnessStatus"),
        "missingFeatureStatus": row.get("missingFeatureStatus"),
        "predictionCreatedAt": row.get("predictionCreatedAt"),
        "firstPredictionCreatedAt": row.get("firstPredictionCreatedAt"),
        "predictionCount": row.get("predictionCount"),
    }


def _dashboard_failures(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "candidateKey": row.get("candidateKey"),
            "strategyKey": row.get("strategyKey"),
            "stage": row.get("stage"),
            "reason": row.get("reason"),
            "createdAt": row.get("createdAt"),
        }
        for row in rows
    ]


def _dashboard_stage_logs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "signalKey": row.get("signalKey"),
            "strategyKey": row.get("strategyKey"),
            "entryOpenTime": row.get("entryOpenTime"),
            "stage": row.get("stage"),
            "status": row.get("status"),
            "reason": row.get("reason"),
            "createdAt": row.get("createdAt"),
        }
        for row in rows
    ]


def _dashboard_status_changes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "candidateKey": row.get("candidateKey"),
            "oldStatus": row.get("oldStatus"),
            "newStatus": row.get("newStatus"),
            "reason": row.get("reason"),
            "changedAt": row.get("changedAt"),
        }
        for row in rows
    ]


def _dedupe_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for row in rows:
        key = str(row.get("candidateKey") or row.get("strategyKey") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def _has_live_readiness_metrics(metrics: dict[str, Any]) -> bool:
    return all(key in metrics for key in LIVE_READINESS_METRIC_KEYS)


def _settled_candidate_count(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if int(row.get("paperLiveSampleCount") or row.get("metrics", {}).get("sampleCount") or 0) > 0)


def _candidate_model_families(rows: list[dict[str, Any]]) -> list[str]:
    families = {
        str(row["modelFamily"])
        for row in rows
        if row.get("candidateType") == "model" and row.get("modelFamily")
    }
    return sorted(families)


def _empty_metrics() -> dict[str, Any]:
    return high_winrate_metrics([])


def _candidate_metrics(candidate: dict[str, Any], recent_rows: list[dict[str, Any]]) -> dict[str, Any]:
    sample_count = int(candidate.get("settled_sample_count") or 0)
    if sample_count <= 0:
        return _empty_metrics()
    recent_metrics = high_winrate_metrics(recent_rows)
    return {
        **recent_metrics,
        "sampleCount": sample_count,
        "winRate": _ratio(int(candidate.get("settled_win_count") or 0), sample_count),
        "profitFactor": _profit_factor(
            gain_sum=float(candidate.get("settled_gain_sum") or 0.0),
            loss_sum=float(candidate.get("settled_loss_sum") or 0.0),
            return_count=int(candidate.get("settled_return_count") or 0),
        ),
        "avgReturn": _avg_return(
            return_sum=float(candidate.get("settled_return_sum") or 0.0),
            return_count=int(candidate.get("settled_return_count") or 0),
        ),
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator <= 0 else round(numerator / denominator, 4)


def _profit_factor(*, gain_sum: float, loss_sum: float, return_count: int) -> float | None:
    if return_count <= 0 or loss_sum == 0:
        return None
    return round(gain_sum / loss_sum, 4)


def _avg_return(*, return_sum: float, return_count: int) -> float | None:
    if return_count <= 0:
        return None
    return round(return_sum / return_count, 8)


def _candidate_type(candidate: dict[str, Any]) -> str:
    family = candidate.get("model_family")
    if family and family not in {"factor", "factor_combo"}:
        return "model"
    return "factor_combo" if family == "factor_combo" else "factor"


def _identity_value(candidate: dict[str, Any]) -> str:
    return str(
        candidate.get("high_winrate_rule")
        or candidate.get("model_version")
        or candidate["signal_key"]
    )


def _candidate_identity_key(candidate: dict[str, Any]) -> tuple[str, str]:
    return str(candidate["signal_key"]), _identity_value(candidate)


def _candidate_key(candidate: dict[str, Any]) -> str:
    if candidate.get("model_family") and candidate.get("model_version"):
        return str(candidate["model_version"])
    return str(candidate["signal_key"])


def _latest_status(candidate: dict[str, Any], rows: list[dict[str, Any]], key: str) -> str | None:
    if rows and rows[0].get(key):
        return str(rows[0][key])
    value = candidate.get(key)
    return None if value is None else str(value)


def _data_leakage_reason(candidate: dict[str, Any], rows: list[dict[str, Any]]) -> str | None:
    if candidate.get("data_freshness_status") == STATUS_LEAKAGE:
        return STATUS_LEAKAGE
    if int(candidate.get("settled_leakage_count") or 0) > 0:
        return STATUS_LEAKAGE
    for row in rows:
        if row.get("data_freshness_status") == STATUS_LEAKAGE:
            return STATUS_LEAKAGE
    return None


def _avoid_next_search(
    candidates: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    failed = [row for row in candidates if row["status"] in {STATUS_FAILED, STATUS_LEAKAGE}]
    avoid = [{"candidateKey": row["candidateKey"], "reason": row["reason"]} for row in failed[:20]]
    avoid.extend({"candidateKey": row["candidateKey"], "reason": row["reason"]} for row in failures[:20])
    return avoid[:20]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
