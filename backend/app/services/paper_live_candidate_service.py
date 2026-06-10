from __future__ import annotations

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
from app.services.paper_live_json_fields import parse_json_field
from app.services.paper_live_stage_log import ensure_stage_log_table, recent_stage_logs

STATUS_COLLECTING = "paper_collecting"
STATUS_STABLE = "paper_stable"
STATUS_FAILED = "paper_failed"
STATUS_BACKTEST = "backtest_candidate"
STATUS_LEAKAGE = "invalid_data_leakage"
OBSERVATION_POOL_LIMIT = 150
FAILED_LIST_LIMIT = 40


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
    conn = get_conn()
    try:
        report = _report_from_conn(conn, symbol, duration)
    finally:
        conn.close()
    return report


def _refresh_states(symbol: str, duration: str) -> dict[str, Any]:
    conn = get_conn()
    try:
        _ensure_report_write_tables(conn)
        report = _report_from_conn(conn, symbol, duration)
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


def _report_from_conn(conn: Any, symbol: str, duration: str) -> dict[str, Any]:
    rows = _candidate_rows(conn, symbol, duration)
    settled_by_identity = _settled_rows_by_identity(conn, symbol, duration)
    candidates = [_candidate_payload(item, settled_by_identity.get(_candidate_identity_key(item), [])) for item in rows]
    failures = recent_prediction_failures(conn, symbol, duration)
    stage_logs = recent_stage_logs(conn, symbol, duration)
    status_changes = recent_status_changes(conn, symbol, duration)
    ranked = sorted(candidates, key=candidate_rank_key, reverse=True)
    return _report_payload(ReportPayloadInput(symbol, duration, ranked, failures, stage_logs, status_changes))


def _candidate_rows(conn: Any, symbol: str, duration: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT signal_key, strategy_key, high_winrate_rule, high_winrate_gate_value,
               high_winrate_gate_min, model_family, model_version, validation_win_rate, feature_window,
               model_duration, model_trained_at, data_freshness_status,
               missing_feature_status, oos_win_rate, walk_forward_result,
               recent_rolling_result, MIN(created_at) AS first_created_at,
               MAX(created_at) AS latest_created_at, COUNT(*) AS prediction_count
        FROM predictions
        WHERE symbol = ? AND duration = ?
        GROUP BY signal_key, strategy_key, COALESCE(high_winrate_rule, model_version, signal_key)
        """,
        (symbol.strip().upper(), duration),
    ).fetchall()
    return [dict(row) for row in rows]


def _settled_rows_by_identity(conn: Any, symbol: str, duration: str) -> dict[tuple[str, str], list[dict[str, Any]]]:
    rows = conn.execute(
        """
        SELECT signal_key, COALESCE(high_winrate_rule, model_version, signal_key) AS lifecycle_identity,
               open_time, direction, entry_price, exit_price, actual_return,
               prediction_correct, high_winrate_rule, data_freshness_status,
               missing_feature_status, settled_at
        FROM predictions
        WHERE symbol = ? AND duration = ? AND settled_at IS NOT NULL
        ORDER BY open_time DESC
        """,
        (symbol.strip().upper(), duration),
    ).fetchall()
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        item = dict(row)
        key = (str(item.pop("signal_key")), str(item.pop("lifecycle_identity")))
        grouped.setdefault(key, []).append(item)
    return grouped


def _candidate_payload(candidate: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = high_winrate_metrics(rows) if rows else _empty_metrics()
    decision = _candidate_decision(candidate, metrics, rows)
    walk_forward = parse_json_field("walkForwardResult", candidate.get("walk_forward_result"))
    recent_rolling = parse_json_field("recentRollingResult", candidate.get("recent_rolling_result"))
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
        "liveReadiness": live_readiness_gate(metrics, decision["status"], status_reason=decision["reason"]),
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


def _report_payload(data: ReportPayloadInput) -> dict[str, Any]:
    focused = focus_pool(data.ranked, OBSERVATION_POOL_LIMIT)
    avoid_next_search = _avoid_next_search(data.ranked, data.failures)
    return {
        "version": "paper_live_candidate_pool_v1",
        "symbol": data.symbol.strip().upper(),
        "duration": data.duration,
        "updatedAt": _utc_now(),
        "realTradingEnabled": False,
        "observationPoolLimit": OBSERVATION_POOL_LIMIT,
        "allCandidateCount": len(data.ranked),
        "allCandidates": data.ranked,
        "candidates": focused,
        "rankingPolicy": _ranking_policy(),
        "collecting": [row for row in focused if row["status"] == STATUS_COLLECTING],
        "stable": [row for row in focused if row["status"] == STATUS_STABLE],
        "failed": [
            row
            for row in data.ranked
            if row["status"] in {STATUS_FAILED, STATUS_LEAKAGE}
        ][:FAILED_LIST_LIMIT],
        "predictionFailures": data.failures,
        "stageLogs": data.stage_logs,
        "statusChanges": data.status_changes,
        "avoidNextSearch": avoid_next_search,
        "answers": candidate_pool_answers(focused, data.ranked, data.failures, avoid_next_search),
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


def _empty_metrics() -> dict[str, Any]:
    return high_winrate_metrics([])


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
