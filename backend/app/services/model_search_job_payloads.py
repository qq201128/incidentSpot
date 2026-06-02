from __future__ import annotations

import hashlib
import uuid
from typing import Any

from app.services.experiment_profiles import normalize_experiment_profile
from app.services.model_family_config import normalize_model_family
from app.services.model_family_search_rules import model_family_training_rules
from app.services.model_search_job_types import (
    JOB_ID_NAMESPACE,
    JOB_STAGE_PAPER_LIVE,
    JOB_STAGE_QUEUED,
    JOB_STAGE_SKIPPED,
    JOB_STAGE_WALK_FORWARD,
    JOB_STATUS_PENDING,
    JOB_STATUS_FAILED,
    JOB_STATUS_REJECTED,
    JOB_STATUS_SKIPPED,
    JOB_STATUS_SUCCEEDED,
    json_dumps,
    json_loads,
    utc_now,
)


def job_spec(
    *,
    symbol: str,
    duration: str,
    family: str,
    profile: str,
    priority: int,
    resource: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selected = normalize_model_family(family)
    params = {"target": "model_family_candidate_search", "trainingRules": model_family_training_rules(selected)}
    if resource:
        params["resource"] = _job_resource_spec(resource)
    params_json = json_dumps(params)
    params_hash = hashlib.sha256(params_json.encode("utf-8")).hexdigest()
    selected_profile = normalize_experiment_profile(profile)
    return {
        "symbol": symbol.strip().upper(),
        "duration": duration,
        "model_family": selected,
        "profile": selected_profile,
        "priority": int(priority),
        "params_json": params_json,
        "params_hash": params_hash,
        "job_id": job_id(
            symbol=symbol,
            duration=duration,
            family=selected,
            profile=selected_profile,
            params_hash=params_hash,
        ),
    }


def _job_resource_spec(resource: dict[str, Any]) -> dict[str, Any]:
    return {
        key: resource[key]
        for key in ("resourceProfile", "internalThreads", "parallelWorkers", "xgboostProcessWorkers")
        if key in resource
    }


def job_id(*, symbol: str, duration: str, family: str, profile: str, params_hash: str) -> str:
    key = "|".join((symbol.strip().upper(), duration, family, normalize_experiment_profile(profile), params_hash))
    return str(uuid.uuid5(uuid.UUID(JOB_ID_NAMESPACE), key))


def enqueue_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "version": "model_search_jobs_v1",
        "realTradingEnabled": False,
        "total": len(rows),
        "created": sum(1 for row in rows if row["enqueueAction"] == "created"),
        "existing": sum(1 for row in rows if row["enqueueAction"] == "existing"),
        "reset": sum(1 for row in rows if row["enqueueAction"] == "reset"),
        "jobs": rows,
    }


def classified_result(result: dict[str, Any]) -> dict[str, str | None]:
    status = str(result.get("status") or "")
    if status == "skipped":
        reason = str(result.get("reason") or "skipped")
        return {"status": JOB_STATUS_SKIPPED, "stage": JOB_STAGE_SKIPPED, "rejection": reason}
    if status in {"trade_active", "shadow_active", "initial_baseline", "trained"}:
        return {"status": JOB_STATUS_SUCCEEDED, "stage": JOB_STAGE_PAPER_LIVE, "rejection": None}
    reason = str(result.get("reason") or result.get("validationFailureReason") or f"offline_gate_rejected:{status}")
    return {"status": JOB_STATUS_REJECTED, "stage": JOB_STAGE_WALK_FORWARD, "rejection": reason}


def decode_job(row: Any) -> dict[str, Any]:
    payload = dict(row)
    for source, target in _JSON_FIELDS:
        payload[target] = json_loads(payload.get(source))
    payload["resetHistory"] = bool(payload.get("reset_history"))
    return payload


def insert_sql() -> str:
    return """
        INSERT INTO model_search_jobs(
          job_id, symbol, duration, model_family, profile, stage, params_json,
          params_hash, status, priority, created_at, resource_profile,
          internal_threads, parallel_workers, xgboost_process_workers,
          reset_history
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """


def insert_values(
    spec: dict[str, Any],
    resource: dict[str, Any] | None = None,
    reset_history: bool = False,
) -> tuple[Any, ...]:
    selected = resource or {}
    return (
        spec["job_id"], spec["symbol"], spec["duration"], spec["model_family"],
        spec["profile"], JOB_STAGE_QUEUED, spec["params_json"], spec["params_hash"],
        JOB_STATUS_PENDING, spec["priority"], utc_now(), selected.get("resourceProfile"),
        selected.get("internalThreads"), selected.get("parallelWorkers"),
        selected.get("xgboostProcessWorkers"), 1 if reset_history else 0,
    )


def finish_values(*, job_id: str, final: dict[str, str | None], result: dict[str, Any], resource: dict[str, Any], artifact_path: str | None, log_path: str | None) -> tuple[Any, ...]:
    now = utc_now()
    return (
        final["status"], final["stage"], now, now, artifact_path,
        json_dumps({"result": result, "resource": resource}), json_dumps(result),
        final["rejection"], log_path, resource.get("resourceProfile"),
        resource.get("internalThreads"), resource.get("parallelWorkers"),
        resource.get("xgboostProcessWorkers"), job_id,
    )


def failure_values(*, job_id: str, failure_type: str, failure_reason: str, failure_context: dict[str, Any], resource: dict[str, Any], log_path: str | None) -> tuple[Any, ...]:
    now = utc_now()
    return (
        JOB_STATUS_REJECTED if failure_type == "offline_rejected" else JOB_STATUS_FAILED,
        now, now, failure_type, failure_reason, json_dumps(failure_context),
        log_path, resource.get("resourceProfile"), resource.get("internalThreads"),
        resource.get("parallelWorkers"), resource.get("xgboostProcessWorkers"), job_id,
    )


def select_sql(filters: dict[str, Any]) -> tuple[str, tuple[Any, ...]]:
    clauses: list[str] = []
    values: list[Any] = []
    for key, column in _FILTER_COLUMNS:
        selected = tuple(filters.get(key) or ())
        if selected:
            clauses.append(f"{column} IN ({','.join('?' for _ in selected)})")
            values.extend(selected)
    where = "" if not clauses else "WHERE " + " AND ".join(clauses)
    return f"SELECT * FROM model_search_jobs {where} ORDER BY created_at DESC", tuple(values)


_JSON_FIELDS = (
    ("params_json", "params"),
    ("metrics_json", "metrics"),
    ("training_report_json", "trainingReport"),
    ("failure_context_json", "failureContext"),
)
_FILTER_COLUMNS = (
    ("symbols", "symbol"),
    ("durations", "duration"),
    ("families", "model_family"),
    ("statuses", "status"),
)
