from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from app.services.model_search_job_payloads import decode_job, insert_sql, insert_values
from app.services.model_search_job_sql import (
    claim_sql,
    existing_select_sql,
    pending_resource_update_sql,
    pending_select_sql,
    reset_sql,
    stale_select_sql,
    stale_update_sql,
)
from app.services.model_search_job_types import (
    JOB_STAGE_COARSE,
    JOB_STAGE_QUEUED,
    JOB_STATUS_FAILED,
    JOB_STATUS_PENDING,
    JOB_STATUS_RUNNING,
    json_dumps,
    utc_now,
)


@dataclass(frozen=True)
class EnqueueJobOptions:
    reset_existing: bool
    reset_history: bool
    resource: dict[str, Any] | None


def update_pending_resource(conn: Any, spec: dict[str, Any], resource: dict[str, Any]) -> bool:
    selected = resource or {}
    cursor = conn.execute(
        pending_resource_update_sql(),
        (
            selected.get("resourceProfile"),
            selected.get("internalThreads"),
            selected.get("parallelWorkers"),
            selected.get("xgboostProcessWorkers"),
            JOB_STATUS_PENDING,
            spec["symbol"],
            spec["duration"],
            spec["model_family"],
            spec["profile"],
        ),
    )
    return int(cursor.rowcount or 0) > 0


def enqueue_one(conn: Any, spec: dict[str, Any], options: EnqueueJobOptions) -> dict[str, Any]:
    existing = _find_existing(conn, spec)
    if existing is None:
        conn.execute(insert_sql(), insert_values(spec, options.resource, options.reset_history))
        return {**decode_job(required_job(conn, spec["job_id"])), "enqueueAction": "created"}
    if options.reset_existing:
        _reset_existing_job(conn, existing["job_id"], options)
        return {**decode_job(required_job(conn, existing["job_id"])), "enqueueAction": "reset"}
    return {**decode_job(existing), "enqueueAction": "existing"}


def mark_stale_running_jobs(conn: Any, stale_after_seconds: int) -> None:
    cutoff = (datetime.now(timezone.utc) - timedelta(seconds=max(stale_after_seconds, 1))).isoformat()
    rows = conn.execute(stale_select_sql(), (JOB_STATUS_RUNNING, cutoff)).fetchall()
    for row in rows:
        context = {"stage": row["stage"], "heartbeatAt": row["heartbeat_at"], "startedAt": row["started_at"]}
        conn.execute(stale_update_sql(), (
            JOB_STATUS_FAILED, utc_now(), "running_timeout",
            "running job heartbeat timed out", json_dumps(context), row["job_id"],
        ))


def claim_available_row(conn: Any, max_running_jobs: int, filters: dict[str, Any] | None = None) -> dict[str, Any] | None:
    while _running_count(conn) < max_running_jobs:
        row = _next_pending_row(conn, filters or {})
        if row is None:
            return None
        job = _claim_row(conn, row["job_id"])
        if job is not None:
            return job
    return None


def required_job(conn: Any, job_id: str) -> Any:
    row = conn.execute("SELECT * FROM model_search_jobs WHERE job_id = ?", (job_id,)).fetchone()
    if row is None:
        raise ValueError(f"model search job not found: {job_id}")
    return row


def _running_count(conn: Any) -> int:
    row = conn.execute("SELECT COUNT(*) AS c FROM model_search_jobs WHERE status = ?", (JOB_STATUS_RUNNING,)).fetchone()
    return int(row["c"])


def _next_pending_row(conn: Any, filters: dict[str, Any]) -> Any | None:
    sql, values = _pending_select(filters)
    return conn.execute(sql, values).fetchone()


def _pending_select(filters: dict[str, Any]) -> tuple[str, tuple[Any, ...]]:
    clauses = ["status = ?"]
    values: list[Any] = [JOB_STATUS_PENDING]
    for key, column in (("symbols", "symbol"), ("durations", "duration"), ("families", "model_family")):
        selected = tuple(filters.get(key) or ())
        if selected:
            clauses.append(f"{column} IN ({','.join('?' for _ in selected)})")
            values.extend(selected)
    sql = f"""
    SELECT * FROM model_search_jobs
    WHERE {' AND '.join(clauses)}
    ORDER BY priority ASC, created_at ASC
    LIMIT 1
    """
    return sql, tuple(values)


def _claim_row(conn: Any, job_id: str) -> dict[str, Any] | None:
    now = utc_now()
    cursor = conn.execute(
        claim_sql(),
        (JOB_STATUS_RUNNING, JOB_STAGE_COARSE, now, now, job_id, JOB_STATUS_PENDING),
    )
    if int(cursor.rowcount or 0) != 1:
        return None
    return decode_job(required_job(conn, job_id))


def _find_existing(conn: Any, spec: dict[str, Any]) -> Any | None:
    return conn.execute(
        existing_select_sql(),
        (spec["symbol"], spec["duration"], spec["model_family"], spec["profile"], spec["params_hash"]),
    ).fetchone()


def _reset_existing_job(conn: Any, job_id: str, options: EnqueueJobOptions) -> None:
    selected = options.resource or {}
    conn.execute(
        reset_sql(),
        (
            JOB_STATUS_PENDING,
            JOB_STAGE_QUEUED,
            selected.get("resourceProfile"),
            selected.get("internalThreads"),
            selected.get("parallelWorkers"),
            selected.get("xgboostProcessWorkers"),
            1 if options.reset_history else 0,
            job_id,
        ),
    )
