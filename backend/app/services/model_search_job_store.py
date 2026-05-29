from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from app.db.session import get_conn, run_db_write_with_retry
from app.services.model_search_job_payloads import (
    classified_result,
    decode_job,
    enqueue_payload,
    failure_values,
    finish_values,
    insert_sql,
    insert_values,
    job_spec,
    select_sql,
)
from app.services.model_search_job_schema import ensure_model_search_jobs_table
from app.services.model_search_job_sql import (
    claim_sql,
    existing_select_sql,
    failure_sql,
    finish_sql,
    pending_select_sql,
    reset_sql,
    retry_sql,
    stale_select_sql,
    stale_update_sql,
)
from app.services.model_search_job_types import (
    DEFAULT_MAX_RUNNING_JOBS,
    DEFAULT_MODEL_SEARCH_PRIORITY,
    DEFAULT_STALE_AFTER_SECONDS,
    JOB_STAGE_COARSE,
    JOB_STAGE_QUEUED,
    JOB_STATUS_FAILED,
    JOB_STATUS_PENDING,
    JOB_STATUS_REJECTED,
    JOB_STATUS_RUNNING,
    json_dumps,
    utc_now,
)


def enqueue_model_search_jobs(
    *,
    symbols: tuple[str, ...],
    durations: tuple[str, ...],
    families: tuple[str, ...],
    profile: str,
    priority: int = DEFAULT_MODEL_SEARCH_PRIORITY,
    reset_existing: bool = False,
) -> dict[str, Any]:
    specs = [
        job_spec(symbol=sym, duration=dur, family=fam, profile=profile, priority=priority)
        for sym in symbols
        for dur in durations
        for fam in families
    ]

    def _operation() -> dict[str, Any]:
        conn = get_conn()
        try:
            ensure_model_search_jobs_table(conn)
            rows = [_enqueue_one(conn, spec, reset_existing) for spec in specs]
            conn.commit()
            return enqueue_payload(rows)
        finally:
            conn.close()

    return run_db_write_with_retry(_operation)


def claim_next_model_search_job(
    *,
    max_running_jobs: int = DEFAULT_MAX_RUNNING_JOBS,
    stale_after_seconds: int = DEFAULT_STALE_AFTER_SECONDS,
) -> dict[str, Any] | None:
    if max_running_jobs <= 0:
        raise ValueError("max_running_jobs must be positive")

    def _operation() -> dict[str, Any] | None:
        conn = get_conn()
        try:
            ensure_model_search_jobs_table(conn)
            _mark_stale_running_jobs(conn, stale_after_seconds)
            if _running_count(conn) >= max_running_jobs:
                conn.commit()
                return None
            row = _next_pending_row(conn)
            if row is None:
                conn.commit()
                return None
            job = _claim_row(conn, row["job_id"])
            conn.commit()
            return job
        finally:
            conn.close()

    return run_db_write_with_retry(_operation)


def heartbeat_model_search_job(job_id: str) -> dict[str, Any]:
    def _operation() -> dict[str, Any]:
        conn = get_conn()
        try:
            ensure_model_search_jobs_table(conn)
            conn.execute("UPDATE model_search_jobs SET heartbeat_at = ? WHERE job_id = ?", (utc_now(), job_id))
            row = _required_job(conn, job_id)
            conn.commit()
            return decode_job(row)
        finally:
            conn.close()

    return run_db_write_with_retry(_operation)


def finish_model_search_job(
    job_id: str,
    *,
    result: dict[str, Any],
    resource: dict[str, Any],
    artifact_path: str | None,
    log_path: str | None,
) -> dict[str, Any]:
    final = classified_result(result)

    def _operation() -> dict[str, Any]:
        conn = get_conn()
        try:
            ensure_model_search_jobs_table(conn)
            conn.execute(finish_sql(), finish_values(
                job_id=job_id, final=final, result=result, resource=resource,
                artifact_path=artifact_path, log_path=log_path,
            ))
            row = _required_job(conn, job_id)
            conn.commit()
            return decode_job(row)
        finally:
            conn.close()

    return run_db_write_with_retry(_operation)


def fail_model_search_job(
    job_id: str,
    *,
    failure_type: str,
    failure_reason: str,
    failure_context: dict[str, Any],
    resource: dict[str, Any],
    log_path: str | None,
) -> dict[str, Any]:
    def _operation() -> dict[str, Any]:
        conn = get_conn()
        try:
            ensure_model_search_jobs_table(conn)
            conn.execute(failure_sql(), failure_values(
                job_id=job_id, failure_type=failure_type, failure_reason=failure_reason,
                failure_context=failure_context, resource=resource, log_path=log_path,
            ))
            row = _required_job(conn, job_id)
            conn.commit()
            return decode_job(row)
        finally:
            conn.close()

    return run_db_write_with_retry(_operation)


def retry_failed_model_search_job(job_id: str) -> dict[str, Any]:
    def _operation() -> dict[str, Any]:
        conn = get_conn()
        try:
            ensure_model_search_jobs_table(conn)
            current = _required_job(conn, job_id)
            if current["status"] not in {JOB_STATUS_FAILED, JOB_STATUS_REJECTED}:
                raise ValueError(f"job {job_id} is not failed/rejected")
            conn.execute(retry_sql(), (JOB_STATUS_PENDING, JOB_STAGE_QUEUED, job_id))
            row = _required_job(conn, job_id)
            conn.commit()
            return decode_job(row)
        finally:
            conn.close()

    return run_db_write_with_retry(_operation)


def list_model_search_jobs(filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    conn = get_conn()
    try:
        ensure_model_search_jobs_table(conn)
        sql, values = select_sql(filters or {})
        rows = conn.execute(sql, values).fetchall()
        return [decode_job(row) for row in rows]
    finally:
        conn.close()


def _enqueue_one(conn: Any, spec: dict[str, Any], reset_existing: bool) -> dict[str, Any]:
    existing = _find_existing(conn, spec)
    if existing is None:
        conn.execute(insert_sql(), insert_values(spec))
        return {**decode_job(_required_job(conn, spec["job_id"])), "enqueueAction": "created"}
    if reset_existing:
        _reset_existing_job(conn, existing["job_id"])
        return {**decode_job(_required_job(conn, existing["job_id"])), "enqueueAction": "reset"}
    return {**decode_job(existing), "enqueueAction": "existing"}


def _mark_stale_running_jobs(conn: Any, stale_after_seconds: int) -> None:
    cutoff = (datetime.now(timezone.utc) - timedelta(seconds=max(stale_after_seconds, 1))).isoformat()
    rows = conn.execute(stale_select_sql(), (JOB_STATUS_RUNNING, cutoff)).fetchall()
    for row in rows:
        context = {"stage": row["stage"], "heartbeatAt": row["heartbeat_at"], "startedAt": row["started_at"]}
        conn.execute(stale_update_sql(), (
            JOB_STATUS_FAILED, utc_now(), "running_timeout",
            "running job heartbeat timed out", json_dumps(context), row["job_id"],
        ))


def _running_count(conn: Any) -> int:
    row = conn.execute("SELECT COUNT(*) AS c FROM model_search_jobs WHERE status = ?", (JOB_STATUS_RUNNING,)).fetchone()
    return int(row["c"])


def _next_pending_row(conn: Any) -> Any | None:
    return conn.execute(pending_select_sql(), (JOB_STATUS_PENDING,)).fetchone()


def _claim_row(conn: Any, job_id: str) -> dict[str, Any]:
    now = utc_now()
    conn.execute(
        claim_sql(),
        (JOB_STATUS_RUNNING, JOB_STAGE_COARSE, now, now, job_id, JOB_STATUS_PENDING),
    )
    return decode_job(_required_job(conn, job_id))


def _find_existing(conn: Any, spec: dict[str, Any]) -> Any | None:
    return conn.execute(
        existing_select_sql(),
        (spec["symbol"], spec["duration"], spec["model_family"], spec["profile"], spec["params_hash"]),
    ).fetchone()


def _required_job(conn: Any, job_id: str) -> Any:
    row = conn.execute("SELECT * FROM model_search_jobs WHERE job_id = ?", (job_id,)).fetchone()
    if row is None:
        raise ValueError(f"model search job not found: {job_id}")
    return row


def _reset_existing_job(conn: Any, job_id: str) -> None:
    conn.execute(reset_sql(), (JOB_STATUS_PENDING, JOB_STAGE_QUEUED, job_id))
