from __future__ import annotations

from typing import Any

from app.db.session import get_conn, run_db_write_with_retry
from app.services.model_search_job_payloads import (
    classified_result,
    decode_job,
    enqueue_payload,
    failure_values,
    finish_values,
    job_spec,
    select_sql,
)
from app.services.model_search_job_store_helpers import (
    EnqueueJobOptions,
    claim_available_row as _claim_available_row,
    enqueue_one as _enqueue_one,
    mark_stale_running_jobs as _mark_stale_running_jobs,
    required_job as _required_job,
    update_pending_resource as _update_pending_resource,
)
from app.services.model_search_job_schema import ensure_model_search_jobs_table
from app.services.model_search_job_sql import (
    failure_sql,
    finish_sql,
    retry_sql,
)
from app.services.model_search_job_types import (
    DEFAULT_MAX_RUNNING_JOBS,
    DEFAULT_MODEL_SEARCH_PRIORITY,
    DEFAULT_STALE_AFTER_SECONDS,
    JOB_STAGE_QUEUED,
    JOB_STATUS_FAILED,
    JOB_STATUS_PENDING,
    JOB_STATUS_REJECTED,
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
    reset_history: bool = False,
    resource: dict[str, Any] | None = None,
) -> dict[str, Any]:
    specs = [
        job_spec(symbol=sym, duration=dur, family=fam, profile=profile, priority=priority, resource=resource)
        for sym in symbols
        for dur in durations
        for fam in families
    ]

    def _operation() -> dict[str, Any]:
        conn = get_conn()
        try:
            ensure_model_search_jobs_table(conn)
            options = EnqueueJobOptions(reset_existing, reset_history, resource)
            rows = [_enqueue_one(conn, spec, options) for spec in specs]
            conn.commit()
            return enqueue_payload(rows)
        finally:
            conn.close()

    return run_db_write_with_retry(_operation)


def claim_next_model_search_job(
    *,
    max_running_jobs: int = DEFAULT_MAX_RUNNING_JOBS,
    stale_after_seconds: int = DEFAULT_STALE_AFTER_SECONDS,
    filters: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if max_running_jobs <= 0:
        raise ValueError("max_running_jobs must be positive")

    def _operation() -> dict[str, Any] | None:
        conn = get_conn()
        try:
            ensure_model_search_jobs_table(conn)
            _mark_stale_running_jobs(conn, stale_after_seconds)
            job = _claim_available_row(conn, max_running_jobs, filters)
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


def retry_failed_model_search_job(
    job_id: str,
    *,
    clear_reset_history: bool = False,
    move_to_back: bool = False,
) -> dict[str, Any]:
    def _operation() -> dict[str, Any]:
        conn = get_conn()
        try:
            ensure_model_search_jobs_table(conn)
            current = _required_job(conn, job_id)
            if current["status"] not in {JOB_STATUS_FAILED, JOB_STATUS_REJECTED}:
                raise ValueError(f"job {job_id} is not failed/rejected")
            sql = retry_sql(clear_reset_history=clear_reset_history, move_to_back=move_to_back)
            values = (
                (JOB_STATUS_PENDING, JOB_STAGE_QUEUED, utc_now(), job_id)
                if move_to_back
                else (JOB_STATUS_PENDING, JOB_STAGE_QUEUED, job_id)
            )
            conn.execute(sql, values)
            row = _required_job(conn, job_id)
            conn.commit()
            return decode_job(row)
        finally:
            conn.close()

    return run_db_write_with_retry(_operation)


def update_pending_model_search_job_resources(
    *,
    symbols: tuple[str, ...],
    durations: tuple[str, ...],
    families: tuple[str, ...],
    profile: str,
    resource: dict[str, Any],
    priority: int = DEFAULT_MODEL_SEARCH_PRIORITY,
) -> dict[str, Any]:
    specs = [
        job_spec(symbol=sym, duration=dur, family=fam, profile=profile, priority=priority, resource=resource)
        for sym in symbols
        for dur in durations
        for fam in families
    ]

    def _operation() -> dict[str, Any]:
        conn = get_conn()
        try:
            ensure_model_search_jobs_table(conn)
            updated = [_update_pending_resource(conn, spec, resource) for spec in specs]
            conn.commit()
            return {"version": "model_search_jobs_resource_update_v1", "matched": sum(updated), "requested": len(specs)}
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
