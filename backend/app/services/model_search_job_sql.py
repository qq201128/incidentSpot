from __future__ import annotations


def finish_sql() -> str:
    return """
    UPDATE model_search_jobs
    SET status = ?, stage = ?, finished_at = ?, heartbeat_at = ?,
        artifact_path = ?, metrics_json = ?, training_report_json = ?,
        failure_type = NULL, failure_reason = NULL, rejection_reason = ?,
        failure_context_json = NULL, log_path = ?, resource_profile = ?,
        internal_threads = ?, parallel_workers = ?, xgboost_process_workers = ?
    WHERE job_id = ?
    """


def failure_sql() -> str:
    return """
    UPDATE model_search_jobs
    SET status = ?, finished_at = ?, heartbeat_at = ?, failure_type = ?,
        failure_reason = ?, failure_context_json = ?, log_path = ?,
        resource_profile = ?, internal_threads = ?, parallel_workers = ?,
        xgboost_process_workers = ?
    WHERE job_id = ?
    """


def retry_sql() -> str:
    return """
    UPDATE model_search_jobs
    SET status = ?, stage = ?, started_at = NULL, finished_at = NULL,
        heartbeat_at = NULL, failure_type = NULL, failure_reason = NULL,
        rejection_reason = NULL, failure_context_json = NULL
    WHERE job_id = ?
    """


def stale_select_sql() -> str:
    return """
    SELECT job_id, stage, heartbeat_at, started_at
    FROM model_search_jobs
    WHERE status = ? AND COALESCE(heartbeat_at, started_at, created_at) < ?
    """


def stale_update_sql() -> str:
    return """
    UPDATE model_search_jobs
    SET status = ?, finished_at = ?, failure_type = ?, failure_reason = ?,
        failure_context_json = ?
    WHERE job_id = ?
    """


def pending_select_sql() -> str:
    return """
    SELECT * FROM model_search_jobs
    WHERE status = ?
    ORDER BY priority ASC, created_at ASC
    LIMIT 1
    """


def claim_sql() -> str:
    return """
    UPDATE model_search_jobs
    SET status = ?, stage = ?, started_at = COALESCE(started_at, ?),
        heartbeat_at = ?, finished_at = NULL, attempt_count = attempt_count + 1
    WHERE job_id = ? AND status = ?
    """


def existing_select_sql() -> str:
    return """
    SELECT * FROM model_search_jobs
    WHERE symbol = ? AND duration = ? AND model_family = ? AND profile = ? AND params_hash = ?
    """


def reset_sql() -> str:
    return """
    UPDATE model_search_jobs
    SET status = ?, stage = ?, started_at = NULL, finished_at = NULL,
        heartbeat_at = NULL, artifact_path = NULL, metrics_json = NULL,
        training_report_json = NULL, failure_type = NULL, failure_reason = NULL,
        rejection_reason = NULL, failure_context_json = NULL, log_path = NULL
    WHERE job_id = ?
    """
