from __future__ import annotations

from typing import Any

CREATE_MODEL_SEARCH_JOBS_SQL = """
CREATE TABLE IF NOT EXISTS model_search_jobs (
  job_id TEXT PRIMARY KEY,
  symbol TEXT NOT NULL,
  duration TEXT NOT NULL,
  model_family TEXT NOT NULL,
  profile TEXT NOT NULL,
  stage TEXT NOT NULL,
  params_json TEXT NOT NULL,
  params_hash TEXT NOT NULL,
  status TEXT NOT NULL,
  priority INTEGER NOT NULL,
  attempt_count INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL,
  started_at TEXT,
  finished_at TEXT,
  heartbeat_at TEXT,
  artifact_path TEXT,
  metrics_json TEXT,
  training_report_json TEXT,
  failure_type TEXT,
  failure_reason TEXT,
  rejection_reason TEXT,
  failure_context_json TEXT,
  log_path TEXT,
  resource_profile TEXT,
  internal_threads INTEGER,
  parallel_workers INTEGER,
  xgboost_process_workers INTEGER,
  reset_history INTEGER NOT NULL DEFAULT 0,
  UNIQUE(symbol, duration, model_family, profile, params_hash)
)
"""

CREATE_MODEL_SEARCH_JOBS_STATUS_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_model_search_jobs_status
ON model_search_jobs(status, priority, created_at)
"""

MODEL_SEARCH_JOB_COLUMN_MIGRATIONS = (
    "ALTER TABLE model_search_jobs ADD COLUMN training_report_json TEXT",
    "ALTER TABLE model_search_jobs ADD COLUMN rejection_reason TEXT",
    "ALTER TABLE model_search_jobs ADD COLUMN resource_profile TEXT",
    "ALTER TABLE model_search_jobs ADD COLUMN internal_threads INTEGER",
    "ALTER TABLE model_search_jobs ADD COLUMN parallel_workers INTEGER",
    "ALTER TABLE model_search_jobs ADD COLUMN xgboost_process_workers INTEGER",
    "ALTER TABLE model_search_jobs ADD COLUMN reset_history INTEGER NOT NULL DEFAULT 0",
)


def ensure_model_search_jobs_table(conn: Any) -> None:
    conn.execute(CREATE_MODEL_SEARCH_JOBS_SQL)
    for sql in MODEL_SEARCH_JOB_COLUMN_MIGRATIONS:
        _execute_ignored_duplicate(conn, sql)
    conn.execute(CREATE_MODEL_SEARCH_JOBS_STATUS_INDEX_SQL)


def _execute_ignored_duplicate(conn: Any, sql: str) -> None:
    try:
        conn.execute(sql)
    except Exception as exc:
        if "duplicate column" not in str(exc).lower():
            raise
