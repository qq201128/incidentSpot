from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

import pytest

from app.services import model_search_job_runner as runner
from app.services import model_search_job_store as store
from app.services import model_search_status_service as status_service
from app.services.model_search_job_schema import ensure_model_search_jobs_table
from app.services.model_search_job_types import JOB_STATUS_FAILED, JOB_STATUS_RUNNING


def test_pending_job_without_worker_requires_command(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_store_db(monkeypatch, _db_path("required"))
    _enqueue_one()

    report = status_service.model_search_queue_status()

    assert report["workerStatus"]["state"] == "worker_required"
    assert report["workerStatus"]["pendingJobs"] == 1
    assert report["workerStatus"]["runningJobs"] == 0
    assert "run_model_search_worker.py --loop" in report["workerStatus"]["workerRequiredCommand"]


def test_running_job_marks_worker_running(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_store_db(monkeypatch, _db_path("running"))
    _enqueue_one()
    store.claim_next_model_search_job(max_running_jobs=1)

    report = status_service.model_search_queue_status()

    assert report["workerStatus"]["state"] == JOB_STATUS_RUNNING
    assert report["workerStatus"]["runningJobs"] == 1
    assert report["workerStatus"]["latestLogPath"] is None


def test_pending_filtered_job_is_queued_when_any_worker_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_store_db(monkeypatch, _db_path("queued"))
    _enqueue_one(symbol="ETHUSDT")
    store.claim_next_model_search_job(max_running_jobs=1)
    _enqueue_one(symbol="BTCUSDT")

    report = status_service.model_search_queue_status({"symbols": ("BTCUSDT",)})

    assert report["workerStatus"]["state"] == "queued"
    assert report["workerStatus"]["pendingJobs"] == 1
    assert report["workerStatus"]["runningJobs"] == 1


def test_failed_job_exposes_reason_and_log_path(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_store_db(monkeypatch, _db_path("failed"))
    _enqueue_one()
    job = store.claim_next_model_search_job(max_running_jobs=1)
    store.fail_model_search_job(
        job["job_id"],
        failure_type="RuntimeError",
        failure_reason="training crashed",
        failure_context={"stage": "coarse"},
        resource=_resource_payload(),
        log_path="runtime/model-search-jobs/fail.log",
    )

    report = status_service.model_search_queue_status()

    assert report["workerStatus"]["state"] == JOB_STATUS_FAILED
    assert report["workerStatus"]["latestFailureReason"] == "training crashed"
    assert report["workerStatus"]["latestLogPath"] == "runtime/model-search-jobs/fail.log"


def test_new_success_clears_previous_worker_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_store_db(monkeypatch, _db_path("success-clears"))
    _enqueue_one()
    failed = store.claim_next_model_search_job(max_running_jobs=1)
    store.fail_model_search_job(
        failed["job_id"],
        failure_type="RuntimeError",
        failure_reason="old failure",
        failure_context={},
        resource=_resource_payload(),
        log_path="runtime/old.log",
    )
    store.enqueue_model_search_jobs(
        symbols=("ETHUSDT",),
        durations=("10m",),
        families=("knn",),
        profile="fast",
    )
    succeeded = store.claim_next_model_search_job(max_running_jobs=1)
    store.finish_model_search_job(
        succeeded["job_id"],
        result={"status": "trained"},
        resource=_resource_payload(),
        artifact_path=None,
        log_path="runtime/success.log",
    )

    report = status_service.model_search_queue_status()

    assert report["workerStatus"]["state"] == "idle"
    assert report["workerStatus"]["latestFailureReason"] is None


def _enqueue_one(symbol: str = "BTCUSDT") -> None:
    store.enqueue_model_search_jobs(
        symbols=(symbol,),
        durations=("10m",),
        families=("knn",),
        profile="fast",
    )


def _patch_store_db(monkeypatch: pytest.MonkeyPatch, db_path: Path) -> None:
    monkeypatch.setattr(store, "get_conn", lambda: _connect(db_path))
    monkeypatch.setattr(status_service, "list_model_search_jobs", store.list_model_search_jobs)
    monkeypatch.setattr(runner, "model_search_queue_status", status_service.model_search_queue_status)


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    ensure_model_search_jobs_table(conn)
    return conn


def _db_path(name: str) -> Path:
    path = Path(__file__).resolve().parents[1] / "runtime" / "pytest-temp" / f"worker-status-{name}-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path / "model-search.db"


def _resource_payload() -> dict:
    return {
        "resourceProfile": "local_safe",
        "internalThreads": 4,
        "parallelWorkers": 1,
        "xgboostProcessWorkers": 1,
    }
