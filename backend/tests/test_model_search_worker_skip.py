from __future__ import annotations

import sqlite3
import tempfile
import uuid
from pathlib import Path

import pytest

from app.services import model_search_job_runner as runner
from app.services import model_search_job_store as store
from app.services import model_search_status_service as status_service
from app.services.model_search_job_schema import ensure_model_search_jobs_table
from app.services.model_search_job_types import JOB_STATUS_SKIPPED, JOB_STATUS_SUCCEEDED


def test_worker_skips_already_trained_pending_job(monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = _runtime_path("skip-worker") / "model-search.db"
    log_dir = _runtime_path("skip-worker-logs")
    _patch_store_db(monkeypatch, db_path)
    store.enqueue_model_search_jobs(
        symbols=("BTCUSDT",),
        durations=("10m",),
        families=("knn",),
        profile="fast",
    )

    monkeypatch.setattr(
        runner,
        "model_family_status",
        lambda *_args: {"status": "shadow_active", "shadowPredictionReady": True},
    )
    monkeypatch.setattr(
        runner,
        "run_model_candidate_search",
        lambda _config: pytest.fail("already trained job must not train"),
    )

    report = runner.run_one_model_search_job(runner.ModelSearchWorkerConfig(log_dir=log_dir))

    assert report["status"] == JOB_STATUS_SKIPPED
    assert report["result"]["reason"] == "already_trained_skipped"
    assert report["job"]["rejection_reason"] == "already_trained_skipped"
    assert "skipped" in Path(report["job"]["log_path"]).read_text(encoding="utf-8")


def test_worker_trains_reset_history_job_even_when_already_trained(monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = _runtime_path("reset-worker") / "model-search.db"
    log_dir = _runtime_path("reset-worker-logs")
    _patch_store_db(monkeypatch, db_path)
    store.enqueue_model_search_jobs(
        symbols=("BTCUSDT",),
        durations=("10m",),
        families=("knn",),
        profile="fast",
        reset_history=True,
    )
    calls = []

    monkeypatch.setattr(
        runner,
        "model_family_status",
        lambda *_args: {"status": "shadow_active", "shadowPredictionReady": True},
    )
    monkeypatch.setattr(
        runner,
        "run_model_candidate_search",
        lambda config: calls.append(config) or {"status": "shadow_active"},
    )

    report = runner.run_one_model_search_job(runner.ModelSearchWorkerConfig(log_dir=log_dir))

    assert report["status"] == "succeeded"
    assert calls
    assert report["job"]["resetHistory"] is True


def test_worker_continues_partial_job_even_when_active_model_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = _runtime_path("partial-ready-worker") / "model-search.db"
    log_dir = _runtime_path("partial-ready-worker-logs")
    _patch_store_db(monkeypatch, db_path)
    store.enqueue_model_search_jobs(
        symbols=("BTCUSDT",),
        durations=("10m",),
        families=("knn",),
        profile="fast",
        reset_history=True,
    )
    calls = []
    responses = iter([
        _batch_result("validation_failed", has_more=True),
        _batch_result("shadow_active", has_more=False),
    ])

    monkeypatch.setattr(
        runner,
        "model_family_status",
        lambda *_args: {"status": "shadow_active", "shadowPredictionReady": True},
    )
    monkeypatch.setattr(
        runner,
        "run_model_candidate_search",
        lambda config: calls.append(config) or next(responses),
    )

    partial = runner.run_one_model_search_job(runner.ModelSearchWorkerConfig(log_dir=log_dir))
    finished = runner.run_one_model_search_job(runner.ModelSearchWorkerConfig(log_dir=log_dir))

    assert partial["status"] == "partial"
    assert finished["status"] == JOB_STATUS_SUCCEEDED
    assert len(calls) == 2
    assert calls[0].reset_history is True
    assert calls[1].reset_history is False


def _patch_store_db(monkeypatch: pytest.MonkeyPatch, db_path: Path) -> None:
    monkeypatch.setattr(store, "get_conn", lambda: _connect(db_path))
    monkeypatch.setattr(status_service, "list_model_search_jobs", store.list_model_search_jobs)
    monkeypatch.setattr(runner, "model_search_queue_status", status_service.model_search_queue_status)


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    ensure_model_search_jobs_table(conn)
    return conn


def _runtime_path(name: str) -> Path:
    path = Path(tempfile.gettempdir()) / "incidentSpot-pytest-temp" / f"{name}-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _batch_result(status: str, *, has_more: bool) -> dict:
    return {
        "status": status,
        "jobBatch": {
            "selectedCandidates": 1,
            "availableCandidatesBeforeJob": 3,
            "remainingCandidatesAfterJob": 2 if has_more else 0,
            "hasMoreCandidates": has_more,
        },
    }
