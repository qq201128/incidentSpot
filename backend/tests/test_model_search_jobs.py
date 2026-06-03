from __future__ import annotations

import sqlite3
import tempfile
import time
import uuid
from inspect import signature
from pathlib import Path

import pytest

from app.services import model_search_job_runner as runner
from app.services import model_search_job_store as store
from app.services import model_search_untrained_enqueue as untrained_enqueue
from app.services import model_search_resource as resource
from app.services import model_search_status_service as status_service
from app.api import models as models_api
from app.services.model_search_job_schema import ensure_model_search_jobs_table
from app.services.model_search_job_types import (
    JOB_STATUS_FAILED,
    JOB_STATUS_PENDING,
    JOB_STATUS_REJECTED,
    JOB_STATUS_RUNNING,
    JOB_STATUS_SUCCEEDED,
)


def test_enqueue_deduplicates_matching_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = _db_path("enqueue")
    _patch_store_db(monkeypatch, db_path)

    first = store.enqueue_model_search_jobs(
        symbols=("BTCUSDT",),
        durations=("10m",),
        families=("knn",),
        profile="fast",
    )
    second = store.enqueue_model_search_jobs(
        symbols=("BTCUSDT",),
        durations=("10m",),
        families=("knn",),
        profile="fast",
    )

    assert first["created"] == 1
    assert second["existing"] == 1
    assert len(store.list_model_search_jobs()) == 1


def test_enqueue_keeps_distinct_resource_configs(monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = _db_path("enqueue-resource")
    _patch_store_db(monkeypatch, db_path)

    first = store.enqueue_model_search_jobs(
        symbols=("BTCUSDT",),
        durations=("10m",),
        families=("knn",),
        profile="fast",
        resource={**_resource_payload(), "parallelWorkers": 2},
    )
    second = store.enqueue_model_search_jobs(
        symbols=("BTCUSDT",),
        durations=("10m",),
        families=("knn",),
        profile="fast",
        resource={**_resource_payload(), "parallelWorkers": 4},
    )

    rows = store.list_model_search_jobs()
    assert first["created"] == 1
    assert second["created"] == 1
    assert len(rows) == 2
    assert {row["parallel_workers"] for row in rows} == {2, 4}


def test_enqueue_persists_reset_history_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = _db_path("enqueue-reset-history")
    _patch_store_db(monkeypatch, db_path)

    store.enqueue_model_search_jobs(
        symbols=("BTCUSDT",),
        durations=("10m",),
        families=("knn",),
        profile="fast",
        reset_history=True,
    )

    job = store.claim_next_model_search_job(max_running_jobs=1)
    assert job["resetHistory"] is True


def test_reset_existing_job_updates_reset_history_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = _db_path("reset-existing-reset-history")
    _patch_store_db(monkeypatch, db_path)
    _enqueue_one()

    reset = store.enqueue_model_search_jobs(
        symbols=("BTCUSDT",),
        durations=("10m",),
        families=("knn",),
        profile="fast",
        reset_existing=True,
        reset_history=True,
    )

    assert reset["reset"] == 1
    job = store.claim_next_model_search_job(max_running_jobs=1)
    assert job["resetHistory"] is True


def test_update_pending_resources_does_not_touch_running_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = _db_path("pending-resource-update")
    _patch_store_db(monkeypatch, db_path)
    store.enqueue_model_search_jobs(
        symbols=("BTCUSDT",),
        durations=("10m",),
        families=("knn",),
        profile="fast",
        resource={**_resource_payload(), "parallelWorkers": 1},
    )
    store.enqueue_model_search_jobs(
        symbols=("BTCUSDT",),
        durations=("10m",),
        families=("svm",),
        profile="fast",
        resource={**_resource_payload(), "parallelWorkers": 1},
    )
    running = store.claim_next_model_search_job(max_running_jobs=1)

    result = store.update_pending_model_search_job_resources(
        symbols=("BTCUSDT",),
        durations=("10m",),
        families=("knn", "svm"),
        profile="fast",
        resource={**_resource_payload(), "parallelWorkers": 6},
    )
    jobs = store.list_model_search_jobs({"symbols": ("BTCUSDT",), "durations": ("10m",)})

    assert result["matched"] == 1
    assert next(job for job in jobs if job["job_id"] == running["job_id"])["parallel_workers"] == 1
    assert next(job for job in jobs if job["status"] == JOB_STATUS_PENDING)["parallel_workers"] == 6


def test_claim_marks_pending_job_running(monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = _db_path("claim")
    _patch_store_db(monkeypatch, db_path)
    _enqueue_one()

    job = store.claim_next_model_search_job(max_running_jobs=1)

    assert job is not None
    assert job["status"] == JOB_STATUS_RUNNING
    assert job["attempt_count"] == 1
    assert store.claim_next_model_search_job(max_running_jobs=1) is None


def test_claim_allows_multiple_running_jobs_when_capacity_allows(monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = _db_path("claim-multiple")
    _patch_store_db(monkeypatch, db_path)
    _enqueue_one(symbol="BTCUSDT")
    _enqueue_one(symbol="ETHUSDT")

    first = store.claim_next_model_search_job(max_running_jobs=2)
    second = store.claim_next_model_search_job(max_running_jobs=2)
    blocked = store.claim_next_model_search_job(max_running_jobs=2)

    assert first is not None
    assert second is not None
    assert {first["symbol"], second["symbol"]} == {"BTCUSDT", "ETHUSDT"}
    assert first["status"] == JOB_STATUS_RUNNING
    assert second["status"] == JOB_STATUS_RUNNING
    assert blocked is None


def test_finish_success_and_rejection_are_written(monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = _db_path("finish")
    _patch_store_db(monkeypatch, db_path)
    _enqueue_one()
    succeeded = store.claim_next_model_search_job(max_running_jobs=1)

    stored = store.finish_model_search_job(
        succeeded["job_id"],
        result={"status": "shadow_active", "modelVersion": "m1"},
        resource=_resource_payload(),
        artifact_path="artifacts/knn",
        log_path="runtime/job.log",
    )

    assert stored["status"] == JOB_STATUS_SUCCEEDED
    assert stored["artifact_path"] == "artifacts/knn"
    assert stored["metrics"]["resource"]["internalThreads"] == 1

    _enqueue_one(symbol="ETHUSDT")
    rejected = store.claim_next_model_search_job(max_running_jobs=1)
    stored = store.finish_model_search_job(
        rejected["job_id"],
        result={"status": "validation_failed", "reason": "recent_rolling_failed"},
        resource=_resource_payload(),
        artifact_path=None,
        log_path="runtime/rejected.log",
    )

    assert stored["status"] == JOB_STATUS_REJECTED
    assert stored["rejection_reason"] == "recent_rolling_failed"


def test_failed_job_can_be_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = _db_path("retry")
    _patch_store_db(monkeypatch, db_path)
    _enqueue_one()
    job = store.claim_next_model_search_job(max_running_jobs=1)
    failed = store.fail_model_search_job(
        job["job_id"],
        failure_type="RuntimeError",
        failure_reason="training crashed",
        failure_context={"stage": "full"},
        resource=_resource_payload(),
        log_path="runtime/fail.log",
    )

    retried = store.retry_failed_model_search_job(failed["job_id"])

    assert failed["status"] == JOB_STATUS_FAILED
    assert failed["failure_reason"] == "training crashed"
    assert retried["status"] == JOB_STATUS_PENDING
    assert retried["failure_reason"] is None


def test_failed_reset_job_retry_preserves_reset_history(monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = _db_path("retry-reset-history")
    _patch_store_db(monkeypatch, db_path)
    store.enqueue_model_search_jobs(
        symbols=("BTCUSDT",),
        durations=("10m",),
        families=("knn",),
        profile="fast",
        reset_history=True,
    )
    job = store.claim_next_model_search_job(max_running_jobs=1)
    failed = store.fail_model_search_job(
        job["job_id"],
        failure_type="RuntimeError",
        failure_reason="training crashed",
        failure_context={"stage": "full"},
        resource=_resource_payload(),
        log_path="runtime/fail.log",
    )

    retried = store.retry_failed_model_search_job(failed["job_id"])

    assert retried["status"] == JOB_STATUS_PENDING
    assert retried["resetHistory"] is True


def test_stale_running_job_is_marked_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = _db_path("stale")
    _patch_store_db(monkeypatch, db_path)
    _enqueue_one()
    job = store.claim_next_model_search_job(max_running_jobs=1)
    conn = _connect(db_path)
    conn.execute(
        "UPDATE model_search_jobs SET heartbeat_at = '2020-01-01T00:00:00+00:00' WHERE job_id = ?",
        (job["job_id"],),
    )
    conn.commit()
    conn.close()

    assert store.claim_next_model_search_job(max_running_jobs=1, stale_after_seconds=1) is None
    failed = store.list_model_search_jobs({"statuses": (JOB_STATUS_FAILED,)})[0]
    assert failed["failure_type"] == "running_timeout"


def test_resource_config_sets_thread_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = resource.apply_model_search_resource_config(
        resource.ModelSearchResourceConfig(internal_threads=3, parallel_workers=1)
    )

    assert payload["internalThreads"] == 3
    assert payload["threadEnv"]["OPENBLAS_NUM_THREADS"] == "3"
    assert payload["MODEL_FAMILY_XGBOOST_PROCESS_WORKERS"] == "1"


def test_worker_runs_one_job_and_persists_resource(monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = _db_path("worker")
    log_dir = _runtime_path("worker-logs")
    _patch_store_db(monkeypatch, db_path)
    _enqueue_one()
    calls = []

    def fake_search(config):
        calls.append(config)
        return {"status": "shadow_active", "modelVersion": "knn_v1"}

    monkeypatch.setattr(runner, "run_model_candidate_search", fake_search)
    report = runner.run_one_model_search_job(
        runner.ModelSearchWorkerConfig(log_dir=log_dir, resource=resource.ModelSearchResourceConfig(internal_threads=2))
    )

    assert report["status"] == JOB_STATUS_SUCCEEDED
    assert calls[0].family == "knn"
    assert calls[0].parallel_workers == 1
    assert calls[0].candidates_per_job == 1
    assert report["job"]["internal_threads"] == 2
    assert Path(report["job"]["log_path"]).exists()


def test_worker_requeues_partial_candidate_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = _db_path("worker-partial")
    log_dir = _runtime_path("worker-partial-logs")
    _patch_store_db(monkeypatch, db_path)
    _enqueue_one()
    calls = []

    def fake_search(config):
        calls.append(config)
        return {
            "status": "validation_failed",
            "jobBatch": {
                "selectedCandidates": 1,
                "availableCandidatesBeforeJob": 3,
                "remainingCandidatesAfterJob": 2,
                "hasMoreCandidates": True,
            },
        }

    monkeypatch.setattr(runner, "run_model_candidate_search", fake_search)

    report = runner.run_one_model_search_job(runner.ModelSearchWorkerConfig(log_dir=log_dir))
    jobs = store.list_model_search_jobs()

    assert report["status"] == "partial"
    assert calls[0].candidates_per_job == 1
    assert len(jobs) == 1
    assert jobs[0]["status"] == JOB_STATUS_PENDING


def test_worker_partial_continuation_clears_reset_history(monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = _db_path("worker-partial-reset-history")
    log_dir = _runtime_path("worker-partial-reset-history-logs")
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

    def fake_search(config):
        calls.append(config)
        return next(responses)

    monkeypatch.setattr(runner, "run_model_candidate_search", fake_search)

    partial = runner.run_one_model_search_job(runner.ModelSearchWorkerConfig(log_dir=log_dir))
    requeued = store.list_model_search_jobs()[0]
    finished = runner.run_one_model_search_job(runner.ModelSearchWorkerConfig(log_dir=log_dir))

    assert partial["status"] == "partial"
    assert calls[0].reset_history is True
    assert requeued["resetHistory"] is False
    assert calls[1].reset_history is False
    assert finished["status"] == JOB_STATUS_SUCCEEDED


def test_worker_uses_resource_config_stored_on_job(monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = _db_path("worker-resource")
    log_dir = _runtime_path("worker-resource-logs")
    _patch_store_db(monkeypatch, db_path)
    store.enqueue_model_search_jobs(
        symbols=("BTCUSDT",),
        durations=("10m",),
        families=("knn",),
        profile="fast",
        resource={
            "resourceProfile": "api_requested",
            "internalThreads": 2,
            "parallelWorkers": 3,
            "xgboostProcessWorkers": 1,
        },
    )
    calls = []
    monkeypatch.setattr(runner, "run_model_candidate_search", lambda config: calls.append(config) or {"status": "shadow_active"})

    report = runner.run_one_model_search_job(
        runner.ModelSearchWorkerConfig(log_dir=log_dir, resource=resource.ModelSearchResourceConfig(parallel_workers=1))
    )

    assert calls[0].parallel_workers == 3
    assert report["job"]["resource_profile"] == "api_requested"
    assert report["job"]["parallel_workers"] == 3


def test_worker_surfaces_heartbeat_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = _db_path("heartbeat")
    log_dir = _runtime_path("heartbeat-logs")
    _patch_store_db(monkeypatch, db_path)
    _enqueue_one()
    heartbeats = []

    def fake_heartbeat(_job_id):
        heartbeats.append(_job_id)
        if len(heartbeats) > 1:
            raise RuntimeError("heartbeat write failed")
        return {}

    monkeypatch.setattr(runner, "heartbeat_model_search_job", fake_heartbeat)
    monkeypatch.setattr(runner, "run_model_candidate_search", lambda _config: time.sleep(1.2) or {"status": "shadow_active"})

    with pytest.raises(RuntimeError, match="heartbeat"):
        runner.run_one_model_search_job(
            runner.ModelSearchWorkerConfig(
                log_dir=log_dir,
                heartbeat_interval_seconds=1,
            )
        )
    failed = store.list_model_search_jobs({"statuses": (JOB_STATUS_FAILED,)})[0]
    assert "failure" in Path(failed["log_path"]).read_text(encoding="utf-8")


def test_status_report_groups_jobs_and_lifecycle_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = _db_path("status")
    _patch_store_db(monkeypatch, db_path)
    _enqueue_one()
    monkeypatch.setattr(
        status_service,
        "model_family_status",
        lambda family, symbol, duration: {
            "status": "shadow_active",
            "shadowPredictionReady": True,
            "shadowPredictionBlockedReason": "passed",
            "candidateSearchProgress": {"status": "running"},
        },
    )
    monkeypatch.setattr(
        status_service,
        "model_family_daily_candidate_report",
        lambda symbol, duration: {"models": [{"paperLiveStatus": "paper_collecting"}]},
    )

    report = status_service.model_search_status_with_lifecycle()

    assert report["counts"][JOB_STATUS_PENDING] == 1
    duration = report["symbols"][0]["durations"][0]
    assert duration["paperLive"]["paperCollectingCount"] == 1
    assert duration["families"][0]["shadowPredictionReady"] is True


def test_status_report_exposes_model_status_failure_details(monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = _db_path("status-failed")
    _patch_store_db(monkeypatch, db_path)
    _enqueue_one()

    def fail_status(*_args):
        raise RuntimeError("artifact metadata missing")

    monkeypatch.setattr(status_service, "model_family_status", fail_status)
    monkeypatch.setattr(
        status_service,
        "model_family_daily_candidate_report",
        lambda symbol, duration: {"models": []},
    )

    report = status_service.model_search_status_with_lifecycle()

    family = report["symbols"][0]["durations"][0]["families"][0]
    assert family["modelStatus"] == "status_failed"
    assert family["shadowPredictionReady"] is False
    assert family["blockedReason"] == "artifact metadata missing"
    assert family["modelStatusError"] == "artifact metadata missing"
    assert family["modelStatusExceptionType"] == "RuntimeError"


def test_candidate_search_api_only_enqueues_job(monkeypatch: pytest.MonkeyPatch) -> None:
    queued = {
        "jobs": [{"job_id": "job-1", "status": JOB_STATUS_PENDING}],
        "created": 1,
        "existing": 0,
    }
    calls = []

    monkeypatch.setattr(
        models_api,
        "enqueue_untrained_model_search_jobs",
        lambda **kwargs: calls.append(kwargs) or queued,
    )
    monkeypatch.setattr(
        models_api,
        "model_family_status",
        lambda family, symbol, duration: {
            "modelFamily": family,
            "symbol": symbol,
            "duration": duration,
            "status": "untrained",
        },
    )

    response = models_api.model_candidate_search(
        "knn",
        symbol="btcusdt",
        duration="10m",
        profile="fast",
        parallel_workers=4,
        internal_threads=4,
        xgboost_process_workers=1,
    )

    assert response["modelSearchJob"]["job_id"] == "job-1"
    assert calls[0]["symbols"] == ("BTCUSDT",)
    assert calls[0]["families"] == ("knn",)
    assert calls[0]["resource"]["parallelWorkers"] == 4
    assert calls[0]["reset_history"] is False


def test_candidate_search_api_passes_reset_history_to_queue(monkeypatch: pytest.MonkeyPatch) -> None:
    queued = {"jobs": [{"job_id": "job-1", "status": JOB_STATUS_PENDING}], "created": 1}
    calls = []
    monkeypatch.setattr(models_api, "enqueue_untrained_model_search_jobs", lambda **kwargs: calls.append(kwargs) or queued)
    monkeypatch.setattr(
        models_api,
        "model_family_status",
        lambda family, symbol, duration: {"modelFamily": family, "symbol": symbol, "duration": duration},
    )

    models_api.model_candidate_search(
        "knn",
        symbol="btcusdt",
        duration="10m",
        profile="fast",
        reset_history=True,
        parallel_workers=4,
        internal_threads=4,
        xgboost_process_workers=1,
    )

    assert calls[0]["reset_existing"] is True
    assert calls[0]["reset_history"] is True


def test_candidate_search_api_default_resource_matches_training_rules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queued = {"jobs": [{"job_id": "job-1", "status": JOB_STATUS_PENDING}], "created": 1}
    calls = []
    monkeypatch.setattr(models_api, "enqueue_untrained_model_search_jobs", lambda **kwargs: calls.append(kwargs) or queued)
    monkeypatch.setattr(
        models_api,
        "model_family_status",
        lambda family, symbol, duration: {"modelFamily": family, "symbol": symbol, "duration": duration},
    )
    route_default = signature(models_api.model_candidate_search).parameters["parallel_workers"].default
    assert route_default.default == 1
    assert route_default.alias == "parallelWorkers"

    models_api.model_candidate_search(
        "knn",
        symbol="btcusdt",
        duration="10m",
        profile="fast",
        parallel_workers=1,
        internal_threads=1,
        xgboost_process_workers=1,
    )

    assert calls[0]["resource"]["internalThreads"] == 1
    assert calls[0]["resource"]["parallelWorkers"] == 1
    assert calls[0]["resource"]["xgboostProcessWorkers"] == 1


def test_candidate_search_api_normalizes_query_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    queued = {"jobs": [{"job_id": "job-1", "status": JOB_STATUS_PENDING}], "created": 1}
    calls = []
    monkeypatch.setattr(models_api, "enqueue_untrained_model_search_jobs", lambda **kwargs: calls.append(kwargs) or queued)
    monkeypatch.setattr(
        models_api,
        "model_family_status",
        lambda family, symbol, duration: {"modelFamily": family, "symbol": symbol, "duration": duration},
    )

    response = models_api.model_candidate_search("knn", symbol="btcusdt")

    assert response["duration"] == "10m"
    assert calls[0]["durations"] == ("10m",)
    assert calls[0]["profile"] == "full"
    assert calls[0]["reset_history"] is False
    assert calls[0]["resource"]["internalThreads"] == 1
    assert calls[0]["resource"]["parallelWorkers"] == 1
    assert calls[0]["resource"]["xgboostProcessWorkers"] == 1


def test_untrained_enqueue_skips_trained_targets_without_cross_product(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_status(family: str, symbol: str, duration: str) -> dict:
        trained = family == "knn" and symbol == "BTCUSDT" and duration == "10m"
        return {"status": "shadow_active", "shadowPredictionReady": True} if trained else {"status": "untrained"}

    def fake_enqueue(**kwargs):
        calls.append(kwargs)
        family = kwargs["families"][0]
        symbol = kwargs["symbols"][0]
        duration = kwargs["durations"][0]
        return {
            "created": 1,
            "existing": 0,
            "reset": 0,
            "jobs": [{"job_id": f"{symbol}-{duration}-{family}", "status": JOB_STATUS_PENDING}],
        }

    monkeypatch.setattr(untrained_enqueue, "model_family_status", fake_status)
    monkeypatch.setattr(untrained_enqueue, "enqueue_model_search_jobs", fake_enqueue)

    payload = untrained_enqueue.enqueue_untrained_model_search_jobs(
        symbols=("BTCUSDT",),
        durations=("10m", "60m"),
        families=("knn", "svm"),
        profile="fast",
    )

    assert payload["trainedSkippedCount"] == 1
    assert len(payload["jobs"]) == 3
    assert {call["families"][0] for call in calls} == {"knn", "svm"}
    assert all(call["symbols"] == ("BTCUSDT",) for call in calls)
    assert ("knn", "10m") not in {(call["families"][0], call["durations"][0]) for call in calls}


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
    monkeypatch.setattr(
        runner,
        "model_family_status",
        lambda *_args: {"status": "untrained", "shadowPredictionReady": False},
    )


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    ensure_model_search_jobs_table(conn)
    return conn


def _db_path(name: str) -> Path:
    return _runtime_path(name) / "model-search.db"


def _runtime_path(name: str) -> Path:
    path = Path(tempfile.gettempdir()) / "incidentSpot-pytest-temp" / f"{name}-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resource_payload() -> dict:
    return {
        "resourceProfile": "local_safe",
        "internalThreads": 1,
        "parallelWorkers": 1,
        "xgboostProcessWorkers": 1,
    }


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
