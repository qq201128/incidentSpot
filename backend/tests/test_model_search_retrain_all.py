from __future__ import annotations

import pytest

from app.api import models as models_api
from app.services import model_search_untrained_enqueue as untrained_enqueue
from app.services.model_search_job_types import JOB_STATUS_PENDING


def test_retrain_enqueue_includes_trained_targets_when_reset_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    monkeypatch.setattr(
        untrained_enqueue,
        "model_family_status",
        lambda *_args: {"status": "shadow_active", "shadowPredictionReady": True},
    )
    monkeypatch.setattr(
        untrained_enqueue,
        "enqueue_model_search_jobs",
        lambda **kwargs: calls.append(kwargs) or _queued_payload("job-1"),
    )

    payload = untrained_enqueue.enqueue_untrained_model_search_jobs(
        symbols=("BTCUSDT",),
        durations=("10m",),
        families=("knn",),
        profile="fast",
        reset_existing=True,
        reset_history=True,
    )

    assert payload["trainedSkippedCount"] == 0
    assert payload["created"] == 1
    assert calls[0]["reset_existing"] is True
    assert calls[0]["reset_history"] is True


def test_retrain_all_api_queues_selected_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    worker_calls = []
    monkeypatch.setattr(
        models_api,
        "enqueue_untrained_model_search_jobs",
        lambda **kwargs: calls.append(kwargs) or _queued_payload("job-1"),
    )

    def fake_queue_status(filters: dict, **_kwargs: object) -> dict:
        return {
            "workerStatus": {
                "state": "queued",
                "workerRequiredCommand": "python backend/scripts/run_model_search_worker.py --loop --adaptive-parallelism",
                "filters": filters,
            }
        }

    monkeypatch.setattr(models_api, "model_search_queue_status", fake_queue_status)
    monkeypatch.setattr(models_api, "ensure_api_model_search_worker", lambda resource: worker_calls.append(resource))

    response = models_api.model_search_retrain_all(
        symbols="btcusdt",
        durations="10m",
        families="knn,svm",
        profile="fast",
        reset_history=True,
        internal_threads=2,
        parallel_workers=1,
        xgboost_process_workers=1,
    )

    assert response["targets"] == {
        "symbols": ["BTCUSDT"],
        "durations": ["10m"],
        "families": ["knn", "svm"],
    }
    assert calls[0]["reset_existing"] is True
    assert calls[0]["reset_history"] is True
    assert calls[0]["resource"]["internalThreads"] == 2
    assert worker_calls[0]["internalThreads"] == 2
    assert "正在执行队列" in response["message"]


def _queued_payload(job_id: str) -> dict:
    return {
        "jobs": [{"job_id": job_id, "status": JOB_STATUS_PENDING}],
        "created": 1,
        "existing": 0,
        "reset": 0,
    }
