from __future__ import annotations

import pytest

from app.api import models as models_api
from app.services.model_search_job_types import JOB_STATUS_PENDING


def test_candidate_search_api_returns_worker_required_command(monkeypatch: pytest.MonkeyPatch) -> None:
    queued = {
        "jobs": [{"job_id": "job-1", "status": JOB_STATUS_PENDING}],
        "created": 1,
        "existing": 0,
    }
    monkeypatch.setattr(models_api, "enqueue_model_search_jobs", lambda **_kwargs: queued)
    monkeypatch.setattr(
        models_api,
        "model_search_queue_status",
        lambda _filters: {
            "workerStatus": {
                "state": "worker_required",
                "workerRequiredCommand": "python backend/scripts/run_model_search_worker.py --loop",
            }
        },
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
    assert response["workerStatus"]["state"] == "worker_required"
    assert "run_model_search_worker.py --loop" in response["message"]
