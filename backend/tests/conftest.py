from __future__ import annotations

import pytest

from app.api import models as models_api


@pytest.fixture(autouse=True)
def default_model_search_worker_status(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    if request.module.__name__ != "test_model_search_jobs":
        return
    monkeypatch.setattr(
        models_api,
        "model_search_queue_status",
        lambda _filters: {
            "workerStatus": {
                "state": "worker_required",
                "workerRequiredCommand": "python backend/scripts/run_model_search_worker.py --loop --adaptive-parallelism",
            }
        },
    )
