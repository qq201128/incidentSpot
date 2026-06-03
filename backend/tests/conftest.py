from __future__ import annotations

import pytest

from app.api import models as models_api
from app.services import model_search_api_worker


@pytest.fixture(autouse=True)
def reset_model_search_api_worker_state() -> None:
    model_search_api_worker.reset_api_model_search_worker_state()
    yield
    model_search_api_worker.reset_api_model_search_worker_state()


@pytest.fixture(autouse=True)
def default_api_model_search_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        models_api,
        "ensure_api_model_search_worker",
        lambda _resource: {"running": True, "managedByApi": True, "started": False},
    )


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
