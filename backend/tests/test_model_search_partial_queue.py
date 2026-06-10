from __future__ import annotations

import sqlite3
import tempfile
import uuid
from pathlib import Path

import pytest

from app.services import model_search_job_store as store
from app.services.model_search_job_schema import ensure_model_search_jobs_table
from app.services.model_search_job_types import JOB_STATUS_PENDING


def test_partial_retry_moves_job_behind_other_pending_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = _db_path("retry-move-to-back")
    monkeypatch.setattr(store, "get_conn", lambda: _connect(db_path))
    _enqueue_one("BTCUSDT")
    _enqueue_one("ETHUSDT")
    first = store.claim_next_model_search_job(max_running_jobs=1)
    partial = store.finish_model_search_job(
        first["job_id"],
        result=_batch_result(has_more=True),
        resource=_resource_payload(),
        artifact_path=None,
        log_path="runtime/partial.log",
    )

    retried = store.retry_failed_model_search_job(partial["job_id"], move_to_back=True)
    claimed = store.claim_next_model_search_job(max_running_jobs=1)

    assert retried["status"] == JOB_STATUS_PENDING
    assert claimed["symbol"] == "ETHUSDT"


def _enqueue_one(symbol: str) -> None:
    store.enqueue_model_search_jobs(
        symbols=(symbol,),
        durations=("10m",),
        families=("knn",),
        profile="fast",
    )


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    ensure_model_search_jobs_table(conn)
    return conn


def _db_path(name: str) -> Path:
    path = Path(tempfile.gettempdir()) / "incidentSpot-pytest-temp" / f"{name}-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path / "model-search.db"


def _resource_payload() -> dict:
    return {
        "resourceProfile": "local_safe",
        "internalThreads": 1,
        "parallelWorkers": 1,
        "xgboostProcessWorkers": 1,
    }


def _batch_result(*, has_more: bool) -> dict:
    return {
        "status": "partial_batch",
        "jobBatch": {
            "selectedCandidates": 1,
            "availableCandidatesBeforeJob": 3,
            "remainingCandidatesAfterJob": 2 if has_more else 0,
            "hasMoreCandidates": has_more,
        },
    }
