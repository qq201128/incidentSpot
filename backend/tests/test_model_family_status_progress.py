from __future__ import annotations

from app.services import model_family_candidates as candidates
from app.services import model_family_status_progress as progress


def test_queued_progress_caps_completed_and_preserves_stage_counts(monkeypatch) -> None:
    monkeypatch.setattr(
        progress,
        "read_model_candidate_progress_view",
        lambda *_args, **_kwargs: {
            "status": "validation_failed",
            "completed": 648,
            "total": 1,
            "searchSpaceTotal": 1,
            "percent": 1.0,
        },
    )
    monkeypatch.setattr(
        progress,
        "list_model_search_jobs",
        lambda _filters: [_pending_job(search_space_total=432)],
    )

    payload = progress.candidate_search_progress("random_forest", "BTCUSDT", "10m", artifact_root=None)

    assert payload["status"] == "queued"
    assert payload["searchSpaceTotal"] == 432
    assert payload["completed"] == 432
    assert payload["total"] == 432
    assert payload["percent"] == 1.0
    assert payload["stageEvaluationCompleted"] == 648
    assert payload["stageEvaluationTotal"] == 1


def test_finish_progress_from_library_keeps_search_space_total_uninflated(monkeypatch) -> None:
    records = [
        {"status": "validation_failed", "profile": "full", "recordedAt": f"2026-06-02T00:00:0{i}+00:00"}
        for i in range(3)
    ]
    monkeypatch.setattr(candidates, "read_model_candidate_library", lambda *_args, **_kwargs: {"records": records})
    monkeypatch.setattr(candidates, "model_search_space_size", lambda _family: 2)
    monkeypatch.setattr(candidates, "update_json", lambda _path, updater: updater(None))

    payload = candidates.finish_model_candidate_progress_from_library(
        "random_forest",
        symbol="BTCUSDT",
        duration="10m",
        profile="full",
        parallel_workers=1,
        status="exhausted",
    )

    assert payload["completed"] == 3
    assert payload["total"] == 3
    assert payload["searchSpaceTotal"] == 2
    assert payload["percent"] == 1.0


def _pending_job(search_space_total: int) -> dict:
    return {
        "status": "pending",
        "profile": "full",
        "created_at": "2026-06-02T03:09:50+00:00",
        "parallel_workers": 1,
        "internal_threads": 1,
        "xgboost_process_workers": 1,
        "params": {"trainingRules": {"searchSpaceTotal": search_space_total}},
    }
