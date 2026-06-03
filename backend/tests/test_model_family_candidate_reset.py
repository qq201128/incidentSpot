from __future__ import annotations

from app.services.lstm_artifacts import read_json, write_json
from app.services.model_family_candidate_reset import reset_model_candidate_history
from app.services.model_family_candidates import candidate_library_path, candidate_progress_path


def test_reset_model_candidate_history_clears_library_and_progress(tmp_path) -> None:
    library_path = candidate_library_path("knn", "BTCUSDT", "10m", artifact_root=tmp_path)
    progress_path = candidate_progress_path("knn", "BTCUSDT", "10m", artifact_root=tmp_path)
    write_json(library_path, {"records": [{"searchKey": "old"}], "total": 1})
    write_json(progress_path, {"status": "running", "completed": 1, "total": 1})

    payload = reset_model_candidate_history("knn", "BTCUSDT", "10m", artifact_root=tmp_path)

    assert payload["library"]["records"] == []
    assert payload["progress"]["completed"] == 0
    assert read_json(library_path)["records"] == []
    assert read_json(progress_path)["status"] == "idle"
