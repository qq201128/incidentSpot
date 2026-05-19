from __future__ import annotations

from pathlib import Path

from app.services import factor_learning_refresh_tasks


def test_refresh_task_queue_creates_visible_status(monkeypatch) -> None:
    saved = []
    monkeypatch.setattr(factor_learning_refresh_tasks, "load_factor_learning_memory", lambda *_args: None)
    monkeypatch.setattr(
        factor_learning_refresh_tasks,
        "save_factor_learning_memory",
        lambda payload: saved.append(payload) or Path("memory.json"),
    )

    payload = factor_learning_refresh_tasks.mark_factor_learning_refresh_queued(
        "btcusdt",
        "10m",
        run_agent=False,
    )

    assert payload["symbol"] == "BTCUSDT"
    assert payload["refreshTask"]["status"] == "queued"
    assert payload["refreshTask"]["runAgent"] is False
    assert saved[0]["source"]["status"] == "queued"


def test_refresh_task_failure_persists_error_without_response_path(monkeypatch) -> None:
    saved = []
    memory = {"symbol": "BTCUSDT", "duration": "10m", "memoryPath": "response-only.json"}
    monkeypatch.setattr(factor_learning_refresh_tasks, "load_factor_learning_memory", lambda *_args: memory)
    monkeypatch.setattr(
        factor_learning_refresh_tasks,
        "save_factor_learning_memory",
        lambda payload: saved.append(payload) or Path("memory.json"),
    )

    payload = factor_learning_refresh_tasks.mark_factor_learning_refresh_failed(
        "btcusdt",
        "10m",
        "rank rebuild failed",
        run_agent=True,
    )

    assert saved[0]["refreshTask"]["status"] == "failed"
    assert saved[0]["refreshTask"]["error"] == "rank rebuild failed"
    assert saved[0]["refreshTask"]["runAgent"] is True
    assert "memoryPath" not in saved[0]
    assert payload["memoryPath"] == "memory.json"
