from __future__ import annotations

from fastapi import BackgroundTasks

from app.api import factor_learning


def test_factor_learning_refresh_queues_agent(monkeypatch) -> None:
    calls = []

    def fake_refresh(symbol: str, duration: str, *, run_llm_agent: bool) -> dict:
        calls.append(("refresh", symbol, duration, run_llm_agent))
        return {"symbol": symbol, "duration": duration, "updatedAt": "now"}

    def fake_queued(symbol: str, duration: str) -> dict:
        calls.append(("queued", symbol, duration, None))
        return {"symbol": symbol, "duration": duration, "llmAgent": {"status": "pending"}}

    def fake_pending(memory: dict) -> dict:
        calls.append(("pending", memory["symbol"], memory["duration"], None))
        return {**memory, "llmAgent": {"status": "pending"}}

    def fake_agent(symbol: str, duration: str) -> dict:
        calls.append(("agent", symbol, duration, None))
        return {"symbol": symbol, "duration": duration}

    monkeypatch.setattr(factor_learning, "refresh_factor_learning_memory", fake_refresh)
    monkeypatch.setattr(factor_learning, "mark_factor_learning_refresh_queued", fake_queued)
    monkeypatch.setattr(factor_learning, "mark_factor_learning_agent_pending", fake_pending)
    monkeypatch.setattr(factor_learning, "run_factor_learning_llm_agent", fake_agent)

    tasks = BackgroundTasks()

    response = factor_learning.factor_learning_refresh(
        tasks,
        symbol="btcusdt",
        duration="10m",
        run_agent=True,
    )

    assert calls == [("queued", "BTCUSDT", "10m", None)]
    tasks.tasks[0].func(*tasks.tasks[0].args, **tasks.tasks[0].kwargs)

    assert response["agentQueued"] is True
    assert response["llmAgent"]["status"] == "pending"
    assert calls == [
        ("queued", "BTCUSDT", "10m", None),
        ("refresh", "BTCUSDT", "10m", False),
        ("pending", "BTCUSDT", "10m", None),
        ("agent", "BTCUSDT", "10m", None),
    ]


def test_background_agent_refresh_marks_failure(monkeypatch) -> None:
    calls = []

    def fake_refresh(symbol: str, duration: str, *, run_llm_agent: bool) -> dict:
        calls.append(("refresh", symbol, duration, run_llm_agent))
        raise RuntimeError("network stalled")

    def fake_failed(symbol: str, duration: str, error: str) -> dict:
        calls.append(("failed", symbol, duration, error))
        return {"symbol": symbol, "duration": duration}

    monkeypatch.setattr(factor_learning, "refresh_factor_learning_memory", fake_refresh)
    monkeypatch.setattr(factor_learning, "mark_factor_learning_agent_failed", fake_failed)

    factor_learning._background_factor_learning_refresh_and_agent("BTCUSDT", "10m")

    assert calls == [
        ("refresh", "BTCUSDT", "10m", False),
        ("failed", "BTCUSDT", "10m", "network stalled"),
    ]


def test_factor_learning_refresh_local_stays_synchronous(monkeypatch) -> None:
    calls = []

    def fake_refresh(symbol: str, duration: str, *, run_llm_agent: bool) -> dict:
        calls.append((symbol, duration, run_llm_agent))
        return {"symbol": symbol, "duration": duration}

    monkeypatch.setattr(factor_learning, "refresh_factor_learning_memory", fake_refresh)

    tasks = BackgroundTasks()

    response = factor_learning.factor_learning_refresh(
        tasks,
        symbol="btcusdt",
        duration="10m",
        run_agent=False,
    )

    assert response["agentQueued"] is False
    assert tasks.tasks == []
    assert calls == [("BTCUSDT", "10m", False)]
