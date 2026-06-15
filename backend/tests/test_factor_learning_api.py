from __future__ import annotations

import pytest
from fastapi import BackgroundTasks

from app.api import factor_learning
from app.api.factor_learning import FactorLearningRefreshJob
from app.services.background_loop_status import background_loop_statuses, reset_background_loop_statuses


def test_factor_learning_refresh_queues_agent(monkeypatch) -> None:
    reset_background_loop_statuses()
    calls = []
    _patch_background_hooks(monkeypatch, calls)

    tasks = BackgroundTasks()
    response = factor_learning.factor_learning_refresh(
        tasks,
        symbol="btcusdt",
        duration="10m",
        run_agent=True,
    )

    assert response["agentQueued"] is True
    assert response["refreshQueued"] is True
    assert response["refreshTask"]["lookbackDays"] == factor_learning.DEFAULT_FACTOR_LEARNING_REFRESH_LOOKBACK_DAYS
    assert response["llmAgent"]["status"] == "pending"
    assert calls == [("queued", "BTCUSDT", "10m", True, 30), ("agent_pending", "BTCUSDT", "10m", None, None)]
    assert isinstance(tasks.tasks[0].args[0], FactorLearningRefreshJob)

    tasks.tasks[0].func(*tasks.tasks[0].args, **tasks.tasks[0].kwargs)

    assert calls == [
        ("queued", "BTCUSDT", "10m", True, 30),
        ("agent_pending", "BTCUSDT", "10m", None, None),
        ("running", "BTCUSDT", "10m", True, 30),
        ("refresh", "BTCUSDT", "10m", False, 30),
        ("completed", "BTCUSDT", "10m", True),
        ("agent_running", "BTCUSDT", "10m", None, None),
        ("agent", "BTCUSDT", "10m", 30),
    ]
    status = background_loop_statuses()["factor_learning_refresh"]
    assert status["status"] == "passed"
    assert status["lastSuccessDetails"] == {
        "symbol": "BTCUSDT",
        "duration": "10m",
        "runAgent": True,
        "factorLookbackDays": 30,
        "stage": "completed",
    }


def test_factor_learning_refresh_local_queues_background(monkeypatch) -> None:
    reset_background_loop_statuses()
    calls = []
    _patch_background_hooks(monkeypatch, calls)

    tasks = BackgroundTasks()
    response = factor_learning.factor_learning_refresh(
        tasks,
        symbol="btcusdt",
        duration="10m",
        run_agent=False,
    )

    assert response["agentQueued"] is False
    assert response["refreshQueued"] is True
    assert response["refreshTask"]["status"] == "queued"
    assert len(tasks.tasks) == 1
    assert calls == [("queued", "BTCUSDT", "10m", False, 30)]
    assert isinstance(tasks.tasks[0].args[0], FactorLearningRefreshJob)

    tasks.tasks[0].func(*tasks.tasks[0].args, **tasks.tasks[0].kwargs)

    assert calls == [
        ("queued", "BTCUSDT", "10m", False, 30),
        ("running", "BTCUSDT", "10m", False, 30),
        ("refresh", "BTCUSDT", "10m", False, 30),
        ("completed", "BTCUSDT", "10m", False),
    ]


def test_background_agent_refresh_marks_failure(monkeypatch) -> None:
    reset_background_loop_statuses()
    calls = []

    def fake_refresh(
        symbol: str,
        duration: str,
        *,
        run_llm_agent: bool,
        factor_lookback_days: int | None = None,
    ) -> dict:
        calls.append(("refresh", symbol, duration, run_llm_agent))
        raise RuntimeError("network stalled")

    monkeypatch.setattr(factor_learning, "refresh_factor_learning_memory", fake_refresh)
    monkeypatch.setattr(
        factor_learning,
        "mark_factor_learning_refresh_running",
        lambda symbol, duration, *, run_agent, lookback_days=None: calls.append(("running", symbol, duration, run_agent)),
    )
    monkeypatch.setattr(
        factor_learning,
        "mark_factor_learning_refresh_failed",
        lambda symbol, duration, error, *, run_agent, lookback_days=None: calls.append(("refresh_failed", symbol, duration, error)),
    )
    monkeypatch.setattr(
        factor_learning,
        "mark_factor_learning_agent_failed",
        lambda symbol, duration, error: calls.append(("agent_failed", symbol, duration, error)),
    )

    job = FactorLearningRefreshJob("BTCUSDT", "10m", True)
    try:
        factor_learning._background_factor_learning_refresh(job)
    except RuntimeError as exc:
        assert str(exc) == "network stalled"
    else:
        raise AssertionError("factor learning background failure was swallowed")

    assert calls == [
        ("running", "BTCUSDT", "10m", True),
        ("refresh", "BTCUSDT", "10m", False),
        ("refresh_failed", "BTCUSDT", "10m", "network stalled"),
        ("agent_failed", "BTCUSDT", "10m", "network stalled"),
    ]
    status = background_loop_statuses()["factor_learning_refresh"]
    assert status["status"] == "failed"
    assert status["lastError"] == "network stalled"
    assert status["lastExceptionType"] == "RuntimeError"
    assert status["lastFailureDetails"] == {
        "symbol": "BTCUSDT",
        "duration": "10m",
        "runAgent": True,
        "factorLookbackDays": 30,
        "stage": "refresh_memory",
    }


def test_background_agent_stage_failure_keeps_refresh_completed(monkeypatch) -> None:
    reset_background_loop_statuses()
    calls = []

    monkeypatch.setattr(
        factor_learning,
        "mark_factor_learning_refresh_running",
        lambda symbol, duration, *, run_agent, lookback_days=None: calls.append(
            ("running", symbol, duration, run_agent, lookback_days)
        ),
    )
    monkeypatch.setattr(
        factor_learning,
        "refresh_factor_learning_memory",
        lambda symbol, duration, *, run_llm_agent, factor_lookback_days=None: calls.append(
            ("refresh", symbol, duration, run_llm_agent, factor_lookback_days)
        )
        or {"symbol": symbol, "duration": duration},
    )
    monkeypatch.setattr(
        factor_learning,
        "mark_factor_learning_refresh_completed",
        lambda memory, *, run_agent, lookback_days=None: calls.append(
            ("completed", memory["symbol"], memory["duration"], run_agent, lookback_days)
        )
        or {**memory, "refreshTask": {"status": "completed", "runAgent": run_agent}},
    )
    monkeypatch.setattr(
        factor_learning,
        "mark_factor_learning_agent_running",
        lambda memory: calls.append(("agent_running", memory["symbol"], memory["duration"], None, None)) or memory,
    )

    def fail_agent(symbol: str, duration: str, *, factor_lookback_days: int | None = None) -> dict:
        calls.append(("agent", symbol, duration, factor_lookback_days))
        raise RuntimeError("llm network stalled")

    monkeypatch.setattr(factor_learning, "run_factor_learning_llm_agent", fail_agent)
    monkeypatch.setattr(
        factor_learning,
        "mark_factor_learning_refresh_failed",
        lambda symbol, duration, error, *, run_agent, lookback_days=None: calls.append(
            ("refresh_failed", symbol, duration, error, lookback_days)
        ),
    )
    monkeypatch.setattr(
        factor_learning,
        "mark_factor_learning_agent_failed",
        lambda symbol, duration, error: calls.append(("agent_failed", symbol, duration, error, None)),
    )

    job = FactorLearningRefreshJob("BTCUSDT", "10m", True, 30)
    with pytest.raises(RuntimeError, match="llm network stalled"):
        factor_learning._background_factor_learning_refresh(job)

    assert ("refresh_failed", "BTCUSDT", "10m", "llm network stalled", 30) not in calls
    assert calls[-1] == ("agent_failed", "BTCUSDT", "10m", "llm network stalled", None)
    status = background_loop_statuses()["factor_learning_refresh"]
    assert status["lastFailureDetails"]["stage"] == "llm_agent"


def _patch_background_hooks(monkeypatch, calls: list) -> None:
    monkeypatch.setattr(
        factor_learning,
        "mark_factor_learning_refresh_queued",
        lambda symbol, duration, *, run_agent, lookback_days=None: _task(
            calls,
            "queued",
            symbol,
            duration,
            run_agent,
            lookback_days,
        ),
    )
    monkeypatch.setattr(
        factor_learning,
        "mark_factor_learning_refresh_running",
        lambda symbol, duration, *, run_agent, lookback_days=None: _task(
            calls,
            "running",
            symbol,
            duration,
            run_agent,
            lookback_days,
        ),
    )
    monkeypatch.setattr(
        factor_learning,
        "mark_factor_learning_refresh_completed",
        lambda memory, *, run_agent, lookback_days=None: _completed(calls, memory, run_agent, lookback_days),
    )
    monkeypatch.setattr(
        factor_learning,
        "mark_factor_learning_agent_pending",
        lambda memory: _agent(calls, "agent_pending", memory),
    )
    monkeypatch.setattr(
        factor_learning,
        "mark_factor_learning_agent_running",
        lambda memory: _agent(calls, "agent_running", memory),
    )
    monkeypatch.setattr(
        factor_learning,
        "refresh_factor_learning_memory",
        lambda symbol, duration, *, run_llm_agent, factor_lookback_days=None: _refresh(
            calls,
            symbol,
            duration,
            run_llm_agent,
            factor_lookback_days,
        ),
    )
    monkeypatch.setattr(
        factor_learning,
        "run_factor_learning_llm_agent",
        lambda symbol, duration, *, factor_lookback_days=None: calls.append(
            ("agent", symbol, duration, factor_lookback_days)
        )
        or {},
    )


def _task(calls: list, action: str, symbol: str, duration: str, run_agent: bool, lookback_days: int | None) -> dict:
    calls.append((action, symbol, duration, run_agent, lookback_days))
    task = {"status": action, "runAgent": run_agent}
    if lookback_days is not None:
        task["lookbackDays"] = lookback_days
    return {"symbol": symbol, "duration": duration, "refreshTask": task}


def _completed(calls: list, memory: dict, run_agent: bool, lookback_days: int | None) -> dict:
    calls.append(("completed", memory["symbol"], memory["duration"], run_agent))
    return {**memory, "refreshTask": {"status": "completed", "runAgent": run_agent}}


def _agent(calls: list, action: str, memory: dict) -> dict:
    calls.append((action, memory["symbol"], memory["duration"], None, None))
    return {**memory, "llmAgent": {"status": action.removeprefix("agent_")}}


def _refresh(
    calls: list,
    symbol: str,
    duration: str,
    run_llm_agent: bool,
    factor_lookback_days: int | None,
) -> dict:
    calls.append(("refresh", symbol, duration, run_llm_agent, factor_lookback_days))
    return {"symbol": symbol, "duration": duration}
