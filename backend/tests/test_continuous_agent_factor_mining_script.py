from __future__ import annotations

import sys

import pytest

from scripts import continuous_agent_factor_mining as mining_script


def test_agent_mining_script_runs_single_cycle(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    calls = _patch_successful_cycle(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["continuous_agent_factor_mining.py", "btcusdt", "--compact"])

    assert mining_script.main() == 0

    captured = capsys.readouterr()
    assert calls == [
        ("init_db",),
        ("ingest", "BTCUSDT", "5m", 500, ("10m",)),
        ("refresh_queued", "BTCUSDT", "10m", True),
        ("agent_pending", "BTCUSDT", "10m"),
        ("refresh_running", "BTCUSDT", "10m", True),
        ("refresh_memory", "BTCUSDT", "10m", False, 30),
        ("refresh_completed", True),
        ("agent_running", "BTCUSDT", "10m"),
        ("run_agent", "BTCUSDT", "10m", 30),
    ]
    assert '"factorLookbackDays": 30' in captured.out
    assert '"agentStatus": "completed"' in captured.out
    assert '"promotedCount": 1' in captured.out


def test_agent_mining_script_loop_sleeps_between_cycles(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_successful_cycle(monkeypatch)
    sleeps = []
    monkeypatch.setattr(
        sys,
        "argv",
        ["continuous_agent_factor_mining.py", "btcusdt", "--loop", "--poll-seconds", "0.01"],
    )

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        raise KeyboardInterrupt

    monkeypatch.setattr(mining_script.time, "sleep", fake_sleep)

    with pytest.raises(KeyboardInterrupt):
        mining_script.main()

    assert sleeps == [0.01]


def test_agent_mining_script_marks_failed_and_surfaces_error(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(mining_script, "init_db", lambda: calls.append(("init_db",)))
    monkeypatch.setattr(mining_script, "ingest_market_context_data", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        mining_script,
        "mark_factor_learning_refresh_queued",
        _fake_mark_refresh_queued(calls),
    )
    monkeypatch.setattr(
        mining_script,
        "mark_factor_learning_refresh_running",
        lambda symbol, duration, *, run_agent: calls.append(("refresh_running", symbol, duration, run_agent)),
    )
    monkeypatch.setattr(
        mining_script,
        "mark_factor_learning_refresh_failed",
        lambda symbol, duration, error, *, run_agent: calls.append(("refresh_failed", symbol, duration, error)),
    )
    monkeypatch.setattr(mining_script, "mark_factor_learning_agent_pending", lambda memory: memory)
    monkeypatch.setattr(mining_script, "mark_factor_learning_agent_running", lambda memory: memory)
    monkeypatch.setattr(
        mining_script,
        "mark_factor_learning_agent_failed",
        lambda symbol, duration, error: calls.append(("agent_failed", symbol, duration, error)),
    )
    monkeypatch.setattr(sys, "argv", ["continuous_agent_factor_mining.py", "btcusdt"])

    def fail_refresh(*_args, **_kwargs):
        raise RuntimeError("ranking cache crashed")

    monkeypatch.setattr(mining_script, "refresh_factor_learning_memory", fail_refresh)

    with pytest.raises(RuntimeError, match="stage=refresh_memory"):
        mining_script.main()

    assert calls[-2][0] == "refresh_failed"
    assert calls[-1][0] == "agent_failed"
    assert "ranking cache crashed" in calls[-1][3]


def test_selected_durations_rejects_unsupported_duration() -> None:
    with pytest.raises(ValueError, match="unsupported durations"):
        mining_script._selected_durations("10m,2h")


def test_full_factor_history_disables_agent_window(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_successful_cycle(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        ["continuous_agent_factor_mining.py", "btcusdt", "--full-factor-history", "--compact"],
    )

    assert mining_script.main() == 0

    assert ("refresh_memory", "BTCUSDT", "10m", False, None) in calls
    assert ("run_agent", "BTCUSDT", "10m", None) in calls


def _patch_successful_cycle(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    calls = []
    monkeypatch.setattr(mining_script, "init_db", lambda: calls.append(("init_db",)))

    def fake_ingest(symbol: str, *, period: str, limit: int, durations: tuple[str, ...]) -> dict:
        calls.append(("ingest", symbol, period, limit, durations))
        return {"symbol": symbol, "positioningRows": 1}

    monkeypatch.setattr(mining_script, "ingest_market_context_data", fake_ingest)
    monkeypatch.setattr(
        mining_script,
        "mark_factor_learning_refresh_queued",
        _fake_mark_refresh_queued(calls),
    )
    monkeypatch.setattr(
        mining_script,
        "mark_factor_learning_refresh_running",
        lambda symbol, duration, *, run_agent: calls.append(("refresh_running", symbol, duration, run_agent)),
    )
    monkeypatch.setattr(
        mining_script,
        "mark_factor_learning_refresh_completed",
        lambda memory, *, run_agent: calls.append(("refresh_completed", run_agent)) or memory,
    )
    monkeypatch.setattr(
        mining_script,
        "mark_factor_learning_agent_pending",
        lambda memory: calls.append(("agent_pending", memory["symbol"], memory["duration"])) or memory,
    )
    monkeypatch.setattr(
        mining_script,
        "mark_factor_learning_agent_running",
        lambda memory: calls.append(("agent_running", memory["symbol"], memory["duration"])) or memory,
    )
    monkeypatch.setattr(mining_script, "refresh_factor_learning_memory", _fake_refresh(calls))
    monkeypatch.setattr(mining_script, "run_factor_learning_llm_agent", _fake_run_agent(calls))
    return calls


def _fake_refresh(calls: list[tuple]):
    def refresh(
        symbol: str,
        duration: str,
        *,
        run_llm_agent: bool,
        factor_lookback_days: int | None,
    ) -> dict:
        calls.append(("refresh_memory", symbol, duration, run_llm_agent, factor_lookback_days))
        return {"symbol": symbol, "duration": duration, "memoryPath": "memory.json"}

    return refresh


def _fake_mark_refresh_queued(calls: list[tuple]):
    def mark(symbol: str, duration: str, *, run_agent: bool) -> dict:
        calls.append(("refresh_queued", symbol, duration, run_agent))
        return {"symbol": symbol, "duration": duration}

    return mark


def _fake_run_agent(calls: list[tuple]):
    def run_agent(symbol: str, duration: str, *, factor_lookback_days: int | None) -> dict:
        calls.append(("run_agent", symbol, duration, factor_lookback_days))
        return {
            "symbol": symbol,
            "duration": duration,
            "memoryPath": "memory.json",
            "llmAgent": {"status": "completed", "model": "model", "usage": {"total_tokens": 3}},
            "agentCandidateEvaluation": {"generatedCount": 2, "promotedCount": 1, "failedCount": 1},
            "agentMinedFactorLibrary": {"total": 5, "simulationEligibleTotal": 3},
        }

    return run_agent
