from __future__ import annotations

import pandas as pd

from app.services import factor_combination_mined_inputs as mined_inputs


def test_combination_mined_inputs_only_builds_agent_candidates(monkeypatch) -> None:
    calls = []
    frame = pd.DataFrame({"open_time": [1], "close": [100.0]})
    agent_frame = frame.assign(agent_a=[0.1])
    candidate = object()
    rows = [_row("agent_a", score=3.0), _row("agent_b", score=2.0)]
    failures = ({"factorName": "agent_failed", "stage": "materialize_agent_factor"},)

    def materialize(frame_arg, *, rows, source_count, excluded_factor_names=None):
        calls.append(("materialize_agent", [row["factorName"] for row in rows], source_count, excluded_factor_names))
        return type("Result", (), {"frame": agent_frame, "source_count": 2, "failures": failures})()

    def build(frame_arg, *, symbol: str, duration: str, rows, excluded_factor_names=None):
        calls.append(("build_agent", symbol, duration, [row["factorName"] for row in rows], excluded_factor_names, frame_arg is agent_frame))
        return (candidate,)

    monkeypatch.setattr(mined_inputs, "agent_factor_rows_for_duration", lambda *_args: rows)
    monkeypatch.setattr(mined_inputs, "materialize_agent_factor_frame_for_rows", materialize)
    monkeypatch.setattr(mined_inputs, "build_agent_mined_candidates_from_rows", build)

    result = mined_inputs.build_mined_candidates(
        frame,
        symbol="BTCUSDT",
        duration="10m",
        excluded_factor_names={"blocked"},
    )

    assert calls == [
        ("materialize_agent", ["agent_a", "agent_b"], 2, {"blocked"}),
        ("build_agent", "BTCUSDT", "10m", ["agent_a", "agent_b"], {"blocked"}, True),
    ]
    assert result.frame is agent_frame
    assert result.source_count == 2
    assert result.failures == failures
    assert result.candidates == (candidate,)


def test_combination_mined_inputs_limits_agent_rows_by_stored_score(monkeypatch) -> None:
    captured = {}
    frame = pd.DataFrame({"open_time": [1], "close": [100.0]})
    rows = [
        _row("agent_low", score=1.0),
        _row("agent_high", score=9.0),
        _row("agent_mid", score=5.0),
    ]

    def materialize(frame_arg, *, rows, source_count, excluded_factor_names=None):
        captured["materialized"] = [row["factorName"] for row in rows]
        return type("Result", (), {"frame": frame_arg, "source_count": source_count, "failures": ()})()

    def build(_frame_arg, *, symbol: str, duration: str, rows, excluded_factor_names=None):
        captured["built"] = [row["factorName"] for row in rows]
        return ()

    monkeypatch.setattr(mined_inputs, "agent_factor_rows_for_duration", lambda *_args: rows)
    monkeypatch.setattr(mined_inputs, "materialize_agent_factor_frame_for_rows", materialize)
    monkeypatch.setattr(mined_inputs, "build_agent_mined_candidates_from_rows", build)

    result = mined_inputs.build_mined_candidates(
        frame,
        symbol="BTCUSDT",
        duration="10m",
        agent_factor_limit=1,
    )

    assert captured == {"materialized": ["agent_high"], "built": ["agent_high"]}
    assert result.source_count == 3


def _row(name: str, *, score: float) -> dict:
    return {
        "symbol": "BTCUSDT",
        "duration": "10m",
        "factorName": name,
        "factorDisplayName": name,
        "formula": "close",
        "candidateStatus": "promoted",
        "metrics": {"winRate": 0.6, "profitFactor": 1.2, "totalPeriods": 1300},
        "score": score,
    }
