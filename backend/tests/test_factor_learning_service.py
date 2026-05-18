from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.services import factor_learning_service
from app.services import factor_mined_candidates
from app.services.factor_learning_core import build_factor_learning_memory
from app.services.factor_learning_signal_filter import apply_factor_learning_memory

ROWS = 130
PREDICTION_START = 100
PREDICTION_COUNT = 16


def test_factor_learning_memory_records_loss_patterns_and_weights() -> None:
    frame = _learning_frame()
    memory = build_factor_learning_memory(
        frame,
        _ranking_report(),
        _settled_predictions(),
        symbol="btcusdt",
        duration="10m",
        settlement_sweep={"checked": PREDICTION_COUNT, "settled": 0, "pendingData": 0},
    )

    loss_patterns = memory["lossMemory"]["patterns"]
    assert memory["symbol"] == "BTCUSDT"
    assert memory["lossMemory"]["status"] == "learned"
    assert any(pattern["feature"] == "factor_a" for pattern in loss_patterns)
    assert memory["factorMining"]["successPatterns"]
    assert memory["factorMining"]["operatorLibrary"]["total"] >= 60
    assert memory["weights"]["factor_a"] > 0
    assert memory["retrieval"]["blockedFactorNames"]
    assert memory["retrieval"]["lossPatterns"]
    assert memory["retrieval"]["topWeights"]
    assert memory["adaptiveLearning"]["algorithmCount"] >= 10
    assert memory["adaptiveLearning"]["overallAccuracy"] == 0.5
    assert memory["minedFactorLibrary"] == {}
    assert memory["monitoring"] == {}


def test_factor_learning_filter_blocks_remembered_loss_feature() -> None:
    frame = _learning_frame()
    memory = build_factor_learning_memory(
        frame,
        _ranking_report(),
        _settled_predictions(),
        symbol="BTCUSDT",
        duration="10m",
    )
    payload = _live_payload()

    result = apply_factor_learning_memory(payload, frame, frame.index[-1], memory)

    assert result["factorLearning"]["state"] == "active"
    assert result["factorLearning"]["filterPassed"] is False
    assert result["factorLearning"]["lossPatternMatches"]
    assert result["qualityPassed"] is False
    assert result["qualityGateReason"] == "factor_learning_filter_blocked"


def test_factor_learning_filter_blocks_forbidden_region_member() -> None:
    frame = _learning_frame()
    memory = build_factor_learning_memory(
        frame,
        _ranking_report(),
        _settled_predictions(),
        symbol="BTCUSDT",
        duration="10m",
    )
    memory["lossMemory"]["patterns"] = []
    memory["factorMining"]["forbiddenRegions"] = [{"members": ["factor_b"]}]
    payload = _live_payload()

    result = apply_factor_learning_memory(payload, frame, frame.index[-1], memory)

    assert result["factorLearning"]["state"] == "active"
    assert result["factorLearning"]["blockedMembers"]
    assert any(item["feature"] == "factor_b" for item in result["factorLearning"]["blockedMembers"])
    assert result["factorLearning"]["filterPassed"] is False
    assert result["qualityPassed"] is False


def test_refresh_rebuilds_stale_combination_ranking(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    saved = []
    frame = _learning_frame()

    monkeypatch.setattr(factor_learning_service, "load_factor_frame", lambda *_args: frame)
    monkeypatch.setattr(factor_learning_service, "load_factor_learning_memory", lambda *_args: None)
    monkeypatch.setattr(
        factor_learning_service,
        "get_cached_combination_ranking",
        lambda *_args: {"cacheStatus": {"usable": False, "reason": "market_data_changed"}},
    )
    monkeypatch.setattr(
        factor_learning_service,
        "run_factor_combination_ranking_on_frame",
        lambda frame_arg, **kwargs: calls.append(("rank", frame_arg is frame, kwargs["duration"])) or _ranking_report(),
    )
    monkeypatch.setattr(
        factor_learning_service,
        "save_cached_combination_ranking",
        lambda report: calls.append(("cache", report["symbol"], report["duration"])),
    )
    monkeypatch.setattr(
        factor_learning_service,
        "settle_due_predictions",
        lambda *_args: {"checked": 0, "settled": 0, "pendingData": 0},
    )
    monkeypatch.setattr(factor_learning_service, "_settled_factor_combo_predictions", lambda *_args: [])
    monkeypatch.setattr(
        factor_learning_service,
        "materialize_mined_factor_frame",
        lambda frame_arg, **_kwargs: factor_mined_candidates.MinedFrameResult(frame_arg, 0, ()),
    )
    monkeypatch.setattr(factor_learning_service, "mined_factor_library_summary", lambda *_args: {})
    monkeypatch.setattr(factor_learning_service, "agent_mined_factor_library_summary", lambda *_args: {})
    monkeypatch.setattr(factor_learning_service, "factor_combo_monitor_report", lambda *_args: {})
    monkeypatch.setattr(factor_learning_service, "lstm_shadow_learning_summary", lambda *_args: {})
    monkeypatch.setattr(
        factor_learning_service,
        "save_factor_learning_memory",
        lambda payload: saved.append(payload) or Path("memory.json"),
    )

    payload = factor_learning_service.refresh_factor_learning_memory("btcusdt", "10m")

    assert payload["source"]["rankingRefreshSource"] == "rebuilt_cache"
    assert payload["memoryPath"] == "memory.json"
    assert calls == [("rank", True, "10m"), ("cache", "BTCUSDT", "10m")]
    assert saved[0]["source"]["rankingRefreshSource"] == "rebuilt_cache"


def test_factor_learning_agent_failure_is_written_to_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    saved = []
    memory = {"symbol": "BTCUSDT", "duration": "10m", "updatedAt": "before"}

    def fake_attach(_memory: dict) -> dict:
        raise RuntimeError("Kimi request failed")

    monkeypatch.setattr(factor_learning_service, "load_factor_learning_memory", lambda *_args: memory)
    monkeypatch.setattr(factor_learning_service, "attach_llm_agent_review", fake_attach)
    monkeypatch.setattr(
        factor_learning_service,
        "save_factor_learning_memory",
        lambda payload: saved.append(payload) or Path("memory.json"),
    )

    with pytest.raises(RuntimeError, match="Kimi request failed"):
        factor_learning_service.run_factor_learning_llm_agent("BTCUSDT", "10m")

    assert saved[0]["llmAgent"]["status"] == "failed"
    assert saved[0]["llmAgent"]["error"] == "Kimi request failed"
    assert "llmAgent" not in memory


def test_pending_agent_status_does_not_persist_response_path(monkeypatch: pytest.MonkeyPatch) -> None:
    saved = []
    memory = {"symbol": "BTCUSDT", "duration": "10m", "memoryPath": "response-only.json"}
    monkeypatch.setattr(
        factor_learning_service,
        "save_factor_learning_memory",
        lambda payload: saved.append(payload) or Path("memory.json"),
    )

    payload = factor_learning_service.mark_factor_learning_agent_pending(memory)

    assert saved[0]["llmAgent"]["status"] == "pending"
    assert "memoryPath" not in saved[0]
    assert payload["memoryPath"] == "memory.json"


def _learning_frame() -> pd.DataFrame:
    base = np.linspace(-0.2, 0.2, ROWS)
    factor_a = base.copy()
    factor_a[PREDICTION_START:PREDICTION_START + 8] = -2.0
    factor_a[PREDICTION_START + 8:PREDICTION_START + PREDICTION_COUNT] = 4.0
    factor_a[-1] = 4.0
    factor_b = factor_a * 0.9 + 0.01
    factor_c = -base
    return pd.DataFrame({
        "open_time": np.arange(ROWS) * 60_000,
        "close": 100 + np.arange(ROWS) * 0.1,
        "factor_a": factor_a,
        "factor_b": factor_b,
        "factor_c": factor_c,
    })


def _settled_predictions() -> list[dict]:
    rows = []
    for offset in range(PREDICTION_COUNT):
        is_loss = offset >= 8
        rows.append({
            "open_time": (PREDICTION_START + offset) * 60_000,
            "direction": "up",
            "confidence": 0.62,
            "actual_return": -0.01 if is_loss else 0.01,
            "prediction_correct": 0 if is_loss else 1,
            "high_winrate_rule": "combo__factor_a__factor_b",
        })
    return rows


def _ranking_report() -> dict:
    return {
        "symbol": "BTCUSDT",
        "duration": "10m",
        "total": 1,
        "baseFactorCount": 3,
        "baseFactors": [
            _base_factor("factor_a", "return", win_rate=0.62, ir=0.80, sharpe=1.20),
            _base_factor("factor_b", "return", win_rate=0.58, ir=0.40, sharpe=0.80),
            _base_factor("factor_c", "momentum", win_rate=0.49, ir=-0.10, sharpe=0.90),
        ],
        "ranking": [{
            "factorName": "combo__factor_a__factor_b",
            "factorDisplayName": "组合：A + B",
            "winRate": 0.63,
            "profitFactor": 1.12,
            "totalPeriods": ROWS,
            "members": [
                {"name": "factor_a", "category": "return", "orientation": 1, "singleWinRate": 0.62},
                {"name": "factor_b", "category": "return", "orientation": 1, "singleWinRate": 0.58},
            ],
        }],
    }


def _base_factor(name: str, category: str, *, win_rate: float, ir: float, sharpe: float) -> dict:
    return {
        "name": name,
        "category": category,
        "formula": f"{name}.rolling(10).mean()",
        "sourceFile": "test.py",
        "winRate": win_rate,
        "ir": ir,
        "sharpe": sharpe,
        "profitFactor": 1.10,
    }


def _live_payload() -> dict:
    return {
        "symbol": "BTCUSDT",
        "duration": "10m",
        "factorName": "combo__factor_a__factor_b",
        "factorDisplayName": "组合：A + B",
        "members": [
            {"name": "factor_a", "orientation": 1},
            {"name": "factor_b", "orientation": 1},
        ],
        "direction": "up",
        "probabilityUp": 0.62,
        "confidence": 0.62,
        "score": 1.0,
        "source": "factor_combination_ranking",
        "method": "test",
        "historicalWinRate": 0.63,
        "qualityPassed": True,
        "qualityMinWinRate": 0.5,
        "frameIndex": str(ROWS - 1),
    }
