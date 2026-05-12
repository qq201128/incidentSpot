from __future__ import annotations

import numpy as np
import pandas as pd

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
    assert memory["weights"]["factor_a"] > 0


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
            _base_factor("factor_a", "return", 0.62, 0.80, 1.20),
            _base_factor("factor_b", "return", 0.58, 0.40, 0.80),
            _base_factor("factor_c", "momentum", 0.49, -0.10, 0.90),
        ],
        "ranking": [{
            "factorName": "combo__factor_a__factor_b",
            "winRate": 0.63,
            "members": [
                {"name": "factor_a", "category": "return", "orientation": 1, "singleWinRate": 0.62},
                {"name": "factor_b", "category": "return", "orientation": 1, "singleWinRate": 0.58},
            ],
        }],
    }


def _base_factor(name: str, category: str, win_rate: float, ir: float, sharpe: float) -> dict:
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
