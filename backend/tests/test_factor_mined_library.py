from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.services import factor_mined_candidates
from app.services import factor_mined_library

ROWS = 130


def test_good_combo_is_promoted_to_mined_factor_library(monkeypatch: pytest.MonkeyPatch) -> None:
    target = Path(__file__).resolve().parents[1] / "runtime" / "pytest-temp" / "mined-library-test.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)
    monkeypatch.setattr(factor_mined_library, "MINED_FACTOR_LIBRARY_PATH", target)
    try:
        promotion = factor_mined_library.upsert_good_combinations(_ranking_report())
        rows = factor_mined_library.mined_factor_rows_for_duration("BTCUSDT", "10m")
    finally:
        target.unlink(missing_ok=True)

    assert promotion["promoted"] == 1
    assert rows[0]["factorDisplayName"] == "组合：factor a + factor b"
    assert rows[0]["metrics"]["winRate"] == 0.63


def test_mined_factor_library_summary_recomputes_display_names(monkeypatch: pytest.MonkeyPatch) -> None:
    row = {
        "symbol": "BTCUSDT",
        "duration": "10m",
        "factorName": "combo__adx_14__combo__adx_14__ma_ratio_120__profit_factor_60__combo__adx_14__profit_factor_60__macd_signal",
        "factorDisplayName": "组合：ADX趋势强度（14周期） + 组合：ADX趋势强度（14周期） + 120周期均线偏离 + 60周期盈亏比",
        "members": [
            {"name": "adx_14", "orientation": 1},
            {
                "name": "combo__adx_14__ma_ratio_120__profit_factor_60",
                "orientation": 1,
            },
            {
                "name": "combo__adx_14__profit_factor_60__macd_signal",
                "orientation": 1,
            },
        ],
        "metrics": {"winRate": 0.7},
        "score": 1.0,
    }
    monkeypatch.setattr(
        factor_mined_library,
        "mined_factor_rows_for_duration",
        lambda *_args: [row],
    )

    summary = factor_mined_library.mined_factor_library_summary("BTCUSDT", "10m")

    assert summary["factors"][0]["factorDisplayName"] == "组合：ADX(14) + 均线偏离(120) + 盈亏比(60) + MACD"


def test_mined_factor_materialization_batches_dependent_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        _mined_row("combo_a_b", ["factor_a", "factor_b"]),
        _mined_row("combo_nested", ["combo_a_b", "factor_c"]),
    ]
    monkeypatch.setattr(factor_mined_candidates, "mined_factor_rows_for_duration", lambda *_args: rows)
    monkeypatch.setattr(
        factor_mined_candidates,
        "materialize_agent_factor_frame",
        lambda frame, **_kwargs: factor_mined_candidates.MinedFrameResult(frame, 0, ()),
    )

    with warnings.catch_warnings(record=True) as caught:
        result = factor_mined_candidates.materialize_mined_factor_frame(
            _learning_frame(),
            symbol="BTCUSDT",
            duration="10m",
        )

    assert not [item for item in caught if item.category is pd.errors.PerformanceWarning]
    assert result.failures == ()
    assert {"combo_a_b", "combo_nested"} <= set(result.frame.columns)
    expected = factor_mined_candidates.combination_score(
        result.frame,
        [{"name": "combo_a_b", "orientation": 1}, {"name": "factor_c", "orientation": 1}],
    )
    pd.testing.assert_series_equal(result.frame["combo_nested"], expected, check_names=False)


def _ranking_report() -> dict:
    return {
        "symbol": "BTCUSDT",
        "duration": "10m",
        "total": 1,
        "baseFactorCount": 3,
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


def _learning_frame() -> pd.DataFrame:
    base = np.linspace(-0.2, 0.2, ROWS)
    factor_a = base.copy()
    return pd.DataFrame({
        "open_time": np.arange(ROWS) * 60_000,
        "close": 100 + np.arange(ROWS) * 0.1,
        "factor_a": factor_a,
        "factor_b": factor_a * 0.9 + 0.01,
        "factor_c": -base,
    })


def _mined_row(name: str, members: list[str]) -> dict:
    return {
        "symbol": "BTCUSDT",
        "duration": "10m",
        "factorName": name,
        "factorDisplayName": name,
        "members": [{"name": member, "orientation": 1} for member in members],
    }
