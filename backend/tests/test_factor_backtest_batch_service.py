from __future__ import annotations

import numpy as np
import pandas as pd

from app.services import factor_backtest_batch_service as batch
from app.services.factor_registry import FactorCategory, FactorDefinition, FactorDirection


def test_run_all_factor_backtests_includes_combo_factors(monkeypatch) -> None:
    frame = _frame()
    combo_row = {
        "symbol": "BTCUSDT",
        "duration": "10m",
        "factorName": "combo__factor_a__factor_b",
        "factorDisplayName": "组合：A + B",
        "formula": "expanding_oriented_zscore_mean_v1(factor_a, factor_b)",
        "members": [
            {"name": "factor_a", "orientation": 1},
            {"name": "factor_b", "orientation": 1},
        ],
    }
    monkeypatch.setattr(batch, "load_factor_frame", lambda *_args: frame)
    monkeypatch.setattr(batch, "list_single_factor_definitions", lambda **_kwargs: [_factor("factor_a")])
    monkeypatch.setattr(batch, "mined_factor_rows_for_duration", lambda *_args: [combo_row])

    report = batch.run_all_factor_backtests("BTCUSDT", durations=("10m",))
    names = {row["factorName"] for row in report["results"]}

    assert report["factorCount"] == 2
    assert report["failureCount"] == 0
    assert {"factor_a", "combo__factor_a__factor_b"} <= names


def _frame() -> pd.DataFrame:
    rows = 130
    idx = np.arange(rows, dtype=float)
    close = 100 + np.cumsum(np.sin(idx / 8.0) * 0.2 + 0.1)
    return pd.DataFrame(
        {
            "open_time": np.arange(rows) * 60_000,
            "close": close,
            "factor_a": np.sin(idx / 7.0),
            "factor_b": np.cos(idx / 11.0),
        }
    )


def _factor(name: str) -> FactorDefinition:
    return FactorDefinition(
        name=name,
        category=FactorCategory.RETURN,
        description=name,
        formula=name,
        direction=FactorDirection.HIGHER_BETTER,
    )
