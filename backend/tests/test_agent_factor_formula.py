from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services.agent_factor_formula import SUPPORTED_AGENT_FORMULA_FUNCTIONS, materialize_agent_formula
from app.services.factor_operator_library import factor_operator_prompt_payload


def test_agent_formula_supports_core_arithmetic_functions() -> None:
    frame = _frame()

    assert materialize_agent_formula(frame, "Abs(ret_1)").notna().all()
    assert set(materialize_agent_formula(frame, "Sign(ret_1)").dropna().unique()) <= {-1.0, 0.0, 1.0}
    assert materialize_agent_formula(frame, "Log(close)").notna().all()

    clipped = materialize_agent_formula(frame, "Clip(ret_1, -0.001, 0.001)")
    assert clipped.min() >= -0.001
    assert clipped.max() <= 0.001


def test_agent_formula_supports_time_series_windows() -> None:
    frame = _frame()

    formulas = [
        "Std(ret_1, 20)",
        "Sum(volume, 20)",
        "Min(low, 20)",
        "Max(high, 20)",
        "Delay(close, 5)",
        "PctChange(close, 5)",
        "SMA(close, 20)",
        "Corr(ret_1, volume, 20)",
    ]

    for formula in formulas:
        series = materialize_agent_formula(frame, formula)
        assert isinstance(series, pd.Series)
        assert series.index.equals(frame.index)
        assert series.notna().any()


def test_agent_formula_supports_ema_vwap_and_donchian() -> None:
    frame = _frame()

    formulas = [
        "EMA(close, 12)",
        "VWAP(close, volume, 20)",
        "VWAPDev(close, volume, 20)",
        "DonchianPos(close, 60)",
    ]

    for formula in formulas:
        series = materialize_agent_formula(frame, formula)
        assert isinstance(series, pd.Series)
        assert series.index.equals(frame.index)
        assert series.notna().any()


def test_agent_formula_rejects_invalid_window_and_unsupported_function() -> None:
    frame = _frame()

    with pytest.raises(ValueError, match="window must be greater than 1"):
        materialize_agent_formula(frame, "EMA(close, 1)")
    with pytest.raises(ValueError, match="PctChange window must be greater than 1"):
        materialize_agent_formula(frame, "PctChange(close, 1)")
    with pytest.raises(ValueError, match="unsupported formula function: TsRank"):
        materialize_agent_formula(frame, "TsRank(close, 20)")
    with pytest.raises(ValueError, match="formula column not found: missing"):
        materialize_agent_formula(frame, "EMA(missing, 12)")


def test_operator_prompt_only_exposes_executable_agent_functions() -> None:
    payload = factor_operator_prompt_payload()
    names = {item["name"] for item in payload["operators"]}

    assert names
    assert names <= SUPPORTED_AGENT_FORMULA_FUNCTIONS
    assert {"EMA", "VWAP", "VWAPDev", "DonchianPos", "Max", "Std", "PctChange"} <= names
    assert "TsRank" not in names


def test_operator_prompt_exposes_pct_change_window_constraint() -> None:
    payload = factor_operator_prompt_payload()
    pct_change = _operator_by_name(payload, "PctChange")

    assert "PctChange(x, 1) is invalid" in " ".join(payload["formulaRules"])
    assert "PctChange(x, 1) is invalid" in " ".join(pct_change["constraints"])


def _operator_by_name(payload: dict, name: str) -> dict:
    for item in payload["operators"]:
        if item["name"] == name:
            return item
    raise AssertionError(f"operator not found: {name}")


def _frame(rows: int = 120) -> pd.DataFrame:
    idx = np.arange(rows, dtype=float)
    close = 100.0 + idx + np.sin(idx / 3.0)
    volume = 10.0 + (idx % 7.0)
    return pd.DataFrame(
        {
            "close": close,
            "high": close + 1.5,
            "low": close - 1.5,
            "volume": volume,
            "ret_1": pd.Series(close).pct_change().fillna(0.0),
            "atr_14": np.full(rows, 2.0),
        }
    )
