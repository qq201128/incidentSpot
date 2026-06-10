from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services.factor_backtest_service import IC_ROLLING_WINDOW, _compute_rolling_ic
from app.services.factor_backtest_service import run_factor_backtest_on_frame
from app.services.factor_registry import FactorCategory, FactorDefinition, FactorDirection


def test_rolling_ic_ranks_inside_each_window() -> None:
    factor = pd.Series([
        -0.234, -0.366, 1.212, 0.494, 0.671, -0.508, 1.924, 1.71, 0.566, 0.684,
        -2.027, 0.638, -0.194, 0.434, 0.682, -0.341, -1.69, 0.368, -0.741, -0.33,
        -0.605, -0.342, -2.31, 1.217, 0.253,
    ])
    fwd_ret = pd.Series([
        1.111, 1.98, 0.023, -1.802, -0.894, -1.207, -0.502, 0.08, -2.002, 0.342,
        -1.51, 0.297, -0.109, -0.314, -0.073, -0.54, -0.612, -1.683, -0.03, 1.846,
        1.98, 1.322, 0.706, -0.676, 1.443,
    ])

    rolling_ic = _compute_rolling_ic(factor, fwd_ret)
    expected = _window_spearman(factor.iloc[:IC_ROLLING_WINDOW], fwd_ret.iloc[:IC_ROLLING_WINDOW])
    global_ranked = pd.DataFrame({
        "factor": factor.rank(method="average"),
        "fwd_ret": fwd_ret.rank(method="average"),
    })
    leaked_global_rank = global_ranked["factor"].iloc[:IC_ROLLING_WINDOW].corr(
        global_ranked["fwd_ret"].iloc[:IC_ROLLING_WINDOW]
    )

    assert rolling_ic.iloc[0] == pytest.approx(expected)
    assert rolling_ic.iloc[0] != pytest.approx(leaked_global_rank)


def test_rolling_ic_skips_constant_windows() -> None:
    factor = pd.Series(np.ones(IC_ROLLING_WINDOW + 1))
    fwd_ret = pd.Series(np.arange(IC_ROLLING_WINDOW + 1, dtype=float))

    assert _compute_rolling_ic(factor, fwd_ret).empty


def test_factor_backtest_exposes_out_of_sample_selection_gate() -> None:
    rows = 700
    close = 100.0 * np.cumprod(1.0 + 0.002 * np.sin(np.arange(rows) / 5.0))
    future = pd.Series(close).pct_change().shift(-1).fillna(0.0)
    frame = pd.DataFrame({
        "open_time": np.arange(rows) * 60_000,
        "open": close,
        "high": close * 1.001,
        "low": close * 0.999,
        "close": close,
        "volume": np.full(rows, 100.0),
        "alpha": future,
    })
    factor = FactorDefinition(
        name="alpha",
        category=FactorCategory.RETURN,
        description="alpha",
        formula="alpha",
        direction=FactorDirection.HIGHER_BETTER,
    )

    result = run_factor_backtest_on_frame(factor, frame, symbol="btcusdt", duration="10m")
    oos = result["outOfSample"]

    assert oos["selectionMetricSource"] == "validation_and_test_only"
    assert set(("train", "validation", "test")) <= set(oos)
    assert oos["validation"]["returnMetrics"]["sampleCount"] > 0
    assert oos["validation"]["researchMetrics"]["quintileReturns"]
    assert oos["selectionGate"]["status"] in {"passed", "failed"}


def test_factor_backtest_reports_regime_buckets() -> None:
    rows = 180
    close = 100.0 + np.arange(rows) * 0.2
    frame = pd.DataFrame({
        "open_time": np.arange(rows) * 60_000,
        "open": close,
        "high": close * 1.001,
        "low": close * 0.999,
        "close": close,
        "volume": np.full(rows, 100.0),
        "alpha": np.arange(rows, dtype=float),
    })
    factor = FactorDefinition(
        name="alpha",
        category=FactorCategory.RETURN,
        description="alpha",
        formula="alpha",
        direction=FactorDirection.HIGHER_BETTER,
    )

    result = run_factor_backtest_on_frame(factor, frame, symbol="btcusdt", duration="10m")

    assert result["regime"]["policy"] == "factor_regime_bucket_v1"
    assert result["regime"]["byTrend"]
    assert result["regime"]["byVolatility"]
    assert result["regime"]["byRegime"]


def _window_spearman(factor: pd.Series, fwd_ret: pd.Series) -> float:
    return float(factor.rank(method="average").corr(fwd_ret.rank(method="average")))
