from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services.factor_backtest_service import IC_ROLLING_WINDOW, _compute_rolling_ic


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


def _window_spearman(factor: pd.Series, fwd_ret: pd.Series) -> float:
    return float(factor.rank(method="average").corr(fwd_ret.rank(method="average")))
