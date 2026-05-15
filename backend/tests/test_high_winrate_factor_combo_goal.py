from __future__ import annotations

import numpy as np
import pandas as pd

from importlib import util
from pathlib import Path
import sys

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "high_winrate_factor_combo_goal.py"
SPEC = util.spec_from_file_location("high_winrate_factor_combo_goal", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
goal = util.module_from_spec(SPEC)
sys.modules[SPEC.name] = goal
SPEC.loader.exec_module(goal)

ROWS = 130
THIRTY_MINUTES_MS = 30 * 60_000
SOURCE_OFFSET_MS = 29 * 60_000


def test_combo_hit_uses_thresholded_directional_returns() -> None:
    frame = _frame()
    score = _score()

    hit = goal._combo_hit(frame, ("factor_a", "factor_b"), score, threshold=1.0)

    assert hit is not None
    assert hit.members == ("factor_a", "factor_b")
    assert hit.trades == ROWS
    assert hit.win_rate == 1.0
    assert hit.profit_factor == np.inf


def test_paper_signal_waits_when_latest_score_is_inside_threshold() -> None:
    frame = _frame()
    score = _score()
    score.iloc[-1] = 0.25
    hit = goal.ComboHit(("factor_a", "factor_b"), 1.0, 0.75, 2.0, ROWS, 0.01, score)

    signal = goal._paper_signal(frame, 1, hit, "30m")

    assert signal["direction"] == "wait"
    assert signal["qualityPassed"] is False
    assert signal["simulationStrategyKey"] == "high_winrate_factor_combo_goal_top1"


def _frame() -> pd.DataFrame:
    score = _score()
    return pd.DataFrame(
        {
            "open_time": SOURCE_OFFSET_MS + np.arange(ROWS) * THIRTY_MINUTES_MS,
            "close": np.linspace(100.0, 110.0, ROWS),
            "fwd_ret": np.where(score > 0, 0.01, -0.01),
        }
    )


def _score() -> pd.Series:
    return pd.Series(np.where(np.arange(ROWS) % 2 == 0, 2.0, -2.0), dtype=float)
