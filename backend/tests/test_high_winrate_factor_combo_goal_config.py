from __future__ import annotations

import json
import sys
import time
from importlib import util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.services.high_winrate_combo_multi_duration import parse_durations, run_multi_duration_goal

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "high_winrate_factor_combo_goal.py"
SPEC = util.spec_from_file_location("high_winrate_factor_combo_goal_config_tests", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
goal = util.module_from_spec(SPEC)
sys.modules[SPEC.name] = goal
SPEC.loader.exec_module(goal)

ROWS = 130
THIRTY_MINUTES_MS = 30 * 60_000


def test_run_goal_uses_expanded_search_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = goal.GoalSearchConfig(
        candidate_limit=40,
        signal_thresholds=goal.goal_search.signal_thresholds(0.1, 0.3, 0.1),
        min_trades=77,
    )
    monkeypatch.setattr(goal, "load_backend_env_file", lambda: None)
    monkeypatch.setattr(goal, "load_factor_frame", lambda *_args: _frame())
    monkeypatch.setattr(goal, "_search_frame", lambda *_args: _frame())
    monkeypatch.setattr(goal, "_oriented_score_search", lambda _frame: _score_search())
    monkeypatch.setattr(goal, "_ranked_hit_search", lambda *_args: _ranked_search())

    payload = goal.run_goal("btcusdt", "10m", 1, tmp_path / "report.json", tmp_path / "library.json", config)

    assert payload["target"]["minTrades"] == 77
    assert payload["target"]["searchCandidateLimit"] == 40
    assert payload["target"]["thresholds"] == [0.1, 0.2, 0.3]


def test_run_multi_duration_goal_parallel_preserves_report_order(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    library = tmp_path / "library.json"

    report = run_multi_duration_goal(
        "btcusdt",
        parse_durations("10m,60m"),
        1,
        output,
        library,
        _run_single,
        parallel_workers=2,
    )

    stored = json.loads(library.read_text(encoding="utf-8"))
    assert [row["duration"] for row in report["perDuration"]] == ["10m", "60m"]
    assert [row["duration"] for row in stored["perDuration"]] == ["10m", "60m"]


def _run_single(symbol: str, duration: str, target_count: int, out: Path, lib: Path) -> dict:
    if duration == "10m":
        time.sleep(0.05)
    payload = _duration_payload(symbol, duration, target_count)
    out.write_text(json.dumps(payload), encoding="utf-8")
    lib.write_text(json.dumps({"factors": payload["ranking"]}), encoding="utf-8")
    return payload


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open_time": np.arange(ROWS) * THIRTY_MINUTES_MS,
            "close": np.linspace(100.0, 110.0, ROWS),
            "fwd_ret": np.repeat(0.01, ROWS),
        }
    )


def _score_search() -> goal.ScoreSearch:
    return goal.ScoreSearch({}, _candidate_diag())


def _ranked_search() -> goal.RankedSearch:
    return goal.RankedSearch(
        [],
        {
            "selectedCandidateFactors": 0,
            "testedCombinations": 0,
            "testedThresholdEvaluations": 0,
            "failureReason": "no_candidate_factors",
        },
    )


def _candidate_diag() -> dict:
    return {
        "stage": "candidate_factor_filter",
        "reason": None,
        "minPeriods": goal.BACKTEST_MIN_PERIODS,
        "numericFactorColumns": 0,
        "eligibleCandidateFactors": 0,
        "rejectedCandidateFactors": 0,
        "maxValidPairs": ROWS,
        "topRejectedByValidPairs": [],
    }


def _duration_payload(symbol: str, duration: str, target_count: int) -> dict:
    return {
        "version": "single",
        "updatedAt": "now",
        "symbol": symbol.upper(),
        "duration": duration,
        "target": {"targetCount": target_count},
        "ranking": [_ranking_row(duration)],
        "promotion": {"symbol": symbol.upper(), "duration": duration, "promoted": 1, "libraryTotal": 1},
    }


def _ranking_row(duration: str) -> dict:
    return {
        "rank": 1,
        "factorName": f"combo_{duration}",
        "winRate": 0.80,
        "profitFactor": 2.0,
        "avgReturn": 0.01,
        "trades": 100,
    }
