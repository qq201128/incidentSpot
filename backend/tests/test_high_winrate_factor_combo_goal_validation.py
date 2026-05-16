from __future__ import annotations

import json
import sys
from importlib import util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "high_winrate_factor_combo_goal.py"
SPEC = util.spec_from_file_location("high_winrate_factor_combo_goal_validation_tests", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
goal = util.module_from_spec(SPEC)
sys.modules[SPEC.name] = goal
SPEC.loader.exec_module(goal)

ROWS = 130
THIRTY_MINUTES_MS = 30 * 60_000


def test_run_goal_publishes_only_validation_passed_combos(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frame = _frame()
    passed = _hit("factor_a", "factor_b", _score())
    rejected = _hit("factor_c", "factor_d", -_score())
    promoted = []
    cached = []
    _patch_goal_runtime(monkeypatch, frame, [passed, rejected], promoted, cached)

    payload = goal.run_goal("btcusdt", "30m", 2, tmp_path / "report.json", tmp_path / "library.json")

    ranking_names = [row["factorName"] for row in payload["ranking"]]
    paper_names = [row["factorName"] for row in payload["paperLiveSimulation"]]
    assert ranking_names == ["goal_combo__factor_a__factor_b"]
    assert paper_names == ranking_names
    assert payload["validationGate"]["status"] == "partial"
    assert payload["validationGate"]["rejectedCount"] == 1
    assert payload["validationGate"]["rejections"][0]["factorName"] == "goal_combo__factor_c__factor_d"
    assert promoted == ["goal_combo__factor_a__factor_b"]
    assert cached == ["30m"]


def test_run_goal_returns_empty_ranking_when_all_validation_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frame = _frame()
    rejected = _hit("factor_c", "factor_d", -_score())
    promoted = []
    cached = []
    _patch_goal_runtime(monkeypatch, frame, [rejected], promoted, cached)

    payload = goal.run_goal("btcusdt", "60m", 1, tmp_path / "report.json", tmp_path / "library.json")
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))

    assert payload["ranking"] == []
    assert payload["paperLiveSimulation"] == []
    assert payload["validationGate"]["status"] == "failed"
    assert payload["validationGate"]["failureReason"] == "all_combos_rejected_by_validation"
    assert payload["rankingFailure"]["stage"] == "validation_gate"
    assert report["rankingFailure"]["details"]["rejectedCount"] == 1
    assert promoted == []
    assert cached == []


def _patch_goal_runtime(
    monkeypatch: pytest.MonkeyPatch,
    frame: pd.DataFrame,
    hits: list,
    promoted: list[str],
    cached: list[str],
) -> None:
    monkeypatch.setattr(goal, "load_backend_env_file", lambda: None)
    monkeypatch.setattr(goal, "load_factor_frame", lambda *_args: frame)
    monkeypatch.setattr(goal, "_search_frame", lambda *_args: frame)
    monkeypatch.setattr(goal, "_oriented_score_search", lambda _frame: _score_search())
    monkeypatch.setattr(goal, "_ranked_hit_search", lambda *_args: _ranked_search(hits))
    monkeypatch.setattr(goal, "upsert_good_combinations", lambda report: _promote(report, promoted))
    monkeypatch.setattr(goal, "save_cached_high_winrate_combo_ranking", lambda report: cached.append(report["duration"]))
    monkeypatch.setattr(goal, "promote_high_winrate_strategy", lambda *_args: {"status": "active"})
    monkeypatch.setattr(goal, "rebuild_combination_signal_watchlist", lambda _symbol: {"symbol": "BTCUSDT"})
    monkeypatch.setattr(goal, "save_cached_combination_signals", lambda _payload: None)


def _promote(report: dict, promoted: list[str]) -> dict:
    promoted.extend(row["factorName"] for row in report["ranking"])
    return {"promoted": len(report["ranking"]), "libraryTotal": len(report["ranking"])}


def _frame() -> pd.DataFrame:
    score = _score()
    return pd.DataFrame(
        {
            "open_time": np.arange(ROWS) * THIRTY_MINUTES_MS,
            "close": np.linspace(100.0, 110.0, ROWS),
            "fwd_ret": np.where(score > 0, 0.01, -0.01),
        }
    )


def _score() -> pd.Series:
    return pd.Series(np.where(np.arange(ROWS) % 2 == 0, 2.0, -2.0), dtype=float)


def _hit(first: str, second: str, score: pd.Series) -> goal.ComboHit:
    return goal.ComboHit((first, second), (1, 1), 1.0, 0.75, 2.0, ROWS, 0.01, score)


def _score_search() -> goal.ScoreSearch:
    return goal.ScoreSearch({}, _candidate_diag())


def _ranked_search(hits: list[goal.ComboHit]) -> goal.RankedSearch:
    return goal.RankedSearch(
        hits,
        {
            "stage": "combo_threshold_gates",
            "searchCandidateLimit": goal.SEARCH_CANDIDATE_LIMIT,
            "selectedCandidateFactors": 0,
            "testedCombinations": 0,
            "testedThresholdEvaluations": 0,
            "gateFailures": {},
            "passedThresholdEvaluations": len(hits),
            "bestRejected": None,
            "failureReason": None,
            "hitCount": len(hits),
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
