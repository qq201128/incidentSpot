from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest

from importlib import util
from pathlib import Path
import sys

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "high_winrate_factor_combo_goal.py"
SPEC = util.spec_from_file_location("high_winrate_factor_combo_goal", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
goal = util.module_from_spec(SPEC)
sys.modules[SPEC.name] = goal
SPEC.loader.exec_module(goal)
from app.services.high_winrate_combo_multi_duration import parse_durations, run_multi_duration_goal

ROWS = 130
THIRTY_MINUTES_MS = 30 * 60_000
SOURCE_OFFSET_MS = 29 * 60_000


def test_combo_hit_uses_thresholded_directional_returns() -> None:
    frame = _frame()
    score = _score()

    hit = goal._combo_hit(frame, ("factor_a", "factor_b"), (1, 1), score, threshold=1.0)

    assert hit is not None
    assert hit.members == ("factor_a", "factor_b")
    assert hit.trades == ROWS
    assert hit.win_rate == 1.0
    assert hit.profit_factor == np.inf


def test_paper_signal_waits_when_latest_score_is_inside_threshold() -> None:
    frame = _frame()
    score = _score()
    score.iloc[-1] = 0.25
    hit = goal.ComboHit(("factor_a", "factor_b"), (1, -1), 1.0, 0.75, 2.0, ROWS, 0.01, score)

    signal = goal._paper_signal(frame, 1, hit, "30m")

    assert signal["direction"] == "wait"
    assert signal["qualityPassed"] is False
    assert signal["simulationStrategyKey"] == "high_winrate_factor_combo_goal_top1"
    assert signal["members"][1]["orientation"] == -1


def test_search_frame_uses_next_duration_bar_return() -> None:
    frame = pd.DataFrame(
        {
            "open_time": [0, THIRTY_MINUTES_MS, THIRTY_MINUTES_MS * 2],
            "close": [100.0, 105.0, 90.0],
            "factor_a": [1.0, 2.0, 3.0],
        }
    )

    result = goal._search_frame(frame, "30m")

    assert result["fwd_ret"].iloc[0] == pytest.approx(0.05)
    assert result["fwd_ret"].iloc[1] == pytest.approx(90.0 / 105.0 - 1.0)
    assert pd.isna(result["fwd_ret"].iloc[2])


def test_run_goal_promotes_selected_combos(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    library = tmp_path / "library.json"
    frame = _frame()
    hit = goal.ComboHit(("factor_a", "factor_b"), (1, -1), 1.0, 0.75, 2.0, ROWS, 0.01, _score())
    promoted = []
    cached = []

    monkeypatch.setattr(goal, "load_backend_env_file", lambda: None)
    monkeypatch.setattr(goal, "load_factor_frame", lambda *_args: frame)
    monkeypatch.setattr(goal, "_search_frame", lambda *_args: frame)
    monkeypatch.setattr(goal, "_oriented_score_search", lambda _frame: _score_search())
    monkeypatch.setattr(goal, "_ranked_hit_search", lambda *_args: _ranked_search([hit]))
    monkeypatch.setattr(
        goal,
        "upsert_good_combinations",
        lambda report: promoted.append(report["ranking"][0]["factorName"]) or {"promoted": 1, "libraryTotal": 1},
    )
    monkeypatch.setattr(goal, "save_cached_high_winrate_combo_ranking", lambda report: cached.append(report["duration"]))
    monkeypatch.setattr(goal, "promote_high_winrate_strategy", lambda *_args: {"status": "active"})
    monkeypatch.setattr(goal, "rebuild_combination_signal_watchlist", lambda _symbol: {"symbol": "BTCUSDT"})
    monkeypatch.setattr(goal, "save_cached_combination_signals", lambda _payload: None)

    payload = goal.run_goal("btcusdt", "30m", 1, output, library)

    report = json.loads(output.read_text(encoding="utf-8"))
    stored = json.loads(library.read_text(encoding="utf-8"))
    assert promoted == ["goal_combo__factor_a__factor_b"]
    assert cached == ["30m"]
    assert payload["promotion"]["promoted"] == 1
    assert report["promotion"]["libraryTotal"] == 1
    assert stored["factors"][0]["members"][1]["orientation"] == -1


def test_library_payload_preserves_trade_threshold(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(goal, "load_backend_env_file", lambda: None)
    monkeypatch.setattr(goal, "load_factor_frame", lambda *_args: _frame())
    monkeypatch.setattr(goal, "_search_frame", lambda *_args: _frame())
    monkeypatch.setattr(goal, "_oriented_score_search", lambda _frame: _score_search())
    monkeypatch.setattr(goal, "save_cached_high_winrate_combo_ranking", lambda _report: None)
    monkeypatch.setattr(goal, "promote_high_winrate_strategy", lambda *_args: {"status": "active"})
    monkeypatch.setattr(goal, "rebuild_combination_signal_watchlist", lambda _symbol: {"symbol": "BTCUSDT"})
    monkeypatch.setattr(goal, "save_cached_combination_signals", lambda _payload: None)
    hit = goal.ComboHit(("factor_a", "factor_b"), (1, -1), 1.75, 0.75, 2.0, ROWS, 0.01, _score())

    captured = {}

    def upsert(report: dict) -> dict:
        captured.update(report["ranking"][0])
        return {"promoted": 1, "libraryTotal": 1}

    monkeypatch.setattr(goal, "_ranked_hit_search", lambda *_args: _ranked_search([hit]))
    monkeypatch.setattr(goal, "upsert_good_combinations", upsert)

    goal.run_goal("btcusdt", "30m", 1, tmp_path / "report.json", tmp_path / "library.json")

    assert captured["threshold"] == 1.75


def test_ranked_hits_can_select_four_member_combo(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = _frame()
    score = _score()
    scores = {
        f"factor_{index}": goal.OrientedScore(score, 1)
        for index in range(4)
    }
    monkeypatch.setattr(goal.goal_search, "search_candidate_names", lambda *_args: list(scores))

    hits = goal._ranked_hits(frame, scores)

    assert any(len(row.members) == 4 for row in hits)


def test_empty_ranking_exposes_combo_gate_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = _frame().assign(fwd_ret=0.01)
    score = pd.Series(np.where(np.arange(ROWS) % 2 == 0, 1.0, -1.0), dtype=float)
    scores = {
        "factor_a": goal.OrientedScore(score, 1),
        "factor_b": goal.OrientedScore(score, 1),
    }
    monkeypatch.setattr(goal.goal_search, "search_candidate_names", lambda *_args: list(scores))

    ranked = goal._ranked_hit_search(frame, scores)
    score_search = goal.ScoreSearch(scores, _candidate_diag(len(scores)))
    payload = goal._report_payload("BTCUSDT", "30m", 1, frame, score_search, ranked, [])

    assert payload["ranking"] == []
    assert payload["rankingFailure"]["stage"] == "combo_threshold_gates"
    assert payload["rankingFailure"]["reason"] == "no_combo_met_target_gates"
    assert payload["rankingDiagnostics"]["gateFailures"]["win_rate_below_min"] > 0
    assert payload["rankingDiagnostics"]["bestRejected"]["reason"] == "win_rate_below_min"


def test_validation_rejects_goal_combo_when_oos_win_rate_falls_below_target() -> None:
    score = _score()
    frame = _frame()
    fwd_ret = np.where(score > 0, 0.01, -0.01)
    fwd_ret[-30:] *= -1
    frame = frame.assign(fwd_ret=fwd_ret)
    hit = goal.ComboHit(("factor_a", "factor_b"), (1, -1), 1.0, 0.75, 2.0, ROWS, 0.01, score)

    validation = goal.validate_goal_combo_hits(frame, [hit], "30m")

    assert validation.passed == []
    rejection = validation.payload["rejections"][0]
    assert rejection["status"] == "rejected"
    assert any("winRate" in reason and "< 0.7" in reason for reason in rejection["reasons"])
    assert validation.payload["thresholds"]["recomputedMinWinRate"] == goal.TARGET_WIN_RATE


def test_empty_candidate_factors_expose_filter_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pd.DataFrame(
        {
            "open_time": np.arange(50) * THIRTY_MINUTES_MS,
            "close": np.linspace(100.0, 101.0, 50),
            "factor_a": np.linspace(0.0, 1.0, 50),
            "fwd_ret": np.linspace(-0.01, 0.01, 50),
        }
    )

    score_search = goal._oriented_score_search(frame)
    ranked = goal._ranked_hit_search(frame, score_search.scores)
    payload = goal._report_payload("BTCUSDT", "1d", 1, frame, score_search, ranked, [])

    assert payload["search"]["candidateFactors"] == 0
    assert payload["ranking"] == []
    assert payload["rankingFailure"]["stage"] == "candidate_factor_filter"
    assert payload["rankingFailure"]["reason"] == "no_candidate_factors_met_min_periods"
    assert payload["candidateDiagnostics"]["maxValidPairs"] == 50


def test_empty_ranking_does_not_create_silent_success_row() -> None:
    frame = _frame()
    score_search = goal.ScoreSearch({}, _candidate_diag(0, reason="no_candidate_factors_met_min_periods"))
    ranked = _ranked_search([])

    payload = goal._report_payload("BTCUSDT", "30m", 1, frame, score_search, ranked, [])

    assert payload["ranking"] == []
    assert payload["paperLiveSimulation"] == []
    assert payload["rankingFailure"] is not None


def test_ranking_row_names_all_combo_members() -> None:
    hit = goal.ComboHit(("factor_a", "factor_b", "factor_c"), (1, -1, 1), 0.75, 0.8, 2.0, ROWS, 0.01, _score())

    row = goal._ranking_row(1, hit)

    assert row["factorName"] == "goal_combo__factor_a__factor_b__factor_c"
    assert row["formula"] == "oriented_zscore_pair_threshold_v1(factor_a, factor_b, factor_c)"
    assert row["comboSize"] == 3


def test_run_multi_duration_goal_writes_aggregate_outputs(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    library = tmp_path / "library.json"
    calls = []

    def run_single(symbol: str, duration: str, target_count: int, out: Path, lib: Path) -> dict:
        calls.append((symbol, duration, target_count, out.name, lib.name))
        payload = _duration_payload(symbol, duration, target_count)
        out.write_text(json.dumps(payload), encoding="utf-8")
        lib.write_text(json.dumps({"factors": payload["ranking"]}), encoding="utf-8")
        return payload

    report = run_multi_duration_goal("btcusdt", parse_durations("10m,30m"), 1, output, library, run_single)

    stored = json.loads(output.read_text(encoding="utf-8"))
    assert [row[1] for row in calls] == ["10m", "30m"]
    assert calls[0][3:] == ("report_10m.json", "library_10m.json")
    assert report["promotion"]["promoted"] == 2
    assert stored["bestRanking"][0]["duration"] == "30m"
    assert json.loads(library.read_text(encoding="utf-8"))["perDuration"][1]["duration"] == "30m"


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


def _duration_payload(symbol: str, duration: str, target_count: int) -> dict:
    win_rate = 0.80 if duration == "10m" else 0.90
    return {
        "version": "single",
        "updatedAt": "now",
        "symbol": symbol.upper(),
        "duration": duration,
        "target": {"targetCount": target_count},
        "ranking": [
            {
                "rank": 1,
                "factorName": f"combo_{duration}",
                "winRate": win_rate,
                "profitFactor": 2.0,
                "avgReturn": 0.01,
                "trades": 100,
            }
        ],
        "promotion": {"symbol": symbol.upper(), "duration": duration, "promoted": 1, "libraryTotal": 1},
    }


def _score_search() -> goal.ScoreSearch:
    return goal.ScoreSearch({}, _candidate_diag(0))


def _ranked_search(hits: list[goal.ComboHit]) -> goal.RankedSearch:
    return goal.RankedSearch(
        hits,
        {
            "stage": "combo_threshold_gates",
            "searchCandidateLimit": goal.SEARCH_CANDIDATE_LIMIT,
            "selectedCandidateFactors": 0,
            "testedCombinations": 0,
            "testedThresholdEvaluations": 0,
            "gateFailures": {
                "min_trades_below_min": 0,
                "win_rate_below_min": 0,
                "profit_factor_below_min": 0,
            },
            "passedThresholdEvaluations": len(hits),
            "bestRejected": None,
            "failureReason": None if hits else "no_candidate_factors",
            "hitCount": len(hits),
        },
    )


def _candidate_diag(count: int, *, reason: str | None = None) -> dict:
    return {
        "stage": "candidate_factor_filter",
        "reason": reason,
        "minPeriods": goal.BACKTEST_MIN_PERIODS,
        "numericFactorColumns": count,
        "eligibleCandidateFactors": count,
        "rejectedCandidateFactors": 0,
        "maxValidPairs": ROWS,
        "topRejectedByValidPairs": [],
    }
