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


def test_run_goal_promotes_selected_combos(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    library = tmp_path / "library.json"
    frame = _frame()
    hit = goal.ComboHit(("factor_a", "factor_b"), (1, -1), 1.0, 0.75, 2.0, ROWS, 0.01, _score())
    promoted = []

    monkeypatch.setattr(goal, "load_backend_env_file", lambda: None)
    monkeypatch.setattr(goal, "load_factor_frame", lambda *_args: frame)
    monkeypatch.setattr(goal, "_search_frame", lambda *_args: frame)
    monkeypatch.setattr(goal, "_oriented_scores", lambda _frame: {})
    monkeypatch.setattr(goal, "_ranked_hits", lambda *_args: [hit])
    monkeypatch.setattr(
        goal,
        "upsert_good_combinations",
        lambda report: promoted.append(report["ranking"][0]["factorName"]) or {"promoted": 1, "libraryTotal": 1},
    )

    payload = goal.run_goal("btcusdt", "30m", 1, output, library)

    report = json.loads(output.read_text(encoding="utf-8"))
    stored = json.loads(library.read_text(encoding="utf-8"))
    assert promoted == ["goal_combo__factor_a__factor_b"]
    assert payload["promotion"]["promoted"] == 1
    assert report["promotion"]["libraryTotal"] == 1
    assert stored["factors"][0]["members"][1]["orientation"] == -1


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
