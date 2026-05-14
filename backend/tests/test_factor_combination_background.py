from __future__ import annotations

from datetime import datetime

import pytest

from app.services import factor_combination_background as combo_background
from app.services import factor_combination_service as combo_service
from app.services.factor_candidate_selection import select_base_candidates
from app.services.factor_combination_service import CombinationSearchConfig
from app.services.factor_registry import FactorCategory, FactorDefinition

ROWS = 1300
SECONDS_PER_MINUTE = 60
SECONDS_PER_DAY = 86_400


def test_base_candidate_selection_preserves_source_quotas() -> None:
    candidates = [
        _candidate("native_a", "kline_features.py", 1.0),
        _candidate("native_b", "kline_features.py", 0.9),
        _candidate("mined_a", "mined_factor_library.json", 10.0),
        _candidate("mined_b", "mined_factor_library.json", 9.0),
        _candidate("agent_a", "agent_mined_factor_library.json", 8.0),
    ]

    selected = select_base_candidates(
        candidates,
        CombinationSearchConfig(
            base_factor_limit=3,
            native_factor_limit=1,
            mined_factor_limit=1,
            agent_factor_limit=1,
            combo_sizes=(2,),
        ),
        rank_key=combo_service._base_rank_key,
    )

    assert [item.factor.name for item in selected] == ["native_a", "mined_a", "agent_a"]


def test_daily_refresh_seconds_targets_next_0030() -> None:
    tz = combo_background.DAILY_REFRESH_TZ
    before_review = datetime(2026, 5, 13, 0, 29, tzinfo=tz)
    exactly_review = datetime(2026, 5, 13, 0, 30, tzinfo=tz)
    assert combo_background.seconds_until_next_daily_refresh(before_review) == SECONDS_PER_MINUTE
    assert combo_background.seconds_until_next_daily_refresh(exactly_review) == SECONDS_PER_DAY


def test_daily_refresh_updates_all_combo_durations(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_run(symbol: str, duration: str, config: object) -> dict:
        calls.append(("run", symbol, duration, config))
        return {"symbol": symbol, "duration": duration, "ranking": []}

    def fake_save(report: dict) -> None:
        calls.append(("save", report["symbol"], report["duration"], None))

    def fake_upsert(report: dict) -> dict:
        calls.append(("promote", report["symbol"], report["duration"], None))
        return {"symbol": report["symbol"], "duration": report["duration"], "promoted": 0, "libraryTotal": 0}

    def fake_sync(symbol: str, duration: str, *, ranking_report: dict) -> dict:
        calls.append(("sync", symbol, duration, ranking_report["duration"]))
        return {"status": "up_to_date"}

    def fake_learning(symbol: str, duration: str, ranking_report: dict, *, run_llm_agent: bool) -> None:
        assert run_llm_agent is True
        calls.append(("learn", symbol, duration, ranking_report["duration"]))

    monkeypatch.setattr(combo_background, "run_factor_combination_ranking", fake_run)
    monkeypatch.setattr(combo_background, "save_cached_combination_ranking", fake_save)
    monkeypatch.setattr(combo_background, "upsert_good_combinations", fake_upsert)
    monkeypatch.setattr(combo_background, "sync_lstm_model_to_combo_ranking", fake_sync)
    monkeypatch.setattr(combo_background, "refresh_factor_learning_memory", fake_learning)
    combo_background.refresh_symbol_combination_rankings("btcusdt")

    run_durations = [item[2] for item in calls if item[0] == "run"]
    assert [item[2] for item in calls if item[0] == "save"] == run_durations
    assert [item[2] for item in calls if item[0] == "promote"] == run_durations
    assert [item[2] for item in calls if item[0] == "sync"] == run_durations
    assert [item[2] for item in calls if item[0] == "learn"] == run_durations
    assert run_durations == ["10m", "30m", "60m", "1d"]


def test_combo_refresh_surfaces_lstm_sync_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(
        combo_background,
        "run_factor_combination_ranking",
        lambda symbol, duration, _config: {"symbol": symbol, "duration": duration, "ranking": []},
    )
    monkeypatch.setattr(combo_background, "save_cached_combination_ranking", lambda _report: None)
    monkeypatch.setattr(combo_background, "upsert_good_combinations", _promotion_report)
    monkeypatch.setattr(combo_background, "sync_lstm_model_to_combo_ranking", _fail_sync)
    monkeypatch.setattr(combo_background, "refresh_factor_learning_memory", lambda *_args, **_kwargs: calls.append("learn"))

    with pytest.raises(RuntimeError, match="lstm sync failed"):
        combo_background.refresh_combination_ranking_for_symbol_duration("BTCUSDT", "10m")

    assert calls == []


def _candidate(name: str, source_file: str, score: float):
    factor = FactorDefinition(
        name=name,
        category=FactorCategory.RETURN,
        description=name,
        formula=name,
        source_file=source_file,
    )
    return combo_service._BaseCandidate(
        factor=factor,
        metrics={"factorScore": score, "winRate": 0.6, "sharpe": 1.0, "totalPeriods": ROWS},
        orientation=1,
    )


def _promotion_report(report: dict) -> dict:
    return {"symbol": report["symbol"], "duration": report["duration"], "promoted": 0, "libraryTotal": 0}


def _fail_sync(*_args, **_kwargs) -> dict:
    raise RuntimeError("lstm sync failed")
