from __future__ import annotations

import asyncio
from datetime import datetime

import pytest

from app.services import factor_combination_background as combo_background
from app.services import factor_combination_service as combo_service
from app.services.background_loop_status import background_loop_statuses, reset_background_loop_statuses
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

    def fake_signal_cache(symbol: str) -> None:
        calls.append(("signals", symbol, None, None))

    monkeypatch.setattr(combo_background, "run_factor_combination_ranking", fake_run)
    monkeypatch.setattr(combo_background, "save_cached_combination_ranking", fake_save)
    monkeypatch.setattr(combo_background, "refresh_factor_combination_data_dependencies", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(combo_background, "upsert_good_combinations", fake_upsert)
    monkeypatch.setattr(combo_background, "sync_lstm_model_to_combo_ranking", fake_sync)
    monkeypatch.setattr(combo_background, "refresh_factor_learning_memory", fake_learning)
    monkeypatch.setattr(combo_background, "_refresh_signal_watchlist_cache", fake_signal_cache)
    combo_background.refresh_symbol_combination_rankings("btcusdt")

    run_durations = [item[2] for item in calls if item[0] == "run"]
    assert [item[2] for item in calls if item[0] == "save"] == run_durations
    assert [item[2] for item in calls if item[0] == "promote"] == run_durations
    assert [item[2] for item in calls if item[0] == "sync"] == run_durations
    assert [item[2] for item in calls if item[0] == "learn"] == run_durations
    assert [item for item in calls if item[0] == "signals"] == [("signals", "BTCUSDT", None, None)]
    assert run_durations == ["10m", "30m", "60m", "1d"]


def test_daily_refresh_loop_runs_paper_live_closed_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    stop_event = asyncio.Event()

    async def sleep_once(_stop_event: asyncio.Event, _seconds: float) -> None:
        calls.append(("sleep", None))

    async def run_once(func):
        calls.append(("run", func.__name__))
        stop_event.set()
        return {"status": "passed"}

    monkeypatch.setattr(combo_background, "_sleep_for", sleep_once)
    monkeypatch.setattr(combo_background, "seconds_until_next_daily_refresh", lambda: 0.0)
    monkeypatch.setattr(combo_background, "run_blocking_daemon", run_once)

    asyncio.run(combo_background.factor_combination_daily_refresh_loop(stop_event))

    assert calls == [("sleep", None), ("run", "run_paper_live_daily_closed_loop")]


def test_daily_refresh_loop_records_failed_paper_live_report(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_background_loop_statuses()
    stop_event = asyncio.Event()

    async def sleep_once(_stop_event: asyncio.Event, _seconds: float) -> None:
        return None

    async def run_once(_func):
        stop_event.set()
        return _failed_daily_report()

    monkeypatch.setattr(combo_background, "_sleep_for", sleep_once)
    monkeypatch.setattr(combo_background, "seconds_until_next_daily_refresh", lambda: 0.0)
    monkeypatch.setattr(combo_background, "run_blocking_daemon", run_once)

    asyncio.run(combo_background.factor_combination_daily_refresh_loop(stop_event))

    status = background_loop_statuses()["factor_combo_daily"]
    assert status["status"] == "failed"
    assert status["lastError"] == "paper-live daily loop failed"
    details = status["lastFailureDetails"]
    assert details["stage"] == "paper_live_daily_loop"
    assert details["status"] == "failed"
    assert details["failedResults"] == [_expected_failed_daily_result()]


def test_combo_refresh_surfaces_lstm_sync_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(
        combo_background,
        "run_factor_combination_ranking",
        lambda symbol, duration, _config: {"symbol": symbol, "duration": duration, "ranking": []},
    )
    monkeypatch.setattr(combo_background, "refresh_factor_combination_data_dependencies", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(combo_background, "save_cached_combination_ranking", lambda _report: None)
    monkeypatch.setattr(combo_background, "upsert_good_combinations", _promotion_report)
    monkeypatch.setattr(combo_background, "sync_lstm_model_to_combo_ranking", _fail_sync)
    monkeypatch.setattr(combo_background, "refresh_factor_learning_memory", lambda *_args, **_kwargs: calls.append("learn"))

    with pytest.raises(RuntimeError, match="lstm sync failed"):
        combo_background.refresh_combination_ranking_for_symbol_duration("BTCUSDT", "10m")

    assert calls == []


def test_combo_refresh_writes_duration_klines_before_ranking(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_dependencies(symbol: str, duration: str, **kwargs) -> None:
        assert kwargs["refresh_duration_klines"] == combo_background._refresh_duration_klines
        calls.append(("dependencies", symbol, duration, None))

    monkeypatch.setattr(combo_background, "refresh_factor_combination_data_dependencies", fake_dependencies)
    monkeypatch.setattr(
        combo_background,
        "run_factor_combination_ranking",
        lambda symbol, duration, _config: calls.append(("rank", symbol, duration, None)) or {"symbol": symbol, "duration": duration, "ranking": []},
    )
    monkeypatch.setattr(combo_background, "save_cached_combination_ranking", lambda _report: None)
    monkeypatch.setattr(combo_background, "upsert_good_combinations", _promotion_report)
    monkeypatch.setattr(combo_background, "sync_lstm_model_to_combo_ranking", lambda *_args, **_kwargs: {"status": "up_to_date"})
    monkeypatch.setattr(combo_background, "refresh_factor_learning_memory", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(combo_background, "_refresh_signal_watchlist_cache", lambda *_args: None)

    combo_background.refresh_combination_ranking_for_symbol_duration("btcusdt", "10m")

    assert calls[:2] == [("dependencies", "BTCUSDT", "10m", None), ("rank", "BTCUSDT", "10m", None)]


def test_duration_kline_refresh_backfills_until_min_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    counts = [0, 300, 540]
    fetches = []
    open_times = [600_000, 0]

    def fetch_klines(symbol: str, duration: str, *, limit: int, end_time: int | None = None) -> list[dict]:
        fetches.append((symbol, duration, end_time))
        return [_kline_row(open_times.pop(0))]

    monkeypatch.setattr(combo_background, "fetch_klines", fetch_klines)
    monkeypatch.setattr(combo_background, "upsert_klines_rows", lambda *_args: None)
    monkeypatch.setattr(combo_background, "oldest_open_time", lambda *_args: 1_200_000)
    monkeypatch.setattr(combo_background, "count_klines", lambda *_args: counts.pop(0))

    combo_background._backfill_duration_klines("BTCUSDT", "10m")

    assert fetches == [("BTCUSDT", "10m", 1_199_999), ("BTCUSDT", "10m", 599_999)]


def _kline_row(open_time: int) -> dict:
    return {
        "openTime": open_time,
        "open": 1.0,
        "high": 1.0,
        "low": 1.0,
        "close": 1.0,
        "volume": 1.0,
        "closeTime": open_time + 1,
    }


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


def _failed_daily_report() -> dict:
    return {
        "status": "failed",
        "results": [
            {
                "symbol": "BTCUSDT",
                "duration": "10m",
                "status": "failed",
                "stages": [{"status": "failed", **_expected_failed_stage()}],
            }
        ],
    }


def _expected_failed_daily_result() -> dict:
    return {
        "symbol": "BTCUSDT",
        "duration": "10m",
        "status": "failed",
        "failedStages": [_expected_failed_stage()],
    }


def _expected_failed_stage() -> dict:
    return {
        "stage": "settle_due_predictions",
        "reason": "settlement failed",
        "exceptionType": "RuntimeError",
    }
