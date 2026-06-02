from __future__ import annotations

from fastapi import BackgroundTasks

from app.api import factor_combinations as factor_combo_api
from app.api import factors as factors_api
from app.api import lstm as lstm_api
from app.services.background_loop_status import background_loop_statuses, reset_background_loop_statuses
from app.services import experiment_profiles as profiles
from app.services import lstm_daily_review as review
from app.services.lstm_daily_review import LstmDailyReviewConfig, LstmDailyReviewDependencies, run_lstm_daily_review


def test_fast_profile_uses_smaller_combo_and_lstm_configs() -> None:
    combo = profiles.combination_search_config_for_profile(profiles.EXPERIMENT_PROFILE_FAST)
    lstm = profiles.lstm_training_config_for_profile("btcusdt", "10m", profiles.EXPERIMENT_PROFILE_FAST)

    assert combo.base_factor_limit == 16
    assert combo.native_factor_limit == 10
    assert combo.mined_factor_limit == 4
    assert combo.agent_factor_limit == 1
    assert combo.combo_sizes == (2,)
    assert combo.result_limit == 50
    assert combo.prefilter_limit == 120
    assert combo.parallel_workers == 2
    assert lstm.feature_window == 32
    assert lstm.epochs == 2
    assert lstm.batch_size == 64
    assert lstm.min_samples == 80


def test_full_profile_gate_blocks_when_shadow_is_weak(monkeypatch) -> None:
    monkeypatch.setattr(
        profiles,
        "lstm_shadow_learning_summary",
        lambda *_args, **_kwargs: {"sampleCount": 20, "winRate": 0.40, "recentWinRate": 0.45, "avgReturn": -0.1},
    )
    monkeypatch.setattr(
        profiles,
        "high_winrate_demotion_status",
        lambda *_args, **_kwargs: {"status": "tradable"},
    )

    gate = profiles.shadow_gate_for_full_profile("BTCUSDT", "10m")

    assert gate.ready is False
    assert gate.reason == "lstm_shadow_not_ready"


def test_full_daily_review_blocks_before_expensive_work(monkeypatch) -> None:
    monkeypatch.setattr(
        review,
        "shadow_gate_for_full_profile",
        lambda *_args, **_kwargs: profiles.ShadowGateResult(False, "lstm_shadow_not_ready", {"reason": "blocked"}),
    )

    deps = LstmDailyReviewDependencies(
        fetch_klines=_forbidden("fetch_klines"),
        upsert_klines=_forbidden("upsert_klines"),
        count_klines=_forbidden("count_klines"),
        oldest_open_time=_forbidden("oldest_open_time"),
        ingest_market_context=_forbidden("ingest_market_context"),
        run_combination_ranking=_forbidden("run_combination_ranking"),
        save_combination_ranking=_forbidden("save_combination_ranking"),
        promote_combinations=_forbidden("promote_combinations"),
        train_lstm=_forbidden("train_lstm"),
        refresh_learning_memory=_forbidden("refresh_learning_memory"),
    )

    report = run_lstm_daily_review(
        LstmDailyReviewConfig(
            symbols=("BTCUSDT",),
            durations=("10m",),
            experiment_profile=profiles.EXPERIMENT_PROFILE_FULL,
        ),
        deps=deps,
    )

    duration_report = report["symbols"][0]["durations"][0]
    assert report["status"] == "blocked"
    assert report["experimentProfile"] == profiles.EXPERIMENT_PROFILE_FULL
    assert duration_report["status"] == "blocked"
    assert duration_report["reason"] == "lstm_shadow_not_ready"


def test_lstm_train_route_rejects_direct_training_overrides() -> None:
    try:
        lstm_api.lstm_train(symbol="btcusdt", feature_window=40)
    except Exception as exc:
        assert "direct in-process LSTM training is disabled" in str(exc)
    else:
        raise AssertionError("direct LSTM training overrides should be rejected")


def test_lstm_candidate_search_route_enqueues_model_job(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        lstm_api,
        "model_candidate_search",
        lambda *args, **kwargs: calls.append((args, kwargs)) or {"status": "queued"},
    )

    report = lstm_api.lstm_candidate_search(
        symbol="btcusdt",
        duration="10m",
        profile="full",
        parallel_workers=3,
    )

    assert report["status"] == "queued"
    assert calls[0][0] == ("lstm",)
    assert calls[0][1]["symbol"] == "btcusdt"
    assert calls[0][1]["duration"] == "10m"
    assert calls[0][1]["parallel_workers"] == 3


def test_factor_combination_refresh_route_uses_profile_defaults_and_aliases() -> None:
    background_tasks = BackgroundTasks()

    report = factor_combo_api.factor_combination_refresh(
        background_tasks=background_tasks,
        symbol="btcusdt",
        duration=None,
        profile="fast",
        base_factor_limit=9,
        combo_sizes="2,3",
        result_limit=20,
    )

    task = background_tasks.tasks[0]
    config = task.args[2]
    assert report["profile"] == "fast"
    assert report["searchConfig"]["baseFactorLimit"] == 9
    assert report["searchConfig"]["comboSizes"] == [2, 3]
    assert report["searchConfig"]["resultLimit"] == 20
    assert task.func == factor_combo_api._background_refresh_combo_rankings
    assert task.args[0] == "BTCUSDT"
    assert task.args[1] is None
    assert config.base_factor_limit == 9
    assert config.combo_sizes == (2, 3)
    assert config.result_limit == 20


def test_factor_ranking_background_refresh_failure_is_visible(monkeypatch) -> None:
    reset_background_loop_statuses()
    monkeypatch.setattr(
        factors_api,
        "refresh_symbol_rankings",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("ranking refresh failed")),
    )

    try:
        factors_api._background_refresh_rankings("BTCUSDT", "10m")
    except RuntimeError as exc:
        assert str(exc) == "ranking refresh failed"
    else:
        raise AssertionError("factor ranking background refresh failure was not exposed")

    status = background_loop_statuses()["factor_ranking"]
    assert status["status"] == "failed"
    assert status["lastError"] == "ranking refresh failed"
    assert status["lastFailureDetails"] == {
        "stage": "manual_api_refresh",
        "symbol": "BTCUSDT",
        "duration": "10m",
    }


def test_factor_combo_background_refresh_failure_is_visible(monkeypatch) -> None:
    reset_background_loop_statuses()
    monkeypatch.setattr(
        factor_combo_api,
        "refresh_symbol_combination_rankings",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("combo refresh failed")),
    )

    try:
        factor_combo_api._background_refresh_combo_rankings("BTCUSDT", "10m", None)
    except RuntimeError as exc:
        assert str(exc) == "combo refresh failed"
    else:
        raise AssertionError("factor combo background refresh failure was not exposed")

    status = background_loop_statuses()["factor_combo_daily"]
    assert status["status"] == "failed"
    assert status["lastError"] == "combo refresh failed"
    assert status["lastFailureDetails"] == {
        "stage": "manual_api_refresh",
        "symbol": "BTCUSDT",
        "duration": "10m",
    }


def _forbidden(name: str):
    def _raise(*_args, **_kwargs):
        raise AssertionError(f"{name} should not be called when full profile is blocked")

    return _raise
