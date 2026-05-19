from __future__ import annotations

from fastapi import BackgroundTasks

from app.api import factor_combinations as factor_combo_api
from app.api import lstm as lstm_api
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


def test_lstm_train_route_uses_profile_defaults_and_overrides(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_train(config):
        captured["config"] = config
        return {"status": "trained", "symbol": config.symbol, "profile": "captured"}

    monkeypatch.setattr(lstm_api, "train_lstm_model", _fake_train)

    report = lstm_api.lstm_train(
        symbol="btcusdt",
        duration="10m",
        profile="fast",
        feature_window=40,
        epochs=None,
        batch_size=32,
        min_samples=99,
        learning_rate=None,
        hidden_size=None,
        num_layers=None,
        min_move_bps=None,
    )

    config = captured["config"]
    assert report["status"] == "trained"
    assert config.symbol == "BTCUSDT"
    assert config.duration == "10m"
    assert config.feature_window == 40
    assert config.epochs == 2
    assert config.batch_size == 32
    assert config.min_samples == 99
    assert config.hidden_size == 32
    assert config.num_layers == 1


def test_lstm_candidate_search_route_queues_background(monkeypatch) -> None:
    queued = {}

    def _fake_queue(**kwargs):
        queued.update(kwargs)
        return {
            "status": "queued",
            "symbol": kwargs["symbol"],
            "duration": kwargs["duration"],
            "total": kwargs["total"],
            "completed": 0,
            "percent": 0.0,
            "parallelWorkers": kwargs["parallel_workers"],
        }

    monkeypatch.setattr(lstm_api, "queue_lstm_candidate_progress", _fake_queue)
    monkeypatch.setattr(lstm_api, "search_space_size", lambda _config: 225)
    monkeypatch.setattr(
        lstm_api,
        "lstm_model_status",
        lambda *_args, **_kwargs: {
            "status": "shadow_active",
            "candidateSearchProgress": {
                "status": "queued",
                "completed": 0,
                "total": 225,
                "parallelWorkers": 10,
            },
            "shadow": {"status": "shadow_active"},
        },
    )

    tasks = BackgroundTasks()
    report = lstm_api.lstm_candidate_search(tasks, symbol="btcusdt", duration="10m", profile="full")

    assert report["message"] == "LSTM候选搜索已排队。"
    assert report["candidateSearchProgress"]["status"] == "queued"
    assert tasks.tasks[0].func == lstm_api._background_lstm_candidate_search
    assert tasks.tasks[0].args == ("BTCUSDT", "10m", "full")
    assert queued["symbol"] == "BTCUSDT"
    assert queued["total"] == 225
    assert queued["parallel_workers"] == 10


def test_lstm_candidate_search_background_finishes_skipped(monkeypatch) -> None:
    configs = []
    finished = []

    def _fake_retry(config):
        configs.append(config)
        return {"status": "skipped"}

    monkeypatch.setattr(lstm_api, "run_lstm_candidate_retry", _fake_retry)
    monkeypatch.setattr(
        lstm_api,
        "finish_lstm_candidate_progress",
        lambda **kwargs: finished.append(kwargs),
    )

    lstm_api._background_lstm_candidate_search("BTCUSDT", "10m", "full")

    assert configs[0].symbols == ("BTCUSDT",)
    assert configs[0].durations == ("10m",)
    assert configs[0].manual_trigger is True
    assert finished[0]["symbol"] == "BTCUSDT"
    assert finished[0]["duration"] == "10m"
    assert finished[0]["status"] == "skipped"


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


def _forbidden(name: str):
    def _raise(*_args, **_kwargs):
        raise AssertionError(f"{name} should not be called when full profile is blocked")

    return _raise
