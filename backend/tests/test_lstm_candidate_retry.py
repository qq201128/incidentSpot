from __future__ import annotations

import threading

from app.services import lstm_candidate_retry as retry
from app.services.experiment_profiles import EXPERIMENT_PROFILE_FAST

ENTRY_OPEN_TIME = 1_778_112_000_000


def test_candidate_retry_trains_when_model_is_untrained(monkeypatch) -> None:
    calls = []

    monkeypatch.setattr(
        retry,
        "current_rule_entry_open_time_for_duration",
        lambda _duration: ENTRY_OPEN_TIME,
    )
    deps = retry.LstmCandidateRetryDependencies(
        lstm_status=lambda *_args: {
            "activeModelStatus": "untrained",
            "shadowPredictionReady": False,
            "comboSnapshotMatches": True,
            "artifactsReady": False,
        },
        refresh_klines=lambda *args: calls.append(("refresh", args)),
        run_combination_ranking=lambda *args: calls.append(("ranking", args)) or _ranking(),
        save_combination_ranking=lambda report: calls.append(("save", report["symbol"])),
        promote_combinations=lambda report: calls.append(("promote", report["duration"])) or {"promoted": 1},
        train_lstm=lambda config: calls.append(("train", config)) or _training_report(),
        attempted_keys=lambda *_args: frozenset(),
        record_candidate=lambda config, profile, report: calls.append(("record", config.feature_window)) or report,
        publish_trade_candidate=lambda config, report: calls.append(("publish", config.feature_window, report["status"])),
        start_progress=lambda **kwargs: calls.append(("progress_start", kwargs["total"])) or {},
        complete_progress=lambda **kwargs: calls.append(("progress_complete", kwargs["completed"])) or {},
        finish_progress=lambda **kwargs: calls.append(("progress_finish", kwargs["status"])) or {},
    )

    report = retry.run_lstm_candidate_retry(
        retry.LstmCandidateRetryConfig(
            symbols=("btcusdt",),
            durations=("10m",),
            profile=EXPERIMENT_PROFILE_FAST,
            search=_one_candidate_search(),
        ),
        deps,
    )

    assert report["status"] == "trained"
    assert calls[0] == ("progress_start", 1)
    assert calls[1] == ("refresh", ("BTCUSDT", "1m", ENTRY_OPEN_TIME - retry.MS_PER_MINUTE))
    assert calls[2] == ("refresh", ("BTCUSDT", "10m", ENTRY_OPEN_TIME - (10 * retry.MS_PER_MINUTE)))
    assert ("progress_complete", 1) in calls
    assert ("progress_finish", "trade_active") in calls
    assert [item for item in calls if item[0] == "train"][0][1].feature_window == 24
    assert ("record", 24) in calls
    assert ("publish", 24, "trained") in calls


def test_candidate_retry_skips_ready_active_model() -> None:
    deps = retry.LstmCandidateRetryDependencies(
        lstm_status=lambda *_args: {
            "activeModelStatus": "trained",
            "shadowPredictionReady": True,
            "comboSnapshotMatches": True,
            "artifactsReady": True,
        },
        refresh_klines=_forbidden("refresh_klines"),
        run_combination_ranking=_forbidden("run_combination_ranking"),
        save_combination_ranking=_forbidden("save_combination_ranking"),
        promote_combinations=_forbidden("promote_combinations"),
        train_lstm=_forbidden("train_lstm"),
        attempted_keys=lambda *_args: frozenset(),
        record_candidate=_forbidden("record_candidate"),
        start_progress=_forbidden("start_progress"),
        complete_progress=_forbidden("complete_progress"),
        finish_progress=_forbidden("finish_progress"),
    )

    report = retry.run_lstm_candidate_retry(
        retry.LstmCandidateRetryConfig(symbols=("BTCUSDT",), durations=("10m",)),
        deps,
    )

    result = report["results"][0]
    assert report["status"] == "skipped"
    assert result["status"] == "skipped"
    assert result["reason"] == "active_model_ready"


def test_candidate_retry_continues_search_when_shadow_active(monkeypatch) -> None:
    calls = []

    monkeypatch.setattr(
        retry,
        "current_rule_entry_open_time_for_duration",
        lambda _duration: ENTRY_OPEN_TIME,
    )
    deps = retry.LstmCandidateRetryDependencies(
        lstm_status=lambda *_args: {
            "activeModelStatus": "shadow_active",
            "lastAttemptStatus": "shadow_active",
            "shadowPredictionReady": True,
            "comboSnapshotMatches": True,
            "artifactsReady": True,
        },
        refresh_klines=lambda *args: calls.append(("refresh", args)),
        run_combination_ranking=lambda *args: calls.append(("ranking", args)) or _ranking(),
        save_combination_ranking=lambda report: calls.append(("save", report["symbol"])),
        promote_combinations=lambda report: calls.append(("promote", report["duration"])) or {"promoted": 0},
        train_lstm=lambda config: calls.append(("train", config)) or _training_report("validation_failed"),
        attempted_keys=lambda *_args: frozenset(),
        record_candidate=lambda config, profile, report: calls.append(("record", profile, config.feature_window)) or report,
        publish_trade_candidate=_forbidden("publish_trade_candidate"),
        start_progress=lambda **kwargs: calls.append(("progress_start", kwargs["total"])) or {},
        complete_progress=lambda **kwargs: calls.append(("progress_complete", kwargs["completed"])) or {},
        finish_progress=lambda **kwargs: calls.append(("progress_finish", kwargs["status"])) or {},
    )

    report = retry.run_lstm_candidate_retry(
        retry.LstmCandidateRetryConfig(
            symbols=("BTCUSDT",),
            durations=("10m",),
            profile=EXPERIMENT_PROFILE_FAST,
            search=_one_candidate_search(),
        ),
        deps,
    )

    result = report["results"][0]
    assert result["reason"] == "shadow_active_candidate_search"
    assert result["status"] == "validation_failed"
    assert ("progress_start", 1) in calls
    assert ("progress_complete", 1) in calls
    assert ("progress_finish", "validation_failed") in calls
    assert [item for item in calls if item[0] == "train"][0][1].feature_window == 24
    assert ("record", EXPERIMENT_PROFILE_FAST, 24) in calls


def test_candidate_retry_trains_candidates_with_limited_parallel_workers(monkeypatch) -> None:
    calls = []

    monkeypatch.setattr(
        retry,
        "current_rule_entry_open_time_for_duration",
        lambda _duration: ENTRY_OPEN_TIME,
    )
    deps = retry.LstmCandidateRetryDependencies(
        lstm_status=lambda *_args: {
            "activeModelStatus": "shadow_active",
            "shadowPredictionReady": True,
            "comboSnapshotMatches": True,
            "artifactsReady": True,
        },
        refresh_klines=lambda *_args: None,
        run_combination_ranking=lambda *args: calls.append(("ranking", args)) or _ranking(),
        save_combination_ranking=lambda *_args: None,
        promote_combinations=lambda *_args: {"promoted": 0},
        train_lstm=lambda config: calls.append(("train", config.feature_window)) or _training_report("validation_failed"),
        attempted_keys=lambda *_args: frozenset(),
        record_candidate=lambda config, profile, report: calls.append(("record", config.feature_window)) or report,
        publish_trade_candidate=_forbidden("publish_trade_candidate"),
        start_progress=lambda **kwargs: calls.append(("progress_start", kwargs["total"], kwargs["parallel_workers"])) or {},
        complete_progress=lambda **kwargs: calls.append(("progress_complete", kwargs["completed"])) or {},
        finish_progress=lambda **kwargs: calls.append(("progress_finish", kwargs["status"])) or {},
    )
    config = retry.LstmCandidateRetryConfig(
        symbols=("BTCUSDT",),
        durations=("10m",),
        search=retry.LstmCandidateSearchConfig(
            feature_windows=(24, 32, 48),
            min_move_bps_values=(8.0,),
            epoch_values=(8,),
            seeds=(20260513,),
            parallel_workers=2,
        ),
    )

    report = retry.run_lstm_candidate_retry(config, deps)

    assert report["results"][0]["status"] == "validation_failed"
    assert sorted(item[1] for item in calls if item[0] == "train") == [24, 32, 48]
    assert sorted(item[1] for item in calls if item[0] == "record") == [24, 32, 48]
    assert ("progress_start", 3, 2) in calls
    assert [item for item in calls if item[0] == "progress_complete"] == [
        ("progress_complete", 1),
        ("progress_complete", 2),
        ("progress_complete", 3),
    ]


def test_parallel_candidate_reports_yield_completed_candidate_before_slow_candidate_finishes() -> None:
    release_slow = threading.Event()
    configs = [
        _training_config(24),
        _training_config(32),
    ]

    def _train(config):
        if config.feature_window == 32:
            assert release_slow.wait(timeout=5)
        return _training_report("validation_failed")

    deps = retry.LstmCandidateRetryDependencies(
        lstm_status=_forbidden("lstm_status"),
        refresh_klines=_forbidden("refresh_klines"),
        run_combination_ranking=_forbidden("run_combination_ranking"),
        save_combination_ranking=_forbidden("save_combination_ranking"),
        promote_combinations=_forbidden("promote_combinations"),
        train_lstm=_train,
        record_candidate=_forbidden("record_candidate"),
        start_progress=_forbidden("start_progress"),
        complete_progress=_forbidden("complete_progress"),
        finish_progress=_forbidden("finish_progress"),
    )

    reports = retry._train_candidate_reports(configs, EXPERIMENT_PROFILE_FAST, 2, deps)
    first = next(reports)
    release_slow.set()
    rest = list(reports)

    assert first.config.feature_window == 24
    assert [item.config.feature_window for item in rest] == [32]


def test_candidate_retry_skips_exhausted_search_space() -> None:
    first_key = "profile=fast|duration=10m|window=24|move_bps=8|epochs=8|seed=20260513"
    deps = retry.LstmCandidateRetryDependencies(
        lstm_status=lambda *_args: {
            "activeModelStatus": "shadow_active",
            "shadowPredictionReady": True,
            "comboSnapshotMatches": True,
            "artifactsReady": True,
        },
        refresh_klines=_forbidden("refresh_klines"),
        run_combination_ranking=_forbidden("run_combination_ranking"),
        save_combination_ranking=_forbidden("save_combination_ranking"),
        promote_combinations=_forbidden("promote_combinations"),
        train_lstm=_forbidden("train_lstm"),
        attempted_keys=lambda *_args: frozenset({first_key}),
        record_candidate=_forbidden("record_candidate"),
        start_progress=_forbidden("start_progress"),
        complete_progress=_forbidden("complete_progress"),
        finish_progress=_forbidden("finish_progress"),
    )
    config = retry.LstmCandidateRetryConfig(
        symbols=("BTCUSDT",),
        durations=("10m",),
        search=retry.LstmCandidateSearchConfig(
            feature_windows=(24,),
            min_move_bps_values=(8.0,),
            epoch_values=(8,),
            seeds=(20260513,),
        ),
    )

    report = retry.run_lstm_candidate_retry(config, deps)

    result = report["results"][0]
    assert report["status"] == "skipped"
    assert result["reason"] == "candidate_search_exhausted"


def test_candidate_retry_manual_trigger_ignores_stale_training_status(monkeypatch) -> None:
    calls = []

    monkeypatch.setattr(
        retry,
        "current_rule_entry_open_time_for_duration",
        lambda _duration: ENTRY_OPEN_TIME,
    )
    deps = retry.LstmCandidateRetryDependencies(
        lstm_status=lambda *_args: {
            "activeModelStatus": "training",
            "lastAttemptStatus": "training",
            "shadowPredictionReady": True,
            "comboSnapshotMatches": True,
            "artifactsReady": True,
        },
        refresh_klines=lambda *args: calls.append(("refresh", args)),
        run_combination_ranking=lambda *args: calls.append(("ranking", args)) or _ranking(),
        save_combination_ranking=lambda report: calls.append(("save", report["symbol"])),
        promote_combinations=lambda report: calls.append(("promote", report["duration"])) or {"promoted": 1},
        train_lstm=lambda config: calls.append(("train", config)) or _training_report(),
        attempted_keys=lambda *_args: frozenset(),
        record_candidate=lambda config, profile, report: calls.append(("record", config.feature_window)) or report,
        publish_trade_candidate=_forbidden("publish_trade_candidate"),
        start_progress=lambda **kwargs: calls.append(("progress_start", kwargs["total"])) or {},
        complete_progress=lambda **kwargs: calls.append(("progress_complete", kwargs["completed"])) or {},
        finish_progress=lambda **kwargs: calls.append(("progress_finish", kwargs["status"])) or {},
    )

    report = retry.run_lstm_candidate_retry(
        retry.LstmCandidateRetryConfig(
            symbols=("BTCUSDT",),
            durations=("10m",),
            profile=EXPERIMENT_PROFILE_FAST,
            search=_one_candidate_search(),
            manual_trigger=True,
        ),
        deps,
    )

    assert report["status"] == "trained"
    assert ("progress_start", 1) in calls
    assert ("progress_finish", "trade_active") in calls
    assert [item for item in calls if item[0] == "train"]


def _ranking() -> dict:
    return {
        "symbol": "BTCUSDT",
        "duration": "10m",
        "ranking": [{"factorName": "combo_a"}],
    }


def _one_candidate_search() -> retry.LstmCandidateSearchConfig:
    return retry.LstmCandidateSearchConfig(
        feature_windows=(24,),
        min_move_bps_values=(8.0,),
        epoch_values=(8,),
        seeds=(20260513,),
    )


def _training_config(feature_window: int) -> retry.LstmTrainingConfig:
    return retry.LstmTrainingConfig(
        symbol="BTCUSDT",
        duration="10m",
        feature_window=feature_window,
        min_move_bps=8.0,
        epochs=8,
        seed=20260513,
    )


def _training_report(status: str = "trained") -> dict:
    return {
        "status": status,
        "modelVersion": "lstm_test",
        "sampleCounts": {"train": 100, "validation": 50, "test": 50},
    }


def _forbidden(name: str):
    def _raise(*_args, **_kwargs):
        raise AssertionError(f"{name} should not be called")

    return _raise
