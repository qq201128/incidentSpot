from __future__ import annotations

from app.services import lstm_candidate_retry as retry

ENTRY_OPEN_TIME = 1_778_112_000_000


def test_candidate_retry_finishes_progress_when_candidate_record_fails(monkeypatch) -> None:
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
        refresh_klines=lambda *_args: None,
        run_combination_ranking=lambda *_args: _ranking(),
        save_combination_ranking=lambda *_args: None,
        promote_combinations=lambda *_args: {"promoted": 0},
        train_lstm=lambda _config: _training_report("validation_failed"),
        attempted_keys=lambda *_args: frozenset(),
        record_candidate=_raise_permission_error,
        publish_trade_candidate=_forbidden("publish_trade_candidate"),
        start_progress=lambda **kwargs: calls.append(("progress_start", kwargs["total"])) or {},
        complete_progress=lambda **kwargs: calls.append(("progress_complete", kwargs["completed"])) or {},
        finish_progress=lambda **kwargs: calls.append(("progress_finish", kwargs["status"])) or {},
    )
    config = retry.LstmCandidateRetryConfig(
        symbols=("BTCUSDT",),
        durations=("10m",),
        search=_one_candidate_search(),
    )

    try:
        retry.run_lstm_candidate_retry(config, deps)
    except PermissionError as exc:
        assert "candidate_library.json" in str(exc)
    else:
        raise AssertionError("candidate record failure should be exposed")

    assert calls == [("progress_start", 1), ("progress_finish", "failed")]


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


def _training_report(status: str) -> dict:
    return {
        "status": status,
        "modelVersion": "lstm_test",
        "sampleCounts": {"train": 100, "validation": 50, "test": 50},
    }


def _raise_permission_error(*_args, **_kwargs):
    raise PermissionError("[WinError 5] candidate_library.json")


def _forbidden(name: str):
    def _raise(*_args, **_kwargs):
        raise AssertionError(f"{name} should not be called")

    return _raise
