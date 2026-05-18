from __future__ import annotations

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
    )

    report = retry.run_lstm_candidate_retry(
        retry.LstmCandidateRetryConfig(symbols=("btcusdt",), durations=("10m",), profile=EXPERIMENT_PROFILE_FAST),
        deps,
    )

    assert report["status"] == "trained"
    assert calls[0] == ("refresh", ("BTCUSDT", "1m", ENTRY_OPEN_TIME - retry.MS_PER_MINUTE))
    assert calls[1] == ("refresh", ("BTCUSDT", "10m", ENTRY_OPEN_TIME - (10 * retry.MS_PER_MINUTE)))
    assert calls[-1][0] == "train"
    assert calls[-1][1].feature_window == 32


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
    )

    report = retry.run_lstm_candidate_retry(
        retry.LstmCandidateRetryConfig(symbols=("BTCUSDT",), durations=("10m",)),
        deps,
    )

    result = report["results"][0]
    assert report["status"] == "skipped"
    assert result["status"] == "skipped"
    assert result["reason"] == "active_model_ready"


def _ranking() -> dict:
    return {
        "symbol": "BTCUSDT",
        "duration": "10m",
        "ranking": [{"factorName": "combo_a"}],
    }


def _training_report() -> dict:
    return {
        "status": "trained",
        "modelVersion": "lstm_test",
        "sampleCounts": {"train": 100, "validation": 50, "test": 50},
    }


def _forbidden(name: str):
    def _raise(*_args, **_kwargs):
        raise AssertionError(f"{name} should not be called")

    return _raise
