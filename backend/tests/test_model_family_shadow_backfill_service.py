from __future__ import annotations

import pandas as pd

from app.services import model_family_shadow_backfill_service as service

DEFAULT_DURATION = "10m"
INTERVAL_MS = 600_000


def test_missing_shadow_entries_exclude_unavailable_feature_rows(monkeypatch) -> None:
    existing_calls = []
    current_entry = 4 * INTERVAL_MS
    start_entry = 2 * INTERVAL_MS
    feature_frame = pd.DataFrame({"open_time": [INTERVAL_MS, 2 * INTERVAL_MS]})

    monkeypatch.setattr(service, "_collection_start_entry_time", lambda *_args: start_entry)
    monkeypatch.setattr(service, "load_lstm_market_frame", lambda *_args: feature_frame)
    monkeypatch.setattr(
        service,
        "_existing_prediction_times",
        lambda *_args: existing_calls.append(_args) or {start_entry},
    )

    missing = service.missing_model_family_shadow_entry_times(
        "lstm",
        "btcusdt",
        DEFAULT_DURATION,
        current_entry,
    )

    assert missing == (3 * INTERVAL_MS,)
    assert existing_calls == [("lstm", "BTCUSDT", DEFAULT_DURATION, start_entry, 3 * INTERVAL_MS)]


def test_missing_shadow_entries_skip_database_when_no_feature_rows(monkeypatch) -> None:
    current_entry = 4 * INTERVAL_MS
    start_entry = 2 * INTERVAL_MS
    feature_frame = pd.DataFrame({"open_time": [0]})

    def fail_existing_lookup(*_args) -> set[int]:
        raise AssertionError("database should not be queried")

    monkeypatch.setattr(service, "_collection_start_entry_time", lambda *_args: start_entry)
    monkeypatch.setattr(service, "load_lstm_market_frame", lambda *_args: feature_frame)
    monkeypatch.setattr(service, "_existing_prediction_times", fail_existing_lookup)

    missing = service.missing_model_family_shadow_entry_times(
        "lstm",
        "btcusdt",
        DEFAULT_DURATION,
        current_entry,
    )

    assert missing == ()


def test_backfill_model_family_shadow_predictions_limits_entries(monkeypatch) -> None:
    predicted_entries = []
    saved_predictions = []
    entries = (INTERVAL_MS, 2 * INTERVAL_MS, 3 * INTERVAL_MS, 4 * INTERVAL_MS)

    monkeypatch.setattr(service, "missing_model_family_shadow_entry_times", lambda *_args: entries)
    monkeypatch.setattr(
        service,
        "predict_model_family_shadow_predictions",
        lambda _family, _symbol, _duration, selected, **_kwargs: predicted_entries.extend(selected)
        or [{"open_time": item} for item in selected],
    )
    monkeypatch.setattr(service, "save_prediction", lambda prediction: saved_predictions.append(prediction) or True)

    summary = service.backfill_model_family_shadow_predictions(
        "lstm",
        "btcusdt",
        DEFAULT_DURATION,
        4 * INTERVAL_MS,
        max_entries=2,
    )

    assert predicted_entries == [INTERVAL_MS, 2 * INTERVAL_MS]
    assert saved_predictions == [{"open_time": INTERVAL_MS}, {"open_time": 2 * INTERVAL_MS}]
    assert summary["missingCount"] == 2
    assert summary["remainingMissingCount"] == 2
    assert summary["savedCount"] == 2


def test_backfill_model_family_shadow_predictions_can_select_current_entry_only(monkeypatch) -> None:
    predicted_entries = []
    saved_predictions = []
    entries = (INTERVAL_MS, 2 * INTERVAL_MS, 3 * INTERVAL_MS, 4 * INTERVAL_MS)

    monkeypatch.setattr(service, "missing_model_family_shadow_entry_times", lambda *_args: entries)
    monkeypatch.setattr(
        service,
        "predict_model_family_shadow_predictions",
        lambda _family, _symbol, _duration, selected, **_kwargs: predicted_entries.extend(selected)
        or [{"open_time": item} for item in selected],
    )
    monkeypatch.setattr(service, "save_prediction", lambda prediction: saved_predictions.append(prediction) or True)

    summary = service.backfill_model_family_shadow_predictions(
        "lstm",
        "btcusdt",
        DEFAULT_DURATION,
        4 * INTERVAL_MS,
        max_entries=2,
        current_entry_only=True,
    )

    assert predicted_entries == [4 * INTERVAL_MS]
    assert saved_predictions == [{"open_time": 4 * INTERVAL_MS}]
    assert summary["missingCount"] == 1
    assert summary["remainingMissingCount"] == 3
    assert summary["savedCount"] == 1


def test_backfill_model_family_shadow_predictions_skips_history_when_current_exists(monkeypatch) -> None:
    entries = (INTERVAL_MS, 2 * INTERVAL_MS)

    monkeypatch.setattr(service, "missing_model_family_shadow_entry_times", lambda *_args: entries)

    def fail_predict(*_args, **_kwargs):
        raise AssertionError("historical gaps should not be predicted in current-only mode")

    monkeypatch.setattr(service, "predict_model_family_shadow_predictions", fail_predict)

    summary = service.backfill_model_family_shadow_predictions(
        "lstm",
        "btcusdt",
        DEFAULT_DURATION,
        4 * INTERVAL_MS,
        current_entry_only=True,
    )

    assert summary["missingCount"] == 0
    assert summary["remainingMissingCount"] == 2
    assert summary["savedCount"] == 0
