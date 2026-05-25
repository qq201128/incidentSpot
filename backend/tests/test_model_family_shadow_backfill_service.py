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
