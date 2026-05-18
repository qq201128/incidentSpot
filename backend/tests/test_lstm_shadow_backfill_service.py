from __future__ import annotations

from app.services import lstm_shadow_backfill_service as service


def test_missing_lstm_shadow_entry_times_finds_gaps_after_model_training(monkeypatch) -> None:
    monkeypatch.setattr(service, "_active_model_trained_at", lambda *_args: "2026-05-14T16:31:14+00:00")
    monkeypatch.setattr(
        service,
        "_existing_lstm_prediction_times",
        lambda *_args: {1778776800000, 1778778000000},
    )

    missing = service.missing_lstm_shadow_entry_times(
        "BTCUSDT",
        "10m",
        1778778600000,
    )

    assert missing == (1778777400000, 1778778600000)


def test_backfill_lstm_shadow_predictions_saves_batch_predictions(monkeypatch) -> None:
    saved = []
    entries = (1778777400000, 1778778600000)

    monkeypatch.setattr(service, "missing_lstm_shadow_entry_times", lambda *_args: entries)
    monkeypatch.setattr(
        service,
        "predict_lstm_shadow_predictions",
        lambda *_args: [{"open_time": entry, "symbol": "BTCUSDT", "duration": "10m"} for entry in entries],
    )
    monkeypatch.setattr(service, "save_prediction", lambda prediction: saved.append(prediction["open_time"]) or True)

    summary = service.backfill_lstm_shadow_predictions("BTCUSDT", "10m", 1778778600000)

    assert saved == list(entries)
    assert summary["missingCount"] == 2
    assert summary["savedCount"] == 2
