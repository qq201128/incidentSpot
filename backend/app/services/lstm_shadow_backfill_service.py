from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.db.session import get_conn
from app.services.lstm_artifacts import artifact_paths, read_json
from app.services.lstm_config import duration_ms, lstm_shadow_strategy_key
from app.services.lstm_prediction_service import predict_lstm_shadow_predictions
from app.services.prediction_cache_service import save_prediction


def backfill_lstm_shadow_predictions(
    symbol: str,
    duration: str,
    current_entry_open_time: int,
) -> dict[str, Any]:
    sym = symbol.strip().upper()
    entries = missing_lstm_shadow_entry_times(sym, duration, current_entry_open_time)
    if not entries:
        return _summary(sym, duration, current_entry_open_time, (), 0)
    predictions = predict_lstm_shadow_predictions(sym, duration, list(entries))
    saved = sum(1 for prediction in predictions if save_prediction(prediction))
    return _summary(sym, duration, current_entry_open_time, entries, saved)


def missing_lstm_shadow_entry_times(
    symbol: str,
    duration: str,
    current_entry_open_time: int,
) -> tuple[int, ...]:
    sym = symbol.strip().upper()
    start = _collection_start_entry_time(sym, duration, int(current_entry_open_time))
    if start > int(current_entry_open_time):
        return ()
    expected = set(range(start, int(current_entry_open_time) + 1, duration_ms(duration)))
    existing = _existing_lstm_prediction_times(sym, duration, min(expected), max(expected))
    return tuple(sorted(expected - existing))


def _collection_start_entry_time(symbol: str, duration: str, current_entry_open_time: int) -> int:
    trained_at = _active_model_trained_at(symbol, duration)
    if trained_at is None:
        return current_entry_open_time
    interval = duration_ms(duration)
    trained_ms = _parse_iso_ms(trained_at)
    return ((trained_ms // interval) + 1) * interval


def _active_model_trained_at(symbol: str, duration: str) -> str | None:
    version = read_json(artifact_paths(symbol, duration).version) or {}
    value = version.get("trainedAt")
    return str(value) if value else None


def _existing_lstm_prediction_times(
    symbol: str,
    duration: str,
    start_open_time: int,
    end_open_time: int,
) -> set[int]:
    conn = get_conn()
    try:
        rows = conn.execute(
            """
            SELECT open_time
            FROM predictions
            WHERE strategy_key = ? AND symbol = ? AND duration = ?
              AND open_time BETWEEN ? AND ?
            """,
            (
                lstm_shadow_strategy_key(duration),
                symbol.strip().upper(),
                duration,
                int(start_open_time),
                int(end_open_time),
            ),
        ).fetchall()
        return {int(row["open_time"]) for row in rows}
    finally:
        conn.close()


def _parse_iso_ms(value: str) -> int:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


def _summary(
    symbol: str,
    duration: str,
    current_entry_open_time: int,
    entries: tuple[int, ...],
    saved: int,
) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "duration": duration,
        "currentEntryOpenTime": int(current_entry_open_time),
        "missingCount": len(entries),
        "savedCount": int(saved),
        "firstMissingEntryOpenTime": entries[0] if entries else None,
        "lastMissingEntryOpenTime": entries[-1] if entries else None,
    }
