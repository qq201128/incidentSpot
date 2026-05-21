from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.db.session import get_conn
from app.services.lstm_artifacts import artifact_paths, read_json
from app.services.lstm_config import duration_ms
from app.services.model_family_config import model_family_strategy_key, normalize_model_family
from app.services.model_family_prediction_service import predict_model_family_shadow_predictions
from app.services.prediction_cache_service import save_prediction


def backfill_model_family_shadow_predictions(
    family: str,
    symbol: str,
    duration: str,
    current_entry_open_time: int,
) -> dict[str, Any]:
    selected = normalize_model_family(family)
    sym = symbol.strip().upper()
    entries = missing_model_family_shadow_entry_times(selected, sym, duration, current_entry_open_time)
    if not entries:
        return _summary(selected, sym, duration, current_entry_open_time, (), 0)
    predictions = predict_model_family_shadow_predictions(selected, sym, duration, list(entries))
    saved = sum(1 for prediction in predictions if save_prediction(prediction))
    return _summary(selected, sym, duration, current_entry_open_time, entries, saved)


def missing_model_family_shadow_entry_times(
    family: str,
    symbol: str,
    duration: str,
    current_entry_open_time: int,
) -> tuple[int, ...]:
    selected = normalize_model_family(family)
    sym = symbol.strip().upper()
    start = _collection_start_entry_time(selected, sym, duration, int(current_entry_open_time))
    if start > int(current_entry_open_time):
        return ()
    expected = set(range(start, int(current_entry_open_time) + 1, duration_ms(duration)))
    existing = _existing_prediction_times(selected, sym, duration, min(expected), max(expected))
    return tuple(sorted(expected - existing))


def _collection_start_entry_time(family: str, symbol: str, duration: str, current_entry_open_time: int) -> int:
    version = read_json(artifact_paths(symbol, duration, family=family).version) or {}
    trained_at = version.get("trainedAt")
    if not trained_at:
        return current_entry_open_time
    interval = duration_ms(duration)
    trained_ms = _parse_iso_ms(str(trained_at))
    return ((trained_ms // interval) + 1) * interval


def _existing_prediction_times(family: str, symbol: str, duration: str, start_open_time: int, end_open_time: int) -> set[int]:
    conn = get_conn()
    try:
        rows = conn.execute(
            """
            SELECT open_time
            FROM predictions
            WHERE signal_key = ? AND symbol = ? AND duration = ?
              AND open_time BETWEEN ? AND ?
            """,
            (model_family_strategy_key(family, duration), symbol, duration, int(start_open_time), int(end_open_time)),
        ).fetchall()
        return {int(row["open_time"]) for row in rows}
    finally:
        conn.close()


def _parse_iso_ms(value: str) -> int:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


def _summary(family: str, symbol: str, duration: str, current_entry_open_time: int, entries: tuple[int, ...], saved: int) -> dict:
    return {
        "modelFamily": family,
        "symbol": symbol,
        "duration": duration,
        "currentEntryOpenTime": int(current_entry_open_time),
        "missingCount": len(entries),
        "savedCount": int(saved),
        "firstMissingEntryOpenTime": entries[0] if entries else None,
        "lastMissingEntryOpenTime": entries[-1] if entries else None,
    }
