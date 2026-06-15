from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.db.session import get_conn
from app.services.lstm_artifacts import artifact_paths, read_json
from app.services.lstm_config import duration_ms
from app.services.lstm_feature_builder import duration_feature_frame
from app.services.lstm_market_feature_builder import load_lstm_market_frame
from app.services.model_family_config import model_family_strategy_key, normalize_model_family
from app.services.model_family_prediction_service import predict_model_family_shadow_predictions
from app.services.prediction_cache_service import save_prediction


@dataclass(frozen=True)
class ShadowBackfillPredictionBatch:
    summary: dict[str, Any]
    predictions: tuple[dict[str, Any], ...]


def backfill_model_family_shadow_predictions(
    family: str,
    symbol: str,
    duration: str,
    current_entry_open_time: int,
    *,
    max_entries: int | None = None,
    current_entry_only: bool = False,
    cycle_context: Any | None = None,
) -> dict[str, Any]:
    batch = build_model_family_shadow_backfill_prediction_batch(
        family,
        symbol,
        duration,
        current_entry_open_time,
        max_entries=max_entries,
        current_entry_only=current_entry_only,
        cycle_context=cycle_context,
    )
    return save_model_family_shadow_backfill_prediction_batch(batch)


def build_model_family_shadow_backfill_prediction_batch(
    family: str,
    symbol: str,
    duration: str,
    current_entry_open_time: int,
    *,
    max_entries: int | None = None,
    current_entry_only: bool = False,
    cycle_context: Any | None = None,
) -> ShadowBackfillPredictionBatch:
    selected = normalize_model_family(family)
    sym = symbol.strip().upper()
    entries = missing_model_family_shadow_entry_times(selected, sym, duration, current_entry_open_time)
    if not entries:
        summary = _summary(selected, sym, duration, current_entry_open_time, (), 0, remaining_count=0)
        return ShadowBackfillPredictionBatch(summary, ())
    selected_entries = _current_entry_entries(entries, current_entry_open_time) if current_entry_only else _limited_entries(entries, max_entries)
    remaining_count = len(entries) - len(selected_entries)
    if not selected_entries:
        summary = _summary(selected, sym, duration, current_entry_open_time, (), 0, remaining_count=remaining_count)
        return ShadowBackfillPredictionBatch(summary, ())
    predictions = predict_model_family_shadow_predictions(
        selected,
        sym,
        duration,
        list(selected_entries),
        cycle_context=cycle_context,
    )
    summary = _summary(
        selected,
        sym,
        duration,
        current_entry_open_time,
        selected_entries,
        0,
        remaining_count=remaining_count,
    )
    return ShadowBackfillPredictionBatch(summary, tuple(predictions))


def save_model_family_shadow_backfill_prediction_batch(batch: ShadowBackfillPredictionBatch) -> dict[str, Any]:
    saved = sum(1 for prediction in batch.predictions if save_prediction(prediction))
    return {**batch.summary, "savedCount": int(saved)}


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
    expected = _predictable_entry_times(sym, duration, start, int(current_entry_open_time))
    if not expected:
        return ()
    existing = _existing_prediction_times(selected, sym, duration, min(expected), max(expected))
    return tuple(sorted(expected - existing))


def _predictable_entry_times(symbol: str, duration: str, start_open_time: int, end_open_time: int) -> set[int]:
    candidates = set(range(int(start_open_time), int(end_open_time) + 1, duration_ms(duration)))
    available = _available_feature_entry_times(symbol, duration, int(end_open_time))
    return candidates & available


def _available_feature_entry_times(symbol: str, duration: str, current_entry_open_time: int) -> set[int]:
    sampled = duration_feature_frame(load_lstm_market_frame(symbol, duration), duration)
    return {
        int(entry)
        for entry in sampled["entry_open_time"].tolist()
        if int(entry) <= int(current_entry_open_time)
    }


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


def _limited_entries(entries: tuple[int, ...], max_entries: int | None) -> tuple[int, ...]:
    if max_entries is None:
        return entries
    limit = max(1, int(max_entries))
    return tuple(entries[:limit])


def _current_entry_entries(entries: tuple[int, ...], current_entry_open_time: int) -> tuple[int, ...]:
    current = int(current_entry_open_time)
    return (current,) if current in entries else ()


def _summary(
    family: str,
    symbol: str,
    duration: str,
    current_entry_open_time: int,
    entries: tuple[int, ...],
    saved: int,
    *,
    remaining_count: int,
) -> dict:
    return {
        "modelFamily": family,
        "symbol": symbol,
        "duration": duration,
        "currentEntryOpenTime": int(current_entry_open_time),
        "missingCount": len(entries),
        "remainingMissingCount": int(remaining_count),
        "savedCount": int(saved),
        "firstMissingEntryOpenTime": entries[0] if entries else None,
        "lastMissingEntryOpenTime": entries[-1] if entries else None,
    }
