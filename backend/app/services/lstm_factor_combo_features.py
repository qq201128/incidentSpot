from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.services.lstm_factor_combo_feature_store import load_factor_combo_feature_snapshots
from app.services.lstm_combo_ranking import resolve_lstm_combo_ranking
from app.services.rule_config import MS_PER_MINUTE, horizon_minutes_for_duration

FACTOR_COMBO_FEATURE_PREFIX = "factor_combo_"
TOP_RANK_LIMIT = 3
LOW_SAMPLE_THRESHOLD = 30

ComboSnapshotLoader = Callable[[str, str], list[dict[str, Any]]]


@dataclass(frozen=True)
class FactorComboFeatureResult:
    frame: pd.DataFrame
    metadata: dict[str, Any]


def attach_training_factor_combo_features(
    frame: pd.DataFrame,
    symbol: str,
    duration: str,
    *,
    snapshots_loader: ComboSnapshotLoader | None = None,
) -> FactorComboFeatureResult:
    snapshots = _load_snapshots(symbol, duration, snapshots_loader)
    if not snapshots:
        raise ValueError(
            "historical factor combo feature snapshots are required for model-family training; "
            "prepare factor_combo_feature_snapshots before training"
        )
    indexed = _snapshots_by_entry(snapshots)
    out = frame.sort_values("entry_open_time").reset_index(drop=True).copy()
    rows = [_features_for_entry(indexed.get(int(entry))) for entry in out["entry_open_time"]]
    features = pd.DataFrame(rows)
    result = pd.concat([out, features], axis=1)
    return FactorComboFeatureResult(result, _metadata(snapshots, features, "historical_replay"))


def attach_live_factor_combo_features(frame: pd.DataFrame, symbol: str, duration: str) -> FactorComboFeatureResult:
    ranking = resolve_lstm_combo_ranking(symbol, duration)
    snapshot = _live_snapshot(frame, ranking)
    features = pd.DataFrame([_features_for_entry(snapshot) for _ in range(len(frame))])
    out = pd.concat([frame.reset_index(drop=True).copy(), features], axis=1)
    return FactorComboFeatureResult(out, _metadata([snapshot] if snapshot else [], features, "current_live_ranking"))


def factor_combo_feature_columns(columns: list[str]) -> list[str]:
    return [column for column in columns if column.startswith(FACTOR_COMBO_FEATURE_PREFIX)]


def _load_snapshots(
    symbol: str,
    duration: str,
    snapshots_loader: ComboSnapshotLoader | None,
) -> list[dict[str, Any]]:
    loader = snapshots_loader or load_factor_combo_feature_snapshots
    return [dict(item) for item in loader(symbol.strip().upper(), duration)]


def _snapshots_by_entry(snapshots: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for snapshot in snapshots:
        entry = _entry_open_time(snapshot)
        indexed[entry] = snapshot
    return indexed


def _features_for_entry(snapshot: dict[str, Any] | None) -> dict[str, float]:
    rows = _ranking_rows(snapshot)
    payload = _base_features(rows)
    previous = _previous_top_name(snapshot)
    names = [_factor_name(row) for row in rows[:TOP_RANK_LIMIT]]
    for rank in range(1, TOP_RANK_LIMIT + 1):
        payload.update(_top_features(rank, rows[rank - 1] if len(rows) >= rank else None, previous))
    payload.update(_vote_features(rows))
    payload[f"{FACTOR_COMBO_FEATURE_PREFIX}top_rank_changed"] = float(bool(previous and names and previous != names[0]))
    return payload


def _base_features(rows: list[dict[str, Any]]) -> dict[str, float]:
    missing = not rows
    low_sample = any(_numeric(row, "totalPeriods") < LOW_SAMPLE_THRESHOLD for row in rows[:TOP_RANK_LIMIT])
    return {
        f"{FACTOR_COMBO_FEATURE_PREFIX}missing": float(missing),
        f"{FACTOR_COMBO_FEATURE_PREFIX}top_count": float(min(len(rows), TOP_RANK_LIMIT)),
        f"{FACTOR_COMBO_FEATURE_PREFIX}low_sample": float(bool(rows and low_sample)),
    }


def _top_features(rank: int, row: dict[str, Any] | None, previous_top_name: str | None) -> dict[str, float]:
    prefix = f"{FACTOR_COMBO_FEATURE_PREFIX}top{rank}_"
    if row is None:
        return _empty_top_features(prefix)
    direction = _direction_value(row)
    member_count = len(_members(row))
    return {
        f"{prefix}direction": direction,
        f"{prefix}score": _numeric(row, "factorScore", "score"),
        f"{prefix}win_rate": _numeric(row, "winRate"),
        f"{prefix}profit_factor": _numeric(row, "profitFactor"),
        f"{prefix}ir": _numeric(row, "ir", "informationRatio", "sharpe"),
        f"{prefix}sample_count": _numeric(row, "totalPeriods", "sampleCount"),
        f"{prefix}member_count": float(member_count),
        f"{prefix}member_direction_consensus": _member_direction_consensus(row),
        f"{prefix}changed_from_previous": float(bool(rank == 1 and previous_top_name and previous_top_name != _factor_name(row))),
        f"{prefix}direction_missing": float(direction == 0.0),
        f"{prefix}low_sample": float(_numeric(row, "totalPeriods", "sampleCount") < LOW_SAMPLE_THRESHOLD),
    }


def _empty_top_features(prefix: str) -> dict[str, float]:
    keys = (
        "direction", "score", "win_rate", "profit_factor", "ir", "sample_count",
        "member_count", "member_direction_consensus", "changed_from_previous",
        "direction_missing", "low_sample",
    )
    return {f"{prefix}{key}": 0.0 for key in keys}


def _vote_features(rows: list[dict[str, Any]]) -> dict[str, float]:
    votes = [_direction_value(row) for row in rows[:TOP_RANK_LIMIT]]
    non_zero = [vote for vote in votes if vote != 0.0]
    total = float(sum(non_zero))
    return {
        f"{FACTOR_COMBO_FEATURE_PREFIX}direction_vote": total,
        f"{FACTOR_COMBO_FEATURE_PREFIX}direction_vote_mean": float(total / len(non_zero)) if non_zero else 0.0,
        f"{FACTOR_COMBO_FEATURE_PREFIX}direction_disagreement": _direction_disagreement(non_zero),
    }


def _metadata(snapshots: list[dict[str, Any]], features: pd.DataFrame, source: str) -> dict[str, Any]:
    entries = [_entry_open_time(snapshot) for snapshot in snapshots] if snapshots else []
    missing_rate = float(features[f"{FACTOR_COMBO_FEATURE_PREFIX}missing"].mean()) if len(features) else 0.0
    return {
        "enabled": True,
        "source": source,
        "snapshotCount": int(len(snapshots)),
        "entryOpenTimeMin": int(min(entries)) if entries else None,
        "entryOpenTimeMax": int(max(entries)) if entries else None,
        "missingRate": missing_rate,
        "featureColumns": list(features.columns),
        "lowSampleThreshold": LOW_SAMPLE_THRESHOLD,
    }


def _live_snapshot(frame: pd.DataFrame, ranking: dict[str, Any] | None) -> dict[str, Any] | None:
    if ranking is None:
        return None
    entry = int(frame["entry_open_time"].iloc[-1]) if len(frame) and "entry_open_time" in frame.columns else 0
    return {"entryOpenTime": entry, "ranking": ranking.get("ranking") or []}


def _ranking_rows(snapshot: dict[str, Any] | None) -> list[dict[str, Any]]:
    rows = None if snapshot is None else snapshot.get("ranking")
    return [dict(row) for row in rows] if isinstance(rows, list) else []


def _direction_value(row: dict[str, Any]) -> float:
    value = row.get("direction") or row.get("signal") or row.get("prediction")
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"up", "long", "buy"}:
            return 1.0
        if lowered in {"down", "short", "sell"}:
            return -1.0
    number = _finite_float(value)
    if number is None:
        return 0.0
    return float(np.sign(number))


def _member_direction_consensus(row: dict[str, Any]) -> float:
    orientations = [_member_orientation(member) for member in _members(row)]
    if not orientations:
        return 0.0
    return float(abs(sum(orientations)) / len(orientations))


def _direction_disagreement(votes: list[float]) -> float:
    if len(votes) < 2:
        return 0.0
    return float(len({np.sign(vote) for vote in votes}) > 1)


def _numeric(row: dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = _finite_float(row.get(key))
        if value is not None:
            return value
    return 0.0


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _members(row: dict[str, Any]) -> list[dict[str, Any]]:
    members = row.get("members")
    return [dict(member) for member in members] if isinstance(members, list) else []


def _member_orientation(member: dict[str, Any]) -> float:
    return float(np.sign(_finite_float(member.get("orientation")) or 1.0))


def _factor_name(row: dict[str, Any]) -> str:
    return str(row.get("factorName") or row.get("name") or "")


def _previous_top_name(snapshot: dict[str, Any] | None) -> str | None:
    value = None if snapshot is None else snapshot.get("previousTopFactorName")
    return str(value) if value else None


def _entry_open_time(snapshot: dict[str, Any]) -> int:
    return int(snapshot.get("entryOpenTime") or snapshot.get("entry_open_time"))


def duration_stale_ms(duration: str) -> int:
    return horizon_minutes_for_duration(duration) * MS_PER_MINUTE
