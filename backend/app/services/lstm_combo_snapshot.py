from __future__ import annotations

from pathlib import Path
from typing import Any

from app.services.factor_combination_cache_service import get_cached_combination_ranking
from app.services.lstm_artifacts import artifact_paths, read_json

SNAPSHOT_RANK_LIMIT = 3


def combo_snapshot_from_ranking(ranking_report: dict[str, Any]) -> list[dict[str, Any]]:
    ranking = ranking_report.get("ranking") or []
    return [
        _combo_row_snapshot(dict(row), rank)
        for rank, row in enumerate(ranking[:SNAPSHOT_RANK_LIMIT], start=1)
    ]


def combo_snapshot_status(
    symbol: str,
    duration: str,
    *,
    ranking_report: dict[str, Any] | None = None,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    current = current_combo_snapshot(symbol, duration, ranking_report=ranking_report)
    trained = trained_combo_snapshot(symbol, duration, artifact_root=artifact_root)
    if not current:
        return _status(False, current, trained, "current_combo_snapshot_missing")
    if len(current) < SNAPSHOT_RANK_LIMIT:
        return _status(False, current, trained, "current_combo_snapshot_incomplete")
    if not trained:
        return _status(False, current, trained, "trained_combo_snapshot_missing")
    if len(trained) < SNAPSHOT_RANK_LIMIT:
        return _status(False, current, trained, "trained_combo_snapshot_incomplete")
    if current != trained:
        return _status(False, current, trained, "combo_snapshot_mismatch")
    return _status(True, current, trained, "passed")


def current_combo_snapshot(
    symbol: str,
    duration: str,
    *,
    ranking_report: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    ranking = ranking_report or get_cached_combination_ranking(symbol.strip().upper(), duration)
    if ranking is None:
        return []
    return combo_snapshot_from_ranking(ranking)


def trained_combo_snapshot(
    symbol: str,
    duration: str,
    *,
    artifact_root: Path | None = None,
) -> list[dict[str, Any]]:
    paths = artifact_paths(symbol.strip().upper(), duration, artifact_root)
    features = read_json(paths.features) or {}
    snapshot = features.get("comboSnapshot")
    return list(snapshot) if isinstance(snapshot, list) else []


def assert_combo_snapshot_matches(
    ranking_report: dict[str, Any],
    expected: list[dict[str, Any]] | None,
) -> None:
    if not expected:
        return
    current = combo_snapshot_from_ranking(ranking_report)
    if current != expected:
        raise ValueError("current factor combo Top1/Top2/Top3 differs from LSTM training snapshot")


def _combo_row_snapshot(row: dict[str, Any], rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "factorName": row.get("factorName"),
        "members": [member.get("name") for member in _members(row)],
    }


def _members(row: dict[str, Any]) -> list[dict[str, Any]]:
    members = row.get("members")
    if not isinstance(members, list) or not members:
        raise ValueError(f"combination row missing members: {row.get('factorName')}")
    return [dict(member) for member in members]


def _status(
    matches: bool,
    current: list[dict[str, Any]],
    trained: list[dict[str, Any]],
    reason: str,
) -> dict[str, Any]:
    return {
        "matches": matches,
        "reason": reason,
        "current": current,
        "trained": trained,
    }
