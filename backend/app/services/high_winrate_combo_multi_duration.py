from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.services.factor_learning_common import utc_now

RunGoal = Callable[[str, str, int, Path, Path], dict[str, Any]]


def parse_durations(raw: str | None) -> list[str]:
    if raw is None:
        return []
    durations = [value.strip() for value in raw.split(",") if value.strip()]
    if not durations:
        raise ValueError("--durations must include at least one duration")
    return durations


def run_multi_duration_goal(
    symbol: str,
    durations: list[str],
    target_count: int,
    output: Path,
    library: Path,
    run_single_goal: RunGoal,
    parallel_workers: int = 1,
) -> dict[str, Any]:
    reports = _duration_reports(symbol, durations, target_count, output, library, run_single_goal, parallel_workers)
    payload = _multi_duration_payload(symbol, durations, reports)
    _write_json(output, payload)
    _write_json(library, _multi_duration_library_payload(payload))
    return payload


def _duration_reports(
    symbol: str,
    durations: list[str],
    target_count: int,
    output: Path,
    library: Path,
    run_single_goal: RunGoal,
    parallel_workers: int,
) -> list[dict[str, Any]]:
    if parallel_workers <= 1 or len(durations) <= 1:
        return [
            _run_duration_goal(symbol, duration, target_count, output, library, run_single_goal)
            for duration in durations
        ]
    workers = min(int(parallel_workers), len(durations))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(
            executor.map(
                lambda duration: _run_duration_goal(symbol, duration, target_count, output, library, run_single_goal),
                durations,
            )
        )


def _run_duration_goal(
    symbol: str,
    duration: str,
    target_count: int,
    output: Path,
    library: Path,
    run_single_goal: RunGoal,
) -> dict[str, Any]:
    try:
        return run_single_goal(
            symbol,
            duration,
            target_count,
            _duration_path(output, duration),
            _duration_path(library, duration),
        )
    except Exception as exc:
        raise RuntimeError(f"duration {duration} failed") from exc


def _duration_path(path: Path, duration: str) -> Path:
    return path.with_name(f"{path.stem}_{duration}{path.suffix}")


def _multi_duration_payload(
    symbol: str,
    durations: list[str],
    reports: list[dict[str, Any]],
) -> dict[str, Any]:
    rankings = [
        {**row, "symbol": report["symbol"], "duration": report["duration"]}
        for report in reports
        for row in report["ranking"]
    ]
    rankings.sort(key=lambda row: (row["winRate"], row["profitFactor"], row["avgReturn"], row["trades"]), reverse=True)
    return {
        "version": "high_winrate_factor_combo_goal_multi_duration_v1",
        "updatedAt": utc_now(),
        "symbol": symbol.strip().upper(),
        "durations": durations,
        "target": reports[0]["target"],
        "perDuration": reports,
        "bestRanking": rankings[: len(durations) * int(reports[0]["target"]["targetCount"])],
        "promotion": _multi_duration_promotion(reports),
    }


def _multi_duration_promotion(reports: list[dict[str, Any]]) -> dict[str, Any]:
    promotions = [report.get("promotion") or {} for report in reports]
    return {
        "promoted": sum(int(row.get("promoted") or 0) for row in promotions),
        "durations": [_duration_promotion(row) for row in promotions],
    }


def _duration_promotion(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "symbol": row.get("symbol"),
        "duration": row.get("duration"),
        "promoted": row.get("promoted"),
        "libraryTotal": row.get("libraryTotal"),
        "thresholds": row.get("thresholds"),
    }


def _multi_duration_library_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": report["version"],
        "updatedAt": report["updatedAt"],
        "symbol": report["symbol"],
        "durations": report["durations"],
        "target": report["target"],
        "factors": report["bestRanking"],
        "perDuration": [
            {"duration": row["duration"], "factors": row["ranking"], "promotion": row.get("promotion")}
            for row in report["perDuration"]
        ],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
