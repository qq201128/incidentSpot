from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from app.services.model_metrics import TARGET_TRADES_PER_DAY, TARGET_WIN_RATE

MODEL_DURATION = "10m"
REPORT_NAME = "optimize_high_winrate_10m.json"

WindowTrainer = Callable[[int], dict[str, Any]]


def optimize_high_winrate(
    train_window: WindowTrainer,
    output_dir: Path,
    symbol: str,
    windows: list[int],
    *,
    target_win_rate: float = TARGET_WIN_RATE,
    min_trades_per_day: float = TARGET_TRADES_PER_DAY,
    min_move_bps: float,
    min_trade_gap_minutes: int,
) -> dict[str, Any]:
    scanned = _scan_windows(train_window, windows)
    feasible = [
        row for row in scanned
        if _meets_target(row, target_win_rate, min_trades_per_day)
    ]
    report = _build_report(
        symbol=symbol,
        windows=windows,
        scanned=scanned,
        feasible=feasible,
        target_win_rate=target_win_rate,
        min_trades_per_day=min_trades_per_day,
        min_move_bps=min_move_bps,
        min_trade_gap_minutes=min_trade_gap_minutes,
    )
    _write_report(output_dir, report)
    return report


def _scan_windows(train_window: WindowTrainer, windows: list[int]) -> list[dict[str, Any]]:
    scanned: list[dict[str, Any]] = []
    for days in windows:
        meta = train_window(int(days))
        scanned.extend(_profile_rows(meta, int(days), "confidence"))
        scanned.extend(_profile_rows(meta, int(days), "quality"))
    return scanned


def _profile_rows(meta: dict[str, Any], train_window_days: int, profile_key: str) -> list[dict[str, Any]]:
    source_key = f"backtest_{profile_key}_profiles"
    return [
        _profile_row(profile, train_window_days, profile_key)
        for profile in meta.get(source_key, [])
    ]


def _profile_row(profile: dict[str, Any], train_window_days: int, profile_type: str) -> dict[str, Any]:
    return {
        "profile_type": profile_type,
        "train_window_days": int(train_window_days),
        "trade_confidence_threshold": _float(profile, "trade_confidence_threshold"),
        "trade_quality_score_threshold": _float(profile, "trade_quality_score_threshold"),
        "win_rate": _float(profile, "win_rate"),
        "direction_hit_rate": _float(profile, "direction_hit_rate"),
        "trades_per_day": _float(profile, "trades_per_day"),
        "test_trades": int(profile.get("test_trades", 0)),
        "strategy_return": _float(profile, "strategy_return"),
        "avg_trade_return": _float(profile, "avg_trade_return"),
    }


def _build_report(
    *,
    symbol: str,
    windows: list[int],
    scanned: list[dict[str, Any]],
    feasible: list[dict[str, Any]],
    target_win_rate: float,
    min_trades_per_day: float,
    min_move_bps: float,
    min_trade_gap_minutes: int,
) -> dict[str, Any]:
    return {
        "symbol": symbol.upper(),
        "duration": MODEL_DURATION,
        "target_win_rate": float(target_win_rate),
        "min_trades_per_day": float(min_trades_per_day),
        "min_move_bps": float(min_move_bps),
        "min_trade_gap_minutes": int(min_trade_gap_minutes),
        "windows": [int(value) for value in windows],
        "feasible_count": len(feasible),
        "best_feasible": _best_row(feasible),
        "best_scanned": _best_row(scanned),
        "scanned": scanned,
    }


def _meets_target(row: dict[str, Any], target_win_rate: float, min_trades_per_day: float) -> bool:
    return (
        row["win_rate"] >= float(target_win_rate)
        and row["trades_per_day"] >= float(min_trades_per_day)
    )


def _best_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(rows, key=lambda row: (row["win_rate"], row["avg_trade_return"], row["trades_per_day"]))


def _float(profile: dict[str, Any], key: str) -> float | None:
    value = profile.get(key)
    return None if value is None else float(value)


def _write_report(output_dir: Path, report: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / REPORT_NAME, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
