from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.services.factor_frame_service import load_factor_frame
from app.services.high_winrate_combo_goal_config import GoalSearchConfig, signal_thresholds
from app.services.high_winrate_combo_goal_payloads import ranking_row
from app.services.high_winrate_combo_goal_search import (
    oriented_score_search,
    ranked_hit_search,
    search_frame,
    validated_search_config,
)
from app.services.lstm_factor_combo_feature_store import save_factor_combo_feature_snapshots
from app.services.rule_config import MS_PER_MINUTE, horizon_minutes_for_duration

DEFAULT_LOOKBACK_ROWS = 1800
DEFAULT_STEP_ROWS = 240
DEFAULT_MIN_HISTORY_ROWS = 900
DEFAULT_RANKING_LIMIT = 12
DEFAULT_CANDIDATE_LIMIT = 12
DEFAULT_MIN_TRADES = 80
DEFAULT_THRESHOLD_MIN = 0.8
DEFAULT_THRESHOLD_MAX = 1.6
DEFAULT_THRESHOLD_STEP = 0.4


@dataclass(frozen=True)
class FactorComboSnapshotBackfillConfig:
    lookback_rows: int = DEFAULT_LOOKBACK_ROWS
    step_rows: int = DEFAULT_STEP_ROWS
    min_history_rows: int = DEFAULT_MIN_HISTORY_ROWS
    ranking_limit: int = DEFAULT_RANKING_LIMIT
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT
    min_trades: int = DEFAULT_MIN_TRADES
    threshold_min: float = DEFAULT_THRESHOLD_MIN
    threshold_max: float = DEFAULT_THRESHOLD_MAX
    threshold_step: float = DEFAULT_THRESHOLD_STEP


def backfill_factor_combo_feature_snapshots(
    symbol: str,
    duration: str,
    config: FactorComboSnapshotBackfillConfig | None = None,
) -> dict[str, Any]:
    cfg = _validated_config(config or FactorComboSnapshotBackfillConfig())
    sym = symbol.strip().upper()
    frame = search_frame(load_factor_frame(sym, duration), duration).reset_index(drop=True)
    positions = _snapshot_positions(len(frame), cfg)
    snapshots = [_snapshot_at(frame, position, sym, duration, cfg) for position in positions]
    selected = [snapshot for snapshot in snapshots if snapshot is not None]
    if not selected:
        raise ValueError(f"no factor combo snapshots generated for {sym} {duration}")
    save_factor_combo_feature_snapshots(sym, duration, selected)
    return {
        "version": "factor_combo_feature_snapshot_backfill_v1",
        "symbol": sym,
        "duration": duration,
        "requested": len(positions),
        "saved": len(selected),
        "firstEntryOpenTime": selected[0]["entryOpenTime"],
        "lastEntryOpenTime": selected[-1]["entryOpenTime"],
        "config": _config_payload(cfg),
    }


def _snapshot_at(
    frame: pd.DataFrame,
    position: int,
    symbol: str,
    duration: str,
    config: FactorComboSnapshotBackfillConfig,
) -> dict[str, Any] | None:
    start = max(0, position - config.lookback_rows)
    history = frame.iloc[start:position].copy()
    if len(history) < config.min_history_rows:
        return None
    score_search = oriented_score_search(history)
    ranked = ranked_hit_search(history, score_search.scores, _goal_config(config))
    rows = [
        ranking_row(rank, hit, _goal_config(config))
        for rank, hit in enumerate(ranked.hits[: config.ranking_limit], start=1)
    ]
    if not rows:
        return None
    return {"entryOpenTime": _entry_open_time(frame, position, duration), "ranking": rows}


def _snapshot_positions(total_rows: int, config: FactorComboSnapshotBackfillConfig) -> list[int]:
    if total_rows <= config.min_history_rows:
        return []
    positions = list(range(config.min_history_rows, total_rows, config.step_rows))
    last = total_rows - 1
    if not positions or positions[-1] != last:
        positions.append(last)
    return positions


def _entry_open_time(frame: pd.DataFrame, position: int, duration: str) -> int:
    open_time = int(frame.iloc[position]["open_time"])
    return open_time + horizon_minutes_for_duration(duration) * MS_PER_MINUTE


def _goal_config(config: FactorComboSnapshotBackfillConfig) -> GoalSearchConfig:
    return validated_search_config(
        GoalSearchConfig(
            candidate_limit=config.candidate_limit,
            signal_thresholds=signal_thresholds(
                config.threshold_min,
                config.threshold_max,
                config.threshold_step,
            ),
            min_trades=config.min_trades,
        )
    )


def _validated_config(config: FactorComboSnapshotBackfillConfig) -> FactorComboSnapshotBackfillConfig:
    if min(config.lookback_rows, config.step_rows, config.min_history_rows, config.ranking_limit) <= 0:
        raise ValueError("lookback_rows, step_rows, min_history_rows, and ranking_limit must be positive")
    if config.min_history_rows > config.lookback_rows:
        raise ValueError("min_history_rows must be <= lookback_rows")
    _goal_config(config)
    return config


def _config_payload(config: FactorComboSnapshotBackfillConfig) -> dict[str, Any]:
    return {
        "lookbackRows": config.lookback_rows,
        "stepRows": config.step_rows,
        "minHistoryRows": config.min_history_rows,
        "rankingLimit": config.ranking_limit,
        "candidateLimit": config.candidate_limit,
        "minTrades": config.min_trades,
        "thresholdMin": config.threshold_min,
        "thresholdMax": config.threshold_max,
        "thresholdStep": config.threshold_step,
    }
