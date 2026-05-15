from __future__ import annotations

import json
from importlib import import_module
from typing import Any

from app.services.factor_combo_simulation_keys import FACTOR_COMBO_TOP_SIMULATION_RANKS
from app.services.high_winrate_combo_cache_service import get_cached_high_winrate_combo_ranking

DEFAULT_ACTIVE_RANK = 1
RANKING_REFRESH_REASON = "candidate_pool_exhausted_refreshing"
ROTATED_REASON = "rotated_after_candidate_failed"


def high_winrate_active_rank_from_status(row: dict[str, Any]) -> int:
    details = _status_details(row)
    return _valid_rank(details.get("activeRank") or DEFAULT_ACTIVE_RANK)


def high_winrate_failed_ranks_from_status(row: dict[str, Any]) -> tuple[int, ...]:
    details = _status_details(row)
    values = details.get("failedRanks")
    if not isinstance(values, list):
        return ()
    return tuple(_valid_rank(value) for value in values)


def high_winrate_candidate_rule(symbol: str, duration: str, rank: int) -> str | None:
    cached = get_cached_high_winrate_combo_ranking(symbol, duration)
    if cached is None:
        return None
    ranking = cached.get("ranking")
    if not isinstance(ranking, list) or rank <= 0 or rank > len(ranking):
        return None
    value = str(dict(ranking[rank - 1]).get("factorName") or "")
    return value or None


def next_high_winrate_candidate_rank(rank: int) -> int | None:
    for candidate in FACTOR_COMBO_TOP_SIMULATION_RANKS:
        if candidate > rank:
            return candidate
    return None


def high_winrate_rotation_payload(
    symbol: str,
    duration: str,
    rank: int,
    failed_ranks: tuple[int, ...] = (),
    previous_candidate: dict[str, Any] | None = None,
    *,
    active_rule: str | None = None,
) -> dict[str, Any]:
    payload = {
        "activeRank": rank,
        "activeRule": active_rule if active_rule is not None else high_winrate_candidate_rule(symbol, duration, rank),
        "candidateRanks": list(FACTOR_COMBO_TOP_SIMULATION_RANKS),
        "failedRanks": list(failed_ranks),
    }
    if previous_candidate is not None:
        payload["previousCandidate"] = previous_candidate
    return payload


def failed_rank_payload(
    rank: int,
    rule: str | None,
    decision: dict[str, str],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {"rank": rank, "rule": rule, "status": decision["status"], "reason": decision["reason"], "metrics": metrics}


def refresh_high_winrate_goal(symbol: str, duration: str) -> dict[str, Any]:
    goal = import_module("scripts.high_winrate_factor_combo_goal")
    return goal.run_goal(symbol, duration, goal.TARGET_COUNT, goal.REPORT_PATH, goal.LIBRARY_PATH)


def ensure_high_winrate_status_table(conn: Any) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS high_winrate_strategy_status (
          strategy_key TEXT NOT NULL,
          symbol TEXT NOT NULL,
          duration TEXT NOT NULL,
          status TEXT NOT NULL,
          reason TEXT NOT NULL,
          details_json TEXT NOT NULL,
          sample_count INTEGER NOT NULL,
          win_rate REAL,
          profit_factor REAL,
          consecutive_losses INTEGER NOT NULL,
          evaluated_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          PRIMARY KEY (strategy_key, symbol, duration)
        )
        """
    )


def _status_details(row: dict[str, Any]) -> dict[str, Any]:
    if not row or not row.get("details_json"):
        return {}
    return json.loads(row["details_json"])


def _valid_rank(value: Any) -> int:
    rank = int(value)
    if rank not in FACTOR_COMBO_TOP_SIMULATION_RANKS:
        raise ValueError(f"unsupported high-winrate candidate rank: {rank}")
    return rank
