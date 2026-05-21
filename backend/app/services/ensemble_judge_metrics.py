from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import Any

from app.services.ensemble_judge_constants import (
    LOSS_STREAK_THRESHOLD,
    LOW_SAMPLE_LIMIT,
    MAJOR_SIGNAL_TYPES,
    RECENT_WINDOW_COUNT,
    SIGNAL_FACTOR_COMBO,
    SIGNAL_HIGH_WINRATE_COMBO,
    SIGNAL_MODEL_FAMILY,
    SIGNAL_OTHER,
    WEIGHT_READY_SAMPLE_COUNT,
)
from app.services.factor_combo_simulation_keys import (
    BATCH_COMBO_KEY_PREFIX,
    BATCH_HIGH_WINRATE_KEY_PREFIX,
)
from app.services.model_family_config import is_model_family_shadow_strategy
from app.services.strategy_registry import (
    FACTOR_COMBO_STRATEGY_KEY,
    HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY,
)

EPSILON = 1e-9
PROFIT_FACTOR_SCORE_MAX = 2.0
AVG_RETURN_SCORE_SPAN = 0.02
AVG_RETURN_SCORE_FLOOR = -0.01
MAX_PROFIT_FACTOR = 99.0
MS_PER_DAY = 86_400_000


@dataclass(frozen=True)
class SignalMetrics:
    signal_key: str
    signal_type: str
    rows: tuple[dict[str, Any], ...]


def score_candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_score_signal(metric) for metric in _group_metrics(rows)]


def ranking_payload(rows: list[Any]) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda row: _ranking_sort_key(row), reverse=True)
    return [_score_payload(row) for row in ranked]


def coverage_from_scores(conn: Any, symbol: str, duration: str, scores: list[Any]) -> dict[str, Any]:
    by_type = {kind: _empty_coverage() for kind in MAJOR_SIGNAL_TYPES}
    for row in scores:
        item = by_type.setdefault(row["signal_type"], _empty_coverage())
        item["sampleCount"] += int(row["sample_count"])
        item["maxConsecutiveLosses"] = max(item["maxConsecutiveLosses"], int(row["consecutive_losses"]))
        item["recentProfitFactorBelowOne"] = item["recentProfitFactorBelowOne"] or float(row["profit_factor"]) < 1
        item["distinctTradingDays"] = max(item["distinctTradingDays"], _distinct_days(conn, symbol, duration, row))
    return {"bySignalType": by_type}


def recent_windows(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    size = max(floor(len(rows) / RECENT_WINDOW_COUNT), 1)
    return [
        rows[-size * (index + 1): len(rows) - size * index or None]
        for index in range(RECENT_WINDOW_COUNT)
    ]


def window_return(rows: list[dict[str, Any]]) -> float:
    return sum(float(row["actual_return"] or 0) for row in rows)


def _group_metrics(rows: list[dict[str, Any]]) -> tuple[SignalMetrics, ...]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["strategy_key"]), []).append(row)
    return tuple(
        SignalMetrics(key, _signal_type(key), tuple(items))
        for key, items in sorted(grouped.items())
    )


def _score_signal(metric: SignalMetrics) -> dict[str, Any]:
    rows = metric.rows
    win_rate = _win_rate(rows)
    avg_return = sum(float(row["actual_return"] or 0) for row in rows) / max(len(rows), 1)
    profit_factor = _profit_factor(rows)
    losses = _consecutive_losses(rows)
    stability = _stability_score(rows)
    weight = _weight_suggestion(len(rows), win_rate, profit_factor, losses)
    score = _ranking_score(len(rows), win_rate, avg_return, profit_factor, stability, losses)
    return {
        "signal_key": metric.signal_key,
        "signal_type": metric.signal_type,
        "sample_count": len(rows),
        "win_rate": win_rate,
        "avg_return": avg_return,
        "profit_factor": profit_factor,
        "consecutive_losses": losses,
        "stability_score": stability,
        "weight_suggestion": weight,
        "score": score,
    }


def _score_payload(row: Any) -> dict[str, Any]:
    sample_count = int(row["sample_count"])
    weak = sample_count >= WEIGHT_READY_SAMPLE_COUNT and (
        float(row["win_rate"]) < 0.48 or float(row["profit_factor"]) < 1
    )
    return {
        "signalKey": row["signal_key"],
        "signalType": row["signal_type"],
        "sampleCount": sample_count,
        "winRate": row["win_rate"],
        "avgReturn": row["avg_return"],
        "profitFactor": row["profit_factor"],
        "consecutiveLosses": row["consecutive_losses"],
        "stabilityScore": row["stability_score"],
        "weightSuggestion": row["weight_suggestion"],
        "score": row["score"],
        "lowSample": sample_count < LOW_SAMPLE_LIMIT,
        "insufficientSample": sample_count < WEIGHT_READY_SAMPLE_COUNT,
        "weakSignal": weak,
        "degraded": float(row["weight_suggestion"]) < 1,
    }


def _ranking_sort_key(row: Any) -> tuple[int, float]:
    sample_count = int(row["sample_count"])
    eligible = 0 if sample_count < LOW_SAMPLE_LIMIT else 1
    return (eligible, float(row["score"]))


def _empty_coverage() -> dict[str, Any]:
    return {
        "sampleCount": 0,
        "distinctTradingDays": 0,
        "maxConsecutiveLosses": 0,
        "recentProfitFactorBelowOne": False,
    }


def _distinct_days(conn: Any, symbol: str, duration: str, row: Any) -> int:
    rows = conn.execute(
        """
        SELECT DISTINCT CAST(open_time / ? AS INTEGER) AS day
        FROM predictions
        WHERE symbol = ? AND duration = ? AND strategy_key = ? AND settled_at IS NOT NULL
        """,
        (MS_PER_DAY, symbol, duration, row["signal_key"]),
    ).fetchall()
    return len(rows)


def _win_rate(rows: tuple[dict[str, Any], ...]) -> float:
    wins = sum(1 for row in rows if bool(row["prediction_correct"]))
    return wins / max(len(rows), 1)


def _profit_factor(rows: tuple[dict[str, Any], ...]) -> float:
    gains = sum(max(float(row["actual_return"] or 0), 0) for row in rows)
    losses = abs(sum(min(float(row["actual_return"] or 0), 0) for row in rows))
    if losses <= EPSILON:
        return MAX_PROFIT_FACTOR if gains > 0 else 0.0
    return gains / losses


def _consecutive_losses(rows: tuple[dict[str, Any], ...]) -> int:
    losses = 0
    for row in reversed(rows):
        if bool(row["prediction_correct"]):
            return losses
        losses += 1
    return losses


def _stability_score(rows: tuple[dict[str, Any], ...]) -> float:
    windows = recent_windows([dict(row) for row in rows])
    positive = sum(1 for window in windows if window_return(window) > 0)
    return positive / RECENT_WINDOW_COUNT


def _weight_suggestion(count: int, win_rate: float, profit_factor: float, losses: int) -> float:
    if losses >= LOSS_STREAK_THRESHOLD:
        return 0.3
    if count < WEIGHT_READY_SAMPLE_COUNT:
        return 0.6
    if win_rate < 0.48 or profit_factor < 1:
        return 0.5
    return 1.0


def _ranking_score(count: int, win_rate: float, avg_return: float, pf: float, stability: float, losses: int) -> float:
    win_rate_score = _clamp((win_rate - 0.4) / 0.2)
    profit_factor_score = _clamp(pf / PROFIT_FACTOR_SCORE_MAX)
    avg_return_score = _clamp((avg_return - AVG_RETURN_SCORE_FLOOR) / AVG_RETURN_SCORE_SPAN)
    sample_score = _clamp(count / WEIGHT_READY_SAMPLE_COUNT)
    drawdown_penalty = _clamp(losses / LOSS_STREAK_THRESHOLD)
    return (
        win_rate_score * 35 + profit_factor_score * 20 + avg_return_score * 15
        + stability * 15 + sample_score * 10 - drawdown_penalty * 5
    )


def _signal_type(strategy_key: str) -> str:
    if strategy_key == HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY:
        return SIGNAL_HIGH_WINRATE_COMBO
    if strategy_key.startswith(BATCH_HIGH_WINRATE_KEY_PREFIX):
        return SIGNAL_HIGH_WINRATE_COMBO
    if strategy_key == FACTOR_COMBO_STRATEGY_KEY:
        return SIGNAL_FACTOR_COMBO
    if strategy_key.startswith(BATCH_COMBO_KEY_PREFIX):
        return SIGNAL_FACTOR_COMBO
    if is_model_family_shadow_strategy(strategy_key):
        return SIGNAL_MODEL_FAMILY
    return SIGNAL_OTHER


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
