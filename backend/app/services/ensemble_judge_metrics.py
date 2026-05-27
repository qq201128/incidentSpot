from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import Any

from app.services.ensemble_judge_constants import (
    LOSS_STREAK_THRESHOLD,
    LOW_SAMPLE_LIMIT,
    MAJOR_SIGNAL_TYPES,
    RECENT_WINDOW_COUNT,
    SIGNAL_FACTOR_CANDIDATE,
    WEIGHT_READY_SAMPLE_COUNT,
)
from app.services.ensemble_signal_identity import (
    signal_label,
    signal_type,
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
    signal_label: str
    rows: tuple[dict[str, Any], ...]


def score_candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_score_signal(metric) for metric in _group_metrics(rows)]


def unscored_candidate_row(row: dict[str, Any]) -> dict[str, Any]:
    rows = [row]
    return {
        "signal_key": str(row["signal_key"]),
        "signal_type": signal_type(str(row["signal_key"]), rows),
        "signal_label": signal_label(str(row["signal_key"]), rows),
        "sample_count": 0,
        "win_rate": 0.0,
        "avg_return": 0.0,
        "profit_factor": 0.0,
        "consecutive_losses": 0,
        "stability_score": 0.0,
        "weight_suggestion": 0.0,
        "score": 0.0,
        "pending_count": int(row.get("pending_count") or 0),
    }


def ranking_payload(rows: list[Any]) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda row: _ranking_sort_key(row), reverse=True)
    return [_score_payload(row) for row in ranked]


def coverage_from_scores(conn: Any, symbol: str, duration: str, scores: list[Any]) -> dict[str, Any]:
    distinct_days = _distinct_days_by_signal_keys(
        conn,
        symbol,
        duration,
        [str(row["signal_key"]) for row in scores],
    )
    by_type = {kind: _empty_coverage() for kind in MAJOR_SIGNAL_TYPES}
    major_by_type = {kind: _empty_coverage() for kind in MAJOR_SIGNAL_TYPES}
    total = _empty_coverage()
    for row in scores:
        item = _row_coverage(row, distinct_days)
        _merge_coverage(total, item)
        _merge_coverage(by_type.setdefault(row["signal_type"], _empty_coverage()), item)
        _merge_coverage(major_by_type.setdefault(_major_signal_type(row["signal_type"]), _empty_coverage()), item)
    return {
        **total,
        **_ready_source_counts(major_by_type),
        "bySignalType": by_type,
        "byMajorSignalType": major_by_type,
    }


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
        grouped.setdefault(str(row["signal_key"]), []).append(row)
    return tuple(
        SignalMetrics(key, signal_type(key, items), signal_label(key, items), tuple(items))
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
        "signal_label": metric.signal_label,
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
        "signalLabel": row.get("signal_label"),
        "sampleCount": sample_count,
        "winRate": row["win_rate"],
        "avgReturn": row["avg_return"],
        "profitFactor": row["profit_factor"],
        "consecutiveLosses": row["consecutive_losses"],
        "stabilityScore": row["stability_score"],
        "weightSuggestion": row["weight_suggestion"],
        "score": row["score"],
        "pendingCount": int(row.get("pending_count") or 0),
        "pendingSettlement": sample_count == 0 and int(row.get("pending_count") or 0) > 0,
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


def _row_coverage(row: Any, distinct_days_by_key: dict[str, int]) -> dict[str, Any]:
    signal_key = str(row["signal_key"])
    return {
        "sampleCount": int(row["sample_count"]),
        "distinctTradingDays": int(distinct_days_by_key.get(signal_key, 0)),
        "maxConsecutiveLosses": int(row["consecutive_losses"]),
        "recentProfitFactorBelowOne": float(row["profit_factor"]) < 1,
    }


def _merge_coverage(target: dict[str, Any], item: dict[str, Any]) -> None:
    target["sampleCount"] += int(item["sampleCount"])
    target["distinctTradingDays"] = max(int(target["distinctTradingDays"]), int(item["distinctTradingDays"]))
    target["maxConsecutiveLosses"] = max(int(target["maxConsecutiveLosses"]), int(item["maxConsecutiveLosses"]))
    target["recentProfitFactorBelowOne"] = bool(
        target["recentProfitFactorBelowOne"] or item["recentProfitFactorBelowOne"]
    )


def _ready_source_counts(by_type: dict[str, dict[str, Any]]) -> dict[str, int]:
    ready_sources = sum(
        1 for kind in MAJOR_SIGNAL_TYPES
        if int(by_type.get(kind, {}).get("sampleCount", 0)) >= WEIGHT_READY_SAMPLE_COUNT
    )
    return {
        "readySignalTypeCount": ready_sources,
        "requiredSignalTypeCount": len(MAJOR_SIGNAL_TYPES),
    }


def _major_signal_type(signal_type: str) -> str:
    return SIGNAL_FACTOR_CANDIDATE if signal_type == "indicator" else signal_type


def _distinct_days_by_signal_keys(
    conn: Any,
    symbol: str,
    duration: str,
    signal_keys: list[str],
) -> dict[str, int]:
    keys = list(dict.fromkeys(key for key in signal_keys if key))
    if not keys:
        return {}
    counts: dict[str, int] = {}
    chunk_size = 400
    for start in range(0, len(keys), chunk_size):
        chunk = keys[start : start + chunk_size]
        placeholders = ",".join("?" * len(chunk))
        rows = conn.execute(
            f"""
            SELECT signal_key,
                   COUNT(DISTINCT CAST(open_time / ? AS INTEGER)) AS day_count
            FROM predictions
            WHERE symbol = ? AND duration = ? AND settled_at IS NOT NULL
              AND signal_key IN ({placeholders})
            GROUP BY signal_key
            """,
            (MS_PER_DAY, symbol, duration, *chunk),
        ).fetchall()
        for row in rows:
            counts[str(row["signal_key"])] = int(row["day_count"] or 0)
    return counts


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


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
