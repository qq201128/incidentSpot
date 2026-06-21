from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

from app.db.session import get_conn
from app.services.event_ai_history import settled_expected_profit_usdt
from app.services.event_regime_status import market_regime_status

CANDIDATE_REGIME_ADMISSION_VERSION = "candidate_regime_admission_v1"
EXPLORATION_SAMPLE_MIN = 50
EVALUABLE_SAMPLE_MIN = 120
STABLE_CANDIDATE_SAMPLE_MIN = 200
STABLE_SAMPLE_MIN = 300


@dataclass(frozen=True)
class CandidateRegimeAdmission:
    allowed: bool
    reason: str
    mode: str
    regime: dict[str, Any]
    sample_count: int
    metrics: dict[str, Any]
    version: str = CANDIDATE_REGIME_ADMISSION_VERSION


@dataclass(frozen=True)
class _Thresholds:
    win_rate_min: float
    profit_factor_min: float
    avg_return_positive: bool
    max_consecutive_losses: int
    recent_windows: tuple[tuple[int, float], ...]
    rolling_window: int | None = None
    rolling_win_rate_min: float | None = None


EVALUABLE_THRESHOLDS = _Thresholds(
    win_rate_min=0.57,
    profit_factor_min=1.0,
    avg_return_positive=True,
    max_consecutive_losses=6,
    recent_windows=((50, 0.54),),
)
STABLE_CANDIDATE_THRESHOLDS = _Thresholds(
    win_rate_min=0.59,
    profit_factor_min=1.05,
    avg_return_positive=True,
    max_consecutive_losses=5,
    recent_windows=((50, 0.56), (100, 0.56)),
    rolling_window=20,
    rolling_win_rate_min=0.50,
)
STABLE_THRESHOLDS = _Thresholds(
    win_rate_min=0.61,
    profit_factor_min=1.10,
    avg_return_positive=True,
    max_consecutive_losses=4,
    recent_windows=((50, 0.58), (100, 0.58), (150, 0.58)),
    rolling_window=30,
    rolling_win_rate_min=0.52,
)


def evaluate_candidate_regime_admission(prediction: dict[str, Any]) -> CandidateRegimeAdmission:
    symbol = str(prediction["symbol"]).strip().upper()
    duration = str(prediction["duration"])
    direction = _normalized_direction(prediction["direction"])
    regime = market_regime_status(symbol, duration, int(prediction["open_time"]))
    if regime.get("ready") is not True:
        return _decision(False, "market_regime_not_ready", "blocked", regime, 0, {})
    rows = _bucket_rows(
        symbol=symbol,
        duration=duration,
        regime_label=str(regime.get("regimeLabel") or "unknown"),
        direction=direction,
        identity=_prediction_identity(prediction),
    )
    metrics = _bucket_metrics(rows)
    sample_count = int(metrics["sampleCount"])
    if sample_count < EXPLORATION_SAMPLE_MIN:
        return _decision(True, "regime_exploration_sample_count_below_50", "exploration", regime, sample_count, metrics)
    if sample_count < EVALUABLE_SAMPLE_MIN:
        return _decision(True, "regime_collecting_sample_count_below_120", "collecting", regime, sample_count, metrics)
    mode, thresholds = _evaluation_policy(sample_count)
    failed_reason = _threshold_failure(metrics, thresholds)
    if failed_reason is not None:
        return _decision(False, failed_reason, mode, regime, sample_count, metrics)
    return _decision(True, f"regime_bucket_{mode}_passed", mode, regime, sample_count, metrics)


def _prediction_identity(prediction: dict[str, Any]) -> tuple[str, str]:
    signal_key = str(prediction.get("signal_key") or prediction.get("strategy_key") or "")
    lifecycle_identity = str(
        prediction.get("high_winrate_rule")
        or prediction.get("model_version")
        or signal_key
    )
    return signal_key, lifecycle_identity


def _bucket_rows(
    *,
    symbol: str,
    duration: str,
    regime_label: str,
    direction: str,
    identity: tuple[str, str],
) -> list[dict[str, Any]]:
    signal_key, lifecycle_identity = identity
    conn = get_conn()
    try:
        if not _required_tables_available(conn):
            return []
        rows = conn.execute(
            """
            SELECT
              e.id AS event_id,
              e.start_time,
              e.result,
              e.ai_prediction_correct,
              o.side AS order_side,
              o.qty AS order_qty,
              o.price AS order_price
            FROM events e
            INNER JOIN predictions p ON p.id = e.prediction_id
            LEFT JOIN orders o ON o.id = (
              SELECT id FROM orders WHERE event_id = e.id ORDER BY id DESC LIMIT 1
            )
            WHERE e.status = 'SETTLED'
              AND e.symbol = ?
              AND e.event_interval = ?
              AND e.market_regime_gate_passed = 1
              AND e.market_regime_label = ?
              AND COALESCE(e.ai_predicted_direction, p.direction) = ?
              AND p.signal_key = ?
              AND COALESCE(p.high_winrate_rule, p.model_version, p.signal_key) = ?
            ORDER BY e.start_time ASC, e.id ASC
            """,
            (symbol, duration, regime_label, direction, signal_key, lifecycle_identity),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def _required_tables_available(conn: Any) -> bool:
    table_rows = conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
          AND name IN ('events', 'predictions', 'orders')
        """
    ).fetchall()
    tables = {str(row["name"]) for row in table_rows}
    if {"events", "predictions", "orders"} - tables:
        return False
    event_columns = {str(row["name"]) for row in conn.execute("PRAGMA table_info(events)").fetchall()}
    prediction_columns = {str(row["name"]) for row in conn.execute("PRAGMA table_info(predictions)").fetchall()}
    return {"prediction_id", "market_regime_gate_passed", "market_regime_label"} <= event_columns and "signal_key" in prediction_columns


def _bucket_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    outcomes = [_outcome(row) for row in rows]
    returns = [item["return"] for item in outcomes if item["return"] is not None]
    wins = [item["win"] for item in outcomes if item["win"] is not None]
    return {
        "sampleCount": len(rows),
        "winRate": _ratio(sum(1 for win in wins if win), len(wins)),
        "profitFactor": _profit_factor(returns),
        "avgReturn": _average(returns),
        "maxConsecutiveLosses": _max_consecutive_losses(wins),
        "recent50WinRate": _recent_win_rate(wins, 50),
        "recent100WinRate": _recent_win_rate(wins, 100),
        "recent150WinRate": _recent_win_rate(wins, 150),
        "rolling20WorstWinRate": _rolling_worst_win_rate(wins, 20),
        "rolling30WorstWinRate": _rolling_worst_win_rate(wins, 30),
    }


def _outcome(row: dict[str, Any]) -> dict[str, Any]:
    actual_return = _actual_return(row)
    correct = row.get("ai_prediction_correct")
    win = bool(correct) if correct is not None else (actual_return is not None and actual_return > 0)
    return {"win": win, "return": actual_return}


def _actual_return(row: dict[str, Any]) -> float | None:
    pnl = settled_expected_profit_usdt(
        status="SETTLED",
        order_side=row.get("order_side"),
        order_qty=row.get("order_qty"),
        order_price=row.get("order_price"),
        result=row.get("result"),
    )
    if pnl is None:
        return None
    qty = _finite_float(row.get("order_qty")) or 1.0
    return float(pnl) / qty


def _evaluation_policy(sample_count: int) -> tuple[str, _Thresholds]:
    if sample_count < STABLE_CANDIDATE_SAMPLE_MIN:
        return "evaluable", EVALUABLE_THRESHOLDS
    if sample_count < STABLE_SAMPLE_MIN:
        return "stable_candidate", STABLE_CANDIDATE_THRESHOLDS
    return "stable", STABLE_THRESHOLDS


def _threshold_failure(metrics: dict[str, Any], thresholds: _Thresholds) -> str | None:
    if _lt(metrics.get("winRate"), thresholds.win_rate_min):
        return "regime_bucket_win_rate_below_min"
    if _lt(metrics.get("profitFactor"), thresholds.profit_factor_min):
        return "regime_bucket_profit_factor_below_min"
    if thresholds.avg_return_positive and _not_positive(metrics.get("avgReturn")):
        return "regime_bucket_avg_return_not_positive"
    if int(metrics.get("maxConsecutiveLosses") or 0) > thresholds.max_consecutive_losses:
        return "regime_bucket_consecutive_losses_above_limit"
    for window, minimum in thresholds.recent_windows:
        if _lt(metrics.get(f"recent{window}WinRate"), minimum):
            return f"regime_bucket_recent{window}_win_rate_below_min"
    if thresholds.rolling_window is not None:
        key = f"rolling{thresholds.rolling_window}WorstWinRate"
        if _lt(metrics.get(key), thresholds.rolling_win_rate_min):
            return f"regime_bucket_rolling{thresholds.rolling_window}_win_rate_below_min"
    return None


def _decision(
    allowed: bool,
    reason: str,
    mode: str,
    regime: dict[str, Any],
    sample_count: int,
    metrics: dict[str, Any],
) -> CandidateRegimeAdmission:
    return CandidateRegimeAdmission(
        allowed=allowed,
        reason=reason,
        mode=mode,
        regime=regime,
        sample_count=sample_count,
        metrics=metrics,
    )


def _normalized_direction(direction: Any) -> str:
    value = str(direction).strip().lower()
    if value in {"up", "buy", "long"}:
        return "up"
    if value in {"down", "sell", "short"}:
        return "down"
    raise ValueError(f"unsupported trade direction for candidate regime admission: {direction}")


def _profit_factor(values: list[float]) -> float | None:
    if not values:
        return None
    gains = sum(value for value in values if value > 0)
    losses = -sum(value for value in values if value < 0)
    if losses == 0:
        return float("inf") if gains > 0 else None
    return gains / losses


def _max_consecutive_losses(wins: list[bool]) -> int:
    current = 0
    maximum = 0
    for win in wins:
        if win:
            current = 0
            continue
        current += 1
        maximum = max(maximum, current)
    return maximum


def _recent_win_rate(wins: list[bool], window: int) -> float | None:
    if len(wins) < window:
        return None
    return _ratio(sum(1 for win in wins[-window:] if win), window)


def _rolling_worst_win_rate(wins: list[bool], window: int) -> float | None:
    if len(wins) < window:
        return None
    values = [_ratio(sum(1 for win in wins[index:index + window] if win), window) for index in range(len(wins) - window + 1)]
    return min(value for value in values if value is not None)


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator > 0 else None


def _average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _lt(value: Any, minimum: float | None) -> bool:
    number = _finite_float(value)
    if number is None or minimum is None:
        return True
    return number < minimum


def _not_positive(value: Any) -> bool:
    number = _finite_float(value)
    return number is None or number <= 0


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None
