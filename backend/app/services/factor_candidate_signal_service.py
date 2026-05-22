from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
from typing import Any

import pandas as pd

from app.services.factor_backtest_materialization import materialized_frame_for_factor
from app.services.factor_backtest_service import BACKTEST_MIN_PERIODS
from app.services.factor_cache_metadata import assert_cache_usable_for_live_signal
from app.services.factor_candidate_signal_keys import factor_candidate_signal_key
from app.services.factor_catalog import factor_definition_for_backtest
from app.services.factor_combo_scoring import oriented_zscore
from app.services.factor_duration_alignment import duration_entry_source_open_time, live_duration_entry_index
from app.services.factor_frame_service import load_factor_frame
from app.services.kline_prediction_refresh import refresh_prediction_klines
from app.services.factor_ranking_cache_service import get_cached_ranking
from app.services.factor_registry import FactorDirection
from app.services.rule_config import MS_PER_MINUTE, SUPPORTED_RULE_DURATIONS

SIGNAL_RULE_NAME = "factor_candidate_signal_v1"
PROBABILITY_DECIMALS = 4
SCORE_DECIMALS = 6
NEUTRAL_WIN_RATE = 0.5
logger = logging.getLogger("uvicorn.error")


@dataclass(frozen=True)
class _FactorSignal:
    score: float
    median: float
    direction: str
    confidence: float
    index: Any
    orientation: int


def factor_candidate_signal_keys(symbol: str, duration: str) -> tuple[str, ...]:
    rows = _ranking_rows(symbol, duration, require_usable=False)
    return tuple(factor_candidate_signal_key(str(row["factorName"])) for row in rows)


def factor_candidate_signal_strategy_keys(symbol: str, duration: str) -> tuple[str, ...]:
    return factor_candidate_signal_keys(symbol, duration)


def predict_factor_candidate_signals(
    symbol: str,
    duration: str,
    *,
    entry_open_time: int,
    entry_grace_ms: int | None = None,
) -> list[dict[str, Any]]:
    del entry_grace_ms
    _refresh_candidate_source_klines(symbol, duration, entry_open_time)
    rows = _ranking_rows(symbol, duration, require_usable=True)
    working = load_factor_frame(symbol, duration)
    predictions = []
    failures = []
    for row in rows:
        try:
            working, prediction = _prediction_for_row(working, row, symbol, duration, entry_open_time)
            predictions.append(prediction)
        except Exception as exc:
            failures.append(_candidate_failure(row, exc))
    if failures:
        _log_candidate_failures(symbol, duration, failures)
    if not predictions and failures:
        raise ValueError(_candidate_failure_message(symbol, duration, failures))
    return predictions


def _refresh_candidate_source_klines(symbol: str, duration: str, entry_open_time: int) -> None:
    source_open_time = duration_entry_source_open_time(entry_open_time, duration)
    refresh_prediction_klines(symbol, "1m", int(entry_open_time) - MS_PER_MINUTE)
    refresh_prediction_klines(symbol, duration, source_open_time)


def _ranking_rows(symbol: str, duration: str, *, require_usable: bool) -> list[dict[str, Any]]:
    if duration not in SUPPORTED_RULE_DURATIONS:
        raise ValueError(f"unsupported duration: {duration}")
    cache = get_cached_ranking(symbol.strip().upper(), duration)
    if cache is None:
        if require_usable:
            raise ValueError(f"factor ranking cache missing for {symbol.upper()} {duration}")
        return []
    if require_usable:
        assert_cache_usable_for_live_signal(cache, f"factor ranking {symbol.upper()} {duration}")
    return [dict(row) for row in cache.get("ranking") or [] if _usable_factor_row(row)]


def _prediction_for_row(
    frame: pd.DataFrame,
    row: dict[str, Any],
    symbol: str,
    duration: str,
    entry_open_time: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    factor = factor_definition_for_backtest(str(row["factorName"]), symbol, duration)
    working = materialized_frame_for_factor(frame, factor, symbol.strip().upper(), duration)
    if factor.name not in working.columns:
        raise ValueError(f"factor candidate signal missing column: {factor.name}")
    signal = _live_factor_signal(working, row, factor.name, duration, entry_open_time)
    return working, _prediction_payload(row, signal, symbol, duration, entry_open_time)


def _live_factor_signal(
    frame: pd.DataFrame,
    row: dict[str, Any],
    factor_name: str,
    duration: str,
    entry_open_time: int,
) -> _FactorSignal:
    orientation = _factor_orientation(row)
    index = live_duration_entry_index(frame, duration, entry_open_time)
    score_series = oriented_zscore(frame[factor_name], orientation)
    score = _finite_float(_series_value_at(score_series, index))
    median = _finite_float(_series_value_at(score_series.expanding(BACKTEST_MIN_PERIODS).median().shift(1), index))
    if score is None or median is None:
        raise ValueError(f"factor candidate signal has insufficient score history: {factor_name}")
    direction = "up" if score >= median else "down"
    return _FactorSignal(score, median, direction, _directional_win_rate(row, orientation), index, orientation)


def _prediction_payload(
    row: dict[str, Any],
    signal: _FactorSignal,
    symbol: str,
    duration: str,
    entry_open_time: int,
) -> dict[str, Any]:
    signal_key = factor_candidate_signal_key(str(row["factorName"]))
    probability_up = signal.confidence if signal.direction == "up" else 1.0 - signal.confidence
    return {
        "symbol": symbol.strip().upper(),
        "signal_key": signal_key,
        "strategy_key": signal_key,
        "duration": duration,
        "open_time": int(entry_open_time),
        "entry_price": None,
        "direction": signal.direction,
        "probability_up": round(probability_up, PROBABILITY_DECIMALS),
        "confidence": round(signal.confidence, PROBABILITY_DECIMALS),
        "certainty_label": "FACTOR_CANDIDATE_OBSERVE",
        "trade_quality_score": round(signal.confidence, PROBABILITY_DECIMALS),
        "trade_quality_passed": True,
        "trade_quality_gate": SIGNAL_RULE_NAME,
        "high_winrate_gate": SIGNAL_RULE_NAME,
        "high_winrate_rule": str(row["factorName"]),
        "high_winrate_gate_passed": True,
        "high_winrate_gate_value": row.get("winRate"),
        "high_winrate_gate_min": None,
        "expected_return": None,
        "model_version": str(row["factorName"]),
        "model_duration": duration,
        "model_trained_at": _utc_now(),
        "rule_score": round(signal.score, SCORE_DECIMALS),
        "rule_reasons": _rule_reasons(row, signal),
        "signal_source": "factor_candidate_signal",
    }


def _rule_reasons(row: dict[str, Any], signal: _FactorSignal) -> list[str]:
    return [
        f"rule={SIGNAL_RULE_NAME}",
        f"factor={row['factorName']}",
        f"category={row.get('category')}",
        f"source_file={row.get('sourceFile')}",
        f"orientation={signal.orientation}",
        f"score={round(signal.score, SCORE_DECIMALS)}",
        f"historical_median={round(signal.median, SCORE_DECIMALS)}",
        f"factor_score={row.get('factorScore')}",
        f"historical_win_rate={row.get('winRate')}",
        f"historical_profit_factor={row.get('profitFactor')}",
    ]


def _usable_factor_row(row: Any) -> bool:
    if not isinstance(row, dict) or not row.get("factorName"):
        return False
    if row.get("backtestValid") is False:
        return False
    return _finite_float(row.get("winRate")) is not None


def _candidate_failure(row: dict[str, Any], exc: Exception) -> dict[str, str]:
    return {
        "factorName": str(row.get("factorName") or "unknown"),
        "errorType": type(exc).__name__,
        "error": str(exc),
    }


def _log_candidate_failures(symbol: str, duration: str, failures: list[dict[str, str]]) -> None:
    logger.warning(
        "factor candidate signal failures symbol=%s duration=%s failures=%s",
        symbol.strip().upper(),
        duration,
        failures,
    )


def _candidate_failure_message(symbol: str, duration: str, failures: list[dict[str, str]]) -> str:
    names = ", ".join(item["factorName"] for item in failures[:5])
    return f"all factor candidate signals failed for {symbol.strip().upper()} {duration}: {names}"


def _factor_orientation(row: dict[str, Any]) -> int:
    direction = str(row.get("direction") or FactorDirection.NEUTRAL)
    if direction == FactorDirection.LOWER_BETTER.value:
        return -1
    if direction == FactorDirection.HIGHER_BETTER.value:
        return 1
    win_rate = _finite_float(row.get("winRate"))
    if win_rate is not None and win_rate < NEUTRAL_WIN_RATE:
        return -1
    return 1 if _metric_sign(row) >= 0 else -1


def _directional_win_rate(row: dict[str, Any], orientation: int) -> float:
    win_rate = _finite_float(row.get("winRate"))
    if win_rate is None:
        raise ValueError(f"factor candidate row missing winRate: {row.get('factorName')}")
    value = win_rate if orientation == 1 else 1.0 - win_rate
    return max(0.0, min(value, 0.99))


def _metric_sign(row: dict[str, Any]) -> float:
    for key in ("icMean", "longShortReturn", "ir"):
        value = _finite_float(row.get(key))
        if value is not None and value != 0:
            return value
    return 1.0


def _series_value_at(series: pd.Series, index: Any) -> Any:
    value = series.loc[index]
    return value.iloc[-1] if isinstance(value, pd.Series) else value


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if isfinite(number) else None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
