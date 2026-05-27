from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.services.binance_service import fetch_klines
from app.services.agent_mined_factor_library import agent_factor_rows_for_duration
from app.services.factor_backtest_gate import meets_backtest_gate
from app.services.factor_backtest_materialization import materialized_frame_for_factor
from app.services.factor_backtest_service import BACKTEST_MIN_PERIODS
from app.services.factor_cache_metadata import assert_cache_usable_for_live_signal
from app.services.factor_candidate_signal_keys import factor_candidate_signal_key
from app.services.paper_live_candidate_service import log_prediction_failure
from app.services.factor_catalog import factor_definition_for_backtest
from app.services.factor_candidate_signal_utils import (
    candidate_failure,
    candidate_failure_message,
    directional_win_rate,
    factor_orientation,
    finite_float,
    series_value_at,
    usable_factor_row,
    utc_now,
)
from app.services.factor_combo_scoring import oriented_zscore
from app.services.factor_duration_alignment import (
    duration_entry_source_open_time,
    live_duration_entry_index,
)
from app.services.factor_frame_service import FACTOR_FRAME_MIN_HISTORY, load_factor_frame
from app.services.kline_backfill import count_klines, oldest_open_time, upsert_klines_rows
from app.services.kline_prediction_refresh import refresh_prediction_klines
from app.services.factor_ranking_cache_service import get_cached_ranking
from app.services.rule_config import MS_PER_MINUTE, SUPPORTED_RULE_DURATIONS, horizon_minutes_for_duration

SIGNAL_RULE_NAME = "factor_candidate_signal_v1"
PROBABILITY_DECIMALS = 4
SCORE_DECIMALS = 6
MIN_DURATION_KLINE_ROWS = 540
DURATION_BACKFILL_LIMIT = 1000
MAX_DURATION_BACKFILL_ROUNDS = 10
CANDIDATE_SCORE_LOOKBACK_BARS = max(
    BACKTEST_MIN_PERIODS * 2,
    FACTOR_FRAME_MIN_HISTORY,
    MIN_DURATION_KLINE_ROWS,
)
logger = logging.getLogger("uvicorn.error")

@dataclass(frozen=True)
class _FactorSignal:
    score: float
    median: float
    direction: str
    confidence: float
    entry_price: float
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
    working = load_factor_frame(symbol, duration, min_history=_candidate_frame_min_history())
    predictions = []
    failures = []
    for row in rows:
        try:
            working, prediction = _prediction_for_row(working, row, symbol, duration, entry_open_time)
            predictions.append(prediction)
        except Exception as exc:
            failures.append(candidate_failure(row, exc))
    if failures:
        _log_candidate_failures(symbol, duration, failures)
    if not predictions and failures:
        raise ValueError(candidate_failure_message(symbol, duration, failures))
    return predictions

def _refresh_candidate_source_klines(symbol: str, duration: str, entry_open_time: int) -> None:
    source_open_time = duration_entry_source_open_time(entry_open_time, duration)
    duration_step_ms = horizon_minutes_for_duration(duration) * MS_PER_MINUTE
    lookback_start = source_open_time - (CANDIDATE_SCORE_LOOKBACK_BARS - 1) * duration_step_ms
    refresh_prediction_klines(symbol, duration, lookback_start)
    refresh_prediction_klines(symbol, duration, source_open_time)
    _backfill_duration_klines_if_needed(symbol, duration)
    one_m_lookback_start = int(entry_open_time) - CANDIDATE_SCORE_LOOKBACK_BARS * MS_PER_MINUTE
    refresh_prediction_klines(symbol, "1m", one_m_lookback_start)
    refresh_prediction_klines(symbol, "1m", int(entry_open_time) - MS_PER_MINUTE)

def _candidate_frame_min_history() -> int:
    return CANDIDATE_SCORE_LOOKBACK_BARS

def _backfill_duration_klines_if_needed(symbol: str, duration: str) -> None:
    sym = symbol.strip().upper()
    end_time = _duration_backfill_end_time(sym, duration)
    rounds = 0
    while count_klines(sym, duration) < MIN_DURATION_KLINE_ROWS:
        if rounds >= MAX_DURATION_BACKFILL_ROUNDS:
            break
        rounds += 1
        rows = fetch_klines(sym, duration, limit=DURATION_BACKFILL_LIMIT, end_time=end_time)
        if not rows:
            raise ValueError(f"no historical {duration} klines returned for {sym} before {end_time}")
        upsert_klines_rows(sym, duration, rows)
        end_time = _next_duration_backfill_end_time(rows, end_time, sym, duration)

def _duration_backfill_end_time(symbol: str, duration: str) -> int | None:
    oldest = oldest_open_time(symbol, duration)
    return int(oldest) - 1 if oldest is not None else None

def _next_duration_backfill_end_time(
    rows: list[dict],
    end_time: int | None,
    symbol: str,
    duration: str,
) -> int:
    new_oldest = min(int(row["openTime"]) for row in rows)
    if end_time is not None and new_oldest >= end_time:
        raise ValueError(f"historical {duration} kline backfill did not move earlier for {symbol}")
    return new_oldest - 1

def _ranking_rows(symbol: str, duration: str, *, require_usable: bool) -> list[dict[str, Any]]:
    if duration not in SUPPORTED_RULE_DURATIONS:
        raise ValueError(f"unsupported duration: {duration}")
    sym = symbol.strip().upper()
    cache = get_cached_ranking(sym, duration)
    if require_usable and cache is not None:
        assert_cache_usable_for_live_signal(cache, f"factor ranking {sym} {duration}")
    rows: list[dict[str, Any]] = []
    if cache is not None:
        rows.extend(dict(row) for row in cache.get("ranking") or [] if usable_factor_row(row))
    for row in agent_factor_rows_for_duration(sym, duration):
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        candidate = {
            **row,
            "winRate": metrics.get("winRate"),
            "profitFactor": metrics.get("profitFactor"),
            "totalPeriods": metrics.get("totalPeriods"),
            "backtestValid": True,
        }
        if usable_factor_row(candidate):
            rows.append(candidate)
    if require_usable and not rows and cache is None:
        raise ValueError(f"factor ranking cache missing for {sym} {duration}")
    return [row for row in rows if meets_backtest_gate(row)]

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
    orientation = factor_orientation(row)
    index = _strict_duration_entry_index(frame, duration, entry_open_time)
    score_series = oriented_zscore(frame[factor_name], orientation)
    score = finite_float(series_value_at(score_series, index))
    median = finite_float(
        series_value_at(score_series.expanding(min_periods=BACKTEST_MIN_PERIODS).median().shift(1), index)
    )
    entry_price = finite_float(series_value_at(frame["close"], index))
    if score is None or median is None or entry_price is None:
        raise ValueError(f"factor candidate signal has insufficient score history: {factor_name}")
    direction = "up" if score >= median else "down"
    return _FactorSignal(score, median, direction, directional_win_rate(row, orientation), entry_price, index, orientation)

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
        "entry_price": signal.entry_price,
        "direction": signal.direction,
        "probability_up": round(probability_up, PROBABILITY_DECIMALS),
        "confidence": round(signal.confidence, PROBABILITY_DECIMALS),
        "certainty_label": "FACTOR_CANDIDATE_OBSERVE",
        "trade_quality_score": round(signal.confidence, PROBABILITY_DECIMALS),
        "trade_quality_passed": meets_backtest_gate(row),
        "trade_quality_gate": SIGNAL_RULE_NAME,
        "high_winrate_gate": SIGNAL_RULE_NAME,
        "high_winrate_rule": str(row["factorName"]),
        "high_winrate_gate_passed": True,
        "high_winrate_gate_value": row.get("winRate"),
        "high_winrate_gate_min": None,
        "expected_return": None,
        "model_version": str(row["factorName"]),
        "model_family": "factor",
        "model_duration": duration,
        "model_trained_at": utc_now(),
        "oos_win_rate": _oos_win_rate(row),
        "walk_forward_result": row.get("walkForward"),
        "recent_rolling_result": row.get("recentRollingResult"),
        "data_freshness_status": "fresh",
        "missing_feature_status": "complete",
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


def _oos_win_rate(row: dict[str, Any]) -> Any:
    walk_forward = row.get("walkForward")
    if isinstance(walk_forward, dict):
        return walk_forward.get("oosWinRate")
    return row.get("oosWinRate")

def _strict_duration_entry_index(
    frame: pd.DataFrame,
    duration: str,
    entry_open_time: int,
) -> Any:
    index = live_duration_entry_index(frame, duration, entry_open_time)
    source_open_time = duration_entry_source_open_time(entry_open_time, duration)
    actual_open_time = int(pd.to_numeric(frame.loc[index, "open_time"], errors="raise"))
    if actual_open_time != source_open_time:
        raise ValueError(
            f"factor candidate signal missing completed {duration} source row at "
            f"open_time={source_open_time}; latest available open_time={actual_open_time}"
        )
    return index

def _log_candidate_failures(symbol: str, duration: str, failures: list[dict[str, str]]) -> None:
    logger.warning(
        "factor candidate signal failures symbol=%s duration=%s failures=%s",
        symbol.strip().upper(),
        duration,
        failures,
    )
    for failure in failures:
        factor_name = str(failure.get("factorName") or "unknown_factor")
        log_prediction_failure(
            candidate_key=factor_candidate_signal_key(factor_name),
            strategy_key=factor_candidate_signal_key(factor_name),
            symbol=symbol,
            duration=duration,
            stage=str(failure.get("stage") or "factor_candidate_signal"),
            reason=str(failure.get("error") or "unknown_factor_candidate_failure"),
            details={"factorName": factor_name},
        )
