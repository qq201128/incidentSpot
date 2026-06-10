from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.services.binance_service import fetch_klines
from app.services.agent_mined_factor_library import agent_factor_rows_for_duration
from app.services.factor_backtest_materialization import materialized_frame_for_factor
from app.services.factor_backtest_service import BACKTEST_MIN_PERIODS
from app.services.factor_cache_metadata import assert_cache_usable_for_live_signal
from app.services.factor_candidate_signal_keys import factor_candidate_signal_key
from app.services.factor_candidate_signal_payloads import (
    FactorSignal,
    factor_candidate_prediction_payload,
)
from app.services.paper_live_candidate_service import log_prediction_failure
from app.services.factor_catalog import factor_definition_for_backtest
from app.services.factor_candidate_signal_utils import (
    agent_candidate_row,
    candidate_failure,
    candidate_failure_message,
    directional_win_rate,
    factor_orientation,
    finite_float,
    series_value_at,
    usable_factor_row,
)
from app.services.factor_combo_scoring import oriented_zscore
from app.services.factor_duration_alignment import (
    duration_entry_source_open_time,
    live_duration_entry_index,
)
from app.services.factor_regime_analysis import current_factor_regime, factor_regime_gate
from app.services.factor_frame_service import FACTOR_FRAME_MIN_HISTORY, load_factor_frame, lookback_days_for_bars
from app.services.kline_backfill import count_klines, oldest_open_time, upsert_klines_rows
from app.services.kline_prediction_refresh import refresh_prediction_klines
from app.services.positioning_feature_backfill import refresh_positioning_features_for_lookback
from app.services.factor_ranking_cache_service import get_cached_ranking
from app.services.rule_config import MS_PER_MINUTE, SUPPORTED_RULE_DURATIONS, horizon_minutes_for_duration

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
class _PredictionContext:
    symbol: str
    duration: str
    entry_open_time: int

@dataclass(frozen=True)
class _SignalContext:
    row: dict[str, Any]
    factor_name: str
    prediction: _PredictionContext

@dataclass(frozen=True)
class _DurationBackfillContext:
    symbol: str
    duration: str

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
    min_history = _candidate_frame_min_history()
    working = load_factor_frame(
        symbol,
        duration,
        min_history=min_history,
        lookback_days=lookback_days_for_bars(duration, min_history),
    )
    context = _PredictionContext(symbol.strip().upper(), duration, int(entry_open_time))
    predictions = []
    failures = []
    for row in rows:
        try:
            working, prediction = _prediction_for_row(working, row, context)
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
    refresh_positioning_features_for_lookback(symbol, lookback_start)
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
    context = _DurationBackfillContext(sym, duration)
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
        end_time = _next_duration_backfill_end_time(rows, end_time, context)

def _duration_backfill_end_time(symbol: str, duration: str) -> int | None:
    oldest = oldest_open_time(symbol, duration)
    return int(oldest) - 1 if oldest is not None else None

def _next_duration_backfill_end_time(
    rows: list[dict],
    end_time: int | None,
    context: _DurationBackfillContext,
) -> int:
    new_oldest = min(int(row["openTime"]) for row in rows)
    if end_time is not None and new_oldest >= end_time:
        raise ValueError(
            f"historical {context.duration} kline backfill did not move earlier for {context.symbol}"
        )
    return new_oldest - 1

def _ranking_rows(symbol: str, duration: str, *, require_usable: bool) -> list[dict[str, Any]]:
    if duration not in SUPPORTED_RULE_DURATIONS:
        raise ValueError(f"unsupported duration: {duration}")
    sym = symbol.strip().upper()
    cache = get_cached_ranking(sym, duration)
    if require_usable and cache is not None:
        assert_cache_usable_for_live_signal(cache, f"factor ranking {sym} {duration}")
    rows = _cached_ranking_rows(cache) + _agent_ranking_rows(sym, duration)
    if require_usable and not rows and cache is None:
        raise ValueError(f"factor ranking cache missing for {sym} {duration}")
    return rows


def _cached_ranking_rows(cache: dict[str, Any] | None) -> list[dict[str, Any]]:
    if cache is None:
        return []
    return [dict(row) for row in cache.get("ranking") or [] if usable_factor_row(row)]


def _agent_ranking_rows(symbol: str, duration: str) -> list[dict[str, Any]]:
    return [
        candidate
        for candidate in (agent_candidate_row(row) for row in agent_factor_rows_for_duration(symbol, duration))
        if usable_factor_row(candidate)
    ]


def _prediction_for_row(
    frame: pd.DataFrame,
    row: dict[str, Any],
    context: _PredictionContext,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    factor = factor_definition_for_backtest(str(row["factorName"]), context.symbol, context.duration)
    working = materialized_frame_for_factor(frame, factor, context.symbol, context.duration)
    if factor.name not in working.columns:
        raise ValueError(f"factor candidate signal missing column: {factor.name}")
    signal = _live_factor_signal(working, _SignalContext(row, factor.name, context))
    return working, factor_candidate_prediction_payload(
        row,
        signal,
        symbol=context.symbol,
        duration=context.duration,
        entry_open_time=context.entry_open_time,
    )

def _live_factor_signal(
    frame: pd.DataFrame,
    context: _SignalContext,
) -> FactorSignal:
    orientation = factor_orientation(context.row)
    prediction = context.prediction
    index = _strict_duration_entry_index(frame, prediction.duration, prediction.entry_open_time)
    score_series = oriented_zscore(frame[context.factor_name], orientation)
    score = finite_float(series_value_at(score_series, index))
    median = finite_float(
        series_value_at(score_series.expanding(min_periods=BACKTEST_MIN_PERIODS).median().shift(1), index)
    )
    entry_price = finite_float(series_value_at(frame["close"], index))
    if score is None or median is None or entry_price is None:
        raise ValueError(f"factor candidate signal has insufficient score history: {context.factor_name}")
    direction = "up" if score >= median else "down"
    confidence = directional_win_rate(context.row, orientation)
    regime = factor_regime_gate(
        direction,
        confidence,
        current_factor_regime(frame, index, prediction.duration),
    )
    return FactorSignal(
        score,
        median,
        direction,
        confidence,
        entry_price,
        index,
        orientation,
        regime.regime,
        regime.passed,
        regime.reason,
        regime.min_win_rate,
    )

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
