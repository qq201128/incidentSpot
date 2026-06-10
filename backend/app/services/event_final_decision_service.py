from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.db.session import get_conn
from app.services.event_regime_detector import EventRegimeDataError, detect_event_regime, upsert_event_regime
from app.services.model_family_config import parse_model_family_strategy
from app.services.retired_strategy_keys import is_retired_strategy_key
from app.services.rule_config import DURATION_TO_MINUTES

EVENT_FINAL_DECISION_STRATEGY_KEY = "event_final_decision_v1"
EVENT_FINAL_DECISION_GATE = "event_final_decision_balanced_v1"
DECISION_UP = "UP"
DECISION_DOWN = "DOWN"
DECISION_SKIP = "SKIP"
MIN_CANDIDATES = 2
BASE_CONFIDENCE_MIN = 0.54
HIGH_RISK_CONFIDENCE_MIN = 0.57
MIN_FINAL_SCORE = 0.52


@dataclass(frozen=True)
class DecisionCandidate:
    signal_key: str
    strategy_key: str
    direction: str
    probability_up: float
    confidence: float
    weight: float


@dataclass(frozen=True)
class FinalDecision:
    symbol: str
    duration: str
    open_time: int
    decision: str
    direction: str | None
    probability_up: float | None
    confidence: float
    final_score: float
    regime_label: str
    candidate_count: int
    reason_codes: tuple[str, ...]


def predict_event_final_decision(symbol: str, duration: str, *, entry_open_time: int) -> dict[str, Any] | None:
    sym = symbol.strip().upper()
    _assert_duration(duration)
    try:
        regime = detect_event_regime(sym, duration, entry_open_time)
    except EventRegimeDataError as exc:
        decision = _skip(
            sym,
            duration,
            entry_open_time,
            regime_label="unknown",
            reasons=(str(exc),),
        )
        _persist_decision(decision)
        return None
    candidates = _decision_candidates(sym, duration, entry_open_time)
    decision = _final_decision(sym, duration, entry_open_time, regime.regime_label, regime.confidence, candidates)
    _persist_regime_and_decision(regime, decision)
    if decision.decision == DECISION_SKIP:
        return None
    return _prediction_payload(decision, candidates)


def event_final_decision_exists(symbol: str, duration: str, open_time: int) -> bool:
    conn = get_conn()
    try:
        sql = "SELECT 1 FROM event_final_decisions WHERE symbol = ? AND duration = ? AND open_time = ? LIMIT 1"
        row = conn.execute(sql, (symbol.strip().upper(), duration, int(open_time))).fetchone()
        return row is not None
    finally:
        conn.close()


def settle_due_final_decisions(conn: Any, symbol: str, duration: str, max_open_time: int) -> int:
    rows = conn.execute(
        """
        SELECT open_time, direction
        FROM event_final_decisions
        WHERE symbol = ? AND duration = ? AND settled_at IS NULL AND open_time <= ?
        ORDER BY open_time
        """,
        (symbol.strip().upper(), duration, int(max_open_time)),
    ).fetchall()
    settled = 0
    for row in rows:
        if _settle_one_decision(conn, symbol, duration, dict(row)):
            settled += 1
    return settled


def _decision_candidates(symbol: str, duration: str, open_time: int) -> list[DecisionCandidate]:
    conn = get_conn()
    try:
        rows = conn.execute(
            """
            SELECT signal_key, strategy_key, direction, probability_up, confidence, validation_win_rate
            FROM predictions
            WHERE symbol = ? AND duration = ? AND open_time = ? AND signal_key != ?
            ORDER BY id
            """,
            (symbol, duration, int(open_time), EVENT_FINAL_DECISION_STRATEGY_KEY),
        ).fetchall()
    finally:
        conn.close()
    return [_candidate(dict(row)) for row in rows if _is_model_family_row(dict(row))]


def _is_model_family_row(row: dict[str, Any]) -> bool:
    key = str(row.get("strategy_key") or "")
    return parse_model_family_strategy(key) is not None and not is_retired_strategy_key(key)


def _candidate(row: dict[str, Any]) -> DecisionCandidate:
    confidence = float(row["confidence"] or 0.0)
    validation = row.get("validation_win_rate")
    reliability = float(validation) if validation is not None else 0.5
    return DecisionCandidate(
        signal_key=str(row["signal_key"]),
        strategy_key=str(row["strategy_key"]),
        direction=str(row["direction"]).lower(),
        probability_up=_clamp(float(row["probability_up"])),
        confidence=confidence,
        weight=max(confidence * max(reliability, 0.01), 0.0),
    )


def _final_decision(
    symbol: str,
    duration: str,
    open_time: int,
    regime_label: str,
    regime_confidence: float,
    candidates: list[DecisionCandidate],
) -> FinalDecision:
    if len(candidates) < MIN_CANDIDATES:
        return _skip(
            symbol,
            duration,
            open_time,
            regime_label=regime_label,
            reasons=("insufficient_candidates",),
            candidate_count=len(candidates),
        )
    probability_up = _weighted_probability(candidates)
    confidence = max(probability_up, 1.0 - probability_up)
    final_score = _final_score(candidates, confidence, regime_confidence, regime_label)
    threshold = _decision_threshold(regime_label)
    if confidence < threshold or final_score < MIN_FINAL_SCORE:
        return _skip(
            symbol,
            duration,
            open_time,
            regime_label=regime_label,
            reasons=("edge_below_threshold",),
            probability_up=probability_up,
            confidence=confidence,
            final_score=final_score,
            candidate_count=len(candidates),
        )
    direction = "up" if probability_up >= 0.5 else "down"
    decision = DECISION_UP if direction == "up" else DECISION_DOWN
    reasons = ("passed", f"threshold_{threshold:.2f}")
    return FinalDecision(symbol, duration, open_time, decision, direction, probability_up, confidence, final_score, regime_label, len(candidates), reasons)


def _weighted_probability(candidates: list[DecisionCandidate]) -> float:
    total = sum(item.weight for item in candidates)
    if total <= 0:
        return 0.5
    return _clamp(sum(item.probability_up * item.weight for item in candidates) / total)


def _final_score(
    candidates: list[DecisionCandidate],
    confidence: float,
    regime_confidence: float,
    regime_label: str,
) -> float:
    total = sum(item.weight for item in candidates)
    up = sum(item.weight for item in candidates if item.direction == "up")
    agreement = max(up, total - up) / total if total > 0 else 0.0
    risk_penalty = 0.08 if "high_vol" in regime_label or "uncertain" in regime_label else 0.0
    return _clamp(confidence * 0.45 + agreement * 0.35 + regime_confidence * 0.20 - risk_penalty)


def _decision_threshold(regime_label: str) -> float:
    if "high_vol" in regime_label or "uncertain" in regime_label:
        return HIGH_RISK_CONFIDENCE_MIN
    return BASE_CONFIDENCE_MIN


def _skip(
    symbol: str,
    duration: str,
    open_time: int,
    *,
    regime_label: str,
    reasons: tuple[str, ...],
    probability_up: float | None = None,
    confidence: float = 0.0,
    final_score: float = 0.0,
    candidate_count: int = 0,
) -> FinalDecision:
    return FinalDecision(
        symbol,
        duration,
        int(open_time),
        DECISION_SKIP,
        None,
        probability_up,
        confidence,
        final_score,
        regime_label,
        candidate_count,
        reasons,
    )


def _persist_regime_and_decision(regime, decision: FinalDecision) -> None:
    conn = get_conn()
    try:
        upsert_event_regime(conn, regime)
        _upsert_decision(conn, decision)
        conn.commit()
    finally:
        conn.close()


def _persist_decision(decision: FinalDecision) -> None:
    conn = get_conn()
    try:
        _upsert_decision(conn, decision)
        conn.commit()
    finally:
        conn.close()


def _upsert_decision(conn: Any, decision: FinalDecision) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO event_final_decisions(
          symbol, duration, open_time, decision, direction, probability_up, confidence,
          final_score, regime_label, candidate_count, reason_codes, settled_at,
          decision_correct, actual_direction, exit_price, created_at
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, ?)
        """,
        (
            decision.symbol, decision.duration, decision.open_time, decision.decision,
            decision.direction, decision.probability_up, decision.confidence,
            decision.final_score, decision.regime_label, decision.candidate_count,
            json.dumps(list(decision.reason_codes), ensure_ascii=True), _utc_now(),
        ),
    )


def _prediction_payload(decision: FinalDecision, candidates: list[DecisionCandidate]) -> dict[str, Any]:
    probability = float(decision.probability_up or 0.5)
    return {
        "signal_key": EVENT_FINAL_DECISION_STRATEGY_KEY,
        "strategy_key": EVENT_FINAL_DECISION_STRATEGY_KEY,
        "symbol": decision.symbol,
        "duration": decision.duration,
        "open_time": decision.open_time,
        "direction": decision.direction,
        "probability_up": round(probability, 6),
        "confidence": round(decision.confidence, 6),
        "certainty_label": "EVENT_FINAL_DECISION",
        "trade_quality_score": round(decision.final_score, 6),
        "trade_quality_passed": True,
        "trade_quality_gate": EVENT_FINAL_DECISION_GATE,
        "high_winrate_gate": EVENT_FINAL_DECISION_GATE,
        "high_winrate_rule": decision.regime_label,
        "high_winrate_gate_passed": True,
        "high_winrate_gate_value": decision.final_score,
        "high_winrate_gate_min": MIN_FINAL_SCORE,
        "expected_return": None,
        "model_version": EVENT_FINAL_DECISION_STRATEGY_KEY,
        "model_family": "event_final_decision",
        "validation_win_rate": None,
        "feature_window": len(candidates),
        "model_duration": decision.duration,
        "model_trained_at": _utc_now(),
    }


def _settle_one_decision(conn: Any, symbol: str, duration: str, row: dict[str, Any]) -> bool:
    entry = _kline_close(conn, symbol, int(row["open_time"]))
    exit_price = _kline_close(conn, symbol, int(row["open_time"]) + _duration_ms(duration))
    if entry is None or exit_price is None:
        return False
    actual = "up" if exit_price > entry else "down"
    direction = row.get("direction")
    correct = None if direction is None else int(str(direction).lower() == actual)
    conn.execute(
        """
        UPDATE event_final_decisions
        SET actual_direction = ?, exit_price = ?, decision_correct = ?, settled_at = ?
        WHERE symbol = ? AND duration = ? AND open_time = ?
        """,
        (actual, exit_price, correct, _utc_now(), symbol.strip().upper(), duration, int(row["open_time"])),
    )
    return True


def _kline_close(conn: Any, symbol: str, open_time: int) -> float | None:
    row = conn.execute(
        """
        SELECT close FROM klines
        WHERE symbol = ? AND interval = '1m' AND open_time >= ?
        ORDER BY open_time
        LIMIT 1
        """,
        (symbol.strip().upper(), int(open_time)),
    ).fetchone()
    return None if row is None else float(row["close"])


def _duration_ms(duration: str) -> int:
    return DURATION_TO_MINUTES[duration] * 60_000


def _assert_duration(duration: str) -> None:
    if duration not in DURATION_TO_MINUTES:
        raise ValueError(f"unsupported final decision duration: {duration}")


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
