from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.services.factor_combination_quality_gate import (
    LIVE_MIN_PROFIT_FACTOR,
    LIVE_MIN_WIN_RATE,
    min_total_periods,
    signal_threshold,
)
from app.services.factor_combination_service import COMBINATION_METHOD
from app.services.factor_combo_simulation_keys import is_high_winrate_combo_name

PROBABILITY_DECIMALS = 4
SCORE_DECIMALS = 6
COMBO_MEDIAN_DECIMALS = 6


@dataclass(frozen=True)
class LiveSignalPayloadContext:
    row: dict[str, Any]
    symbol: str
    duration: str
    score: float
    historical_median: float
    index: Any
    direction: str
    confidence: float
    quality: dict[str, Any]
    timing: Any
    regime: dict[str, Any]


def live_signal_payload(ctx: LiveSignalPayloadContext) -> dict[str, Any]:
    return {
        **_core_payload(ctx),
        **_history_payload(ctx.row),
        **_quality_payload(ctx),
        "qualityMinWinRate": LIVE_MIN_WIN_RATE,
        "qualityMinProfitFactor": LIVE_MIN_PROFIT_FACTOR,
        "qualityMinPeriods": min_total_periods(ctx.row),
        "frameIndex": str(ctx.index),
    }


def _core_payload(ctx: LiveSignalPayloadContext) -> dict[str, Any]:
    probability_up = ctx.confidence if ctx.direction == "up" else 1.0 - ctx.confidence
    return {
        "symbol": ctx.symbol.upper(),
        "duration": ctx.duration,
        "factorName": ctx.row.get("factorName"),
        "factorDisplayName": ctx.row.get("factorDisplayName"),
        "comboRank": ctx.row.get("comboRank"),
        "members": _row_members(ctx.row),
        "direction": ctx.direction,
        "probabilityUp": round(probability_up, PROBABILITY_DECIMALS),
        "confidence": round(ctx.confidence, PROBABILITY_DECIMALS),
        "score": round(ctx.score, SCORE_DECIMALS),
        "historicalMedianScore": round(ctx.historical_median, COMBO_MEDIAN_DECIMALS),
        "source": "factor_combination_ranking",
        "method": ctx.row.get("method") or COMBINATION_METHOD,
    }


def _history_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "historicalWinRate": row.get("winRate"),
        "historicalProfitFactor": row.get("profitFactor"),
        "historicalSharpe": row.get("sharpe"),
        "historicalIr": row.get("ir"),
        "historicalTotalPeriods": row.get("totalPeriods"),
        "oosWinRate": _oos_win_rate(row),
        "walkForwardResult": row.get("walkForward"),
        "recentRollingResult": row.get("recentRollingResult"),
        "walkForwardPassed": row.get("walkForwardPassed"),
        "walkForwardFailureReason": row.get("walkForwardFailureReason"),
    }


def _quality_payload(ctx: LiveSignalPayloadContext) -> dict[str, Any]:
    return {
        "qualityPassed": ctx.quality["passed"],
        "qualityMetricsPassed": ctx.quality["metricsPassed"],
        "qualityThresholdPassed": ctx.quality["thresholdPassed"],
        "signalThreshold": signal_threshold(ctx.row),
        "qualityEntryWindowPassed": ctx.quality["entryWindowPassed"],
        "factorTimingMode": ctx.timing.mode,
        "factorTimingPassed": ctx.quality["factorTimingPassed"],
        "factorTimingReason": ctx.timing.reason,
        "factorTimingEligibleMembers": list(ctx.timing.eligible_members),
        "factorTimingBlockedMembers": list(ctx.timing.blocked_members),
        "regime": ctx.regime,
        "regimePassed": ctx.quality["regimePassed"],
        "regimeReason": ctx.quality["regimeReason"],
        "regimeMinWinRate": ctx.quality["regimeMinWinRate"],
        "liveEvidenceSource": ctx.quality["liveEvidenceSource"],
        "liveEvidenceWinRate": ctx.quality["liveEvidenceWinRate"],
        "liveEvidenceProfitFactor": ctx.quality["liveEvidenceProfitFactor"],
        "liveEvidenceSampleCount": ctx.quality["liveEvidenceSampleCount"],
        "qualityGateReason": ctx.quality["reason"],
    }


def combo_regime_rule_reasons(signal: dict[str, Any]) -> list[str]:
    return [
        f"regime={regime_label(signal)}",
        f"regime_passed={signal.get('regimePassed')}",
        f"regime_reason={signal.get('regimeReason')}",
    ]


def regime_label(signal: dict[str, Any]) -> str:
    regime = signal.get("regime")
    if not isinstance(regime, dict):
        return "unknown"
    return str(regime.get("regimeLabel") or "unknown")


def _row_members(row: dict[str, Any]) -> list[dict[str, Any]]:
    members = row.get("members")
    if not isinstance(members, list) or not members:
        raise ValueError(f"combination row missing members: {row.get('factorName')}")
    return [dict(member) for member in members]


def _oos_win_rate(row: dict[str, Any]) -> Any:
    walk_forward = row.get("walkForward")
    if isinstance(walk_forward, dict):
        return walk_forward.get("oosWinRate")
    return row.get("oosWinRate")


def live_direction(row: dict[str, Any], score: float, historical_median: float) -> str:
    threshold = signal_threshold(row)
    if is_high_winrate_combo_name(str(row.get("factorName") or "")) and threshold > 0:
        if score >= threshold:
            return "up"
        if score <= -threshold:
            return "down"
    return "up" if score >= historical_median else "down"
