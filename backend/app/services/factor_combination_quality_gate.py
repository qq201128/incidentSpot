from __future__ import annotations

import os
from dataclasses import dataclass
from math import isfinite
from typing import Any

from app.services.factor_backtest_service import BACKTEST_MIN_PERIODS
from app.services.factor_combo_simulation_keys import is_high_winrate_combo_name
from app.services.factor_learning_common import SUCCESS_PROFIT_FACTOR_MIN, SUCCESS_WIN_RATE_MIN
from app.services.factor_signal_timing import FactorSignalTiming
from app.services.kline_timing import is_within_entry_grace

LIVE_MIN_WIN_RATE = SUCCESS_WIN_RATE_MIN
LIVE_MIN_PROFIT_FACTOR = SUCCESS_PROFIT_FACTOR_MIN
LIVE_MIN_TOTAL_PERIODS = BACKTEST_MIN_PERIODS
LIVE_MIN_OOS_SAMPLE_COUNT = 50
LIVE_MIN_PAPER_SAMPLE_COUNT = 30
DEFAULT_SIGNAL_THRESHOLD = 0.0


@dataclass(frozen=True)
class EntryWindow:
    open_time: int | None
    grace_ms: int | None


@dataclass(frozen=True)
class RegimeGate:
    passed: bool
    reason: str
    regime: dict[str, Any] | None = None
    min_win_rate: float | None = None


def resolve_apply_quality_gate(requested: bool) -> bool:
    env = os.getenv("FACTOR_COMBO_LIVE_QUALITY_GATE", "").strip().lower()
    if env in {"1", "true", "yes"}:
        return True
    if env in {"0", "false", "no"}:
        return False
    return requested


def backtest_aligned_quality() -> dict[str, Any]:
    return {
        "passed": True,
        "metricsPassed": True,
        "thresholdPassed": True,
        "entryWindowPassed": True,
        "factorTimingPassed": True,
        "regimePassed": True,
        "regimeReason": "passed",
        "regime": None,
        "regimeMinWinRate": None,
        "liveEvidenceSource": "disabled_backtest_alignment",
        "liveEvidenceWinRate": None,
        "liveEvidenceProfitFactor": None,
        "liveEvidenceSampleCount": 0,
        "reason": "backtest_aligned",
    }


def quality_gate(
    row: dict[str, Any],
    confidence: float,
    window: EntryWindow,
    *,
    timing: FactorSignalTiming,
    score: float,
    regime: RegimeGate | None = None,
) -> dict[str, Any]:
    evidence = live_evidence(row)
    metric_reason = quality_metric_reason(row, confidence, evidence=evidence)
    threshold_reason = threshold_reason_for(row, score)
    entry_passed = entry_window_passed(window)
    regime_gate = regime or RegimeGate(True, "passed")
    metrics_passed = metric_reason == "passed"
    threshold_passed = threshold_reason == "passed"
    passed = (
        metrics_passed
        and threshold_passed
        and entry_passed is not False
        and timing.passed
        and regime_gate.passed
    )
    return {
        "passed": passed,
        "metricsPassed": metrics_passed,
        "thresholdPassed": threshold_passed,
        "entryWindowPassed": entry_passed,
        "factorTimingPassed": timing.passed,
        "regimePassed": regime_gate.passed,
        "regimeReason": regime_gate.reason,
        "regime": regime_gate.regime,
        "regimeMinWinRate": regime_gate.min_win_rate,
        "liveEvidenceSource": evidence["source"],
        "liveEvidenceWinRate": evidence["winRate"],
        "liveEvidenceProfitFactor": evidence["profitFactor"],
        "liveEvidenceSampleCount": evidence["sampleCount"],
        "reason": quality_reason(
            metric_reason=metric_reason,
            threshold_reason=threshold_reason,
            entry_passed=entry_passed,
            timing=timing,
            regime=regime_gate,
        ),
    }


def quality_metric_reason(
    row: dict[str, Any],
    confidence: float,
    *,
    evidence: dict[str, Any] | None = None,
) -> str:
    selected = evidence or live_evidence(row)
    if selected["reason"] != "passed":
        return str(selected["reason"])
    if float(selected["winRate"]) < LIVE_MIN_WIN_RATE:
        return "win_rate_below_min"
    profit_factor = finite_float(selected.get("profitFactor"))
    if profit_factor is None:
        return "profit_factor_missing"
    if profit_factor < LIVE_MIN_PROFIT_FACTOR:
        return "profit_factor_below_min"
    return period_and_walk_forward_reason(row)


def period_and_walk_forward_reason(row: dict[str, Any]) -> str:
    min_periods = min_total_periods(row)
    total_periods = finite_float(row.get("totalPeriods"))
    if total_periods is None:
        return "total_periods_missing"
    if total_periods < min_periods:
        return "total_periods_below_min"
    reason = walk_forward_reason(row)
    return reason or "passed"


def live_evidence(row: dict[str, Any]) -> dict[str, Any]:
    paper = _paper_live_evidence(row)
    if paper["reason"] == "passed":
        return paper
    oos = _oos_evidence(row)
    if oos["reason"] == "passed":
        return oos
    return paper if paper["reason"] != "paper_live_evidence_missing" else oos


def _paper_live_evidence(row: dict[str, Any]) -> dict[str, Any]:
    sample_count = int(finite_float(row.get("paperLiveSampleCount")) or 0)
    win_rate = finite_float(row.get("paperLiveWinRate"))
    profit_factor = finite_float(row.get("paperLiveProfitFactor"))
    reason = _paper_live_evidence_reason(sample_count, win_rate, profit_factor)
    return _evidence_payload(
        source="paper_live",
        sample_count=sample_count,
        win_rate=win_rate,
        profit_factor=profit_factor,
        reason=reason,
    )


def _paper_live_evidence_reason(
    sample_count: int,
    win_rate: float | None,
    profit_factor: float | None,
) -> str:
    if sample_count <= 0 and win_rate is None and profit_factor is None:
        return "paper_live_evidence_missing"
    if sample_count < LIVE_MIN_PAPER_SAMPLE_COUNT:
        return "paper_live_sample_count_below_min"
    if win_rate is None:
        return "paper_live_win_rate_missing"
    if profit_factor is None:
        return "paper_live_profit_factor_missing"
    return "passed"


def _oos_evidence(row: dict[str, Any]) -> dict[str, Any]:
    walk_forward = row.get("walkForward") if isinstance(row.get("walkForward"), dict) else {}
    test = walk_forward.get("test") if isinstance(walk_forward.get("test"), dict) else {}
    win_rate = _first_float(walk_forward.get("oosWinRate"), row.get("oosWinRate"), test.get("winRate"))
    profit_factor = _first_float(
        walk_forward.get("oosProfitFactor"),
        row.get("oosProfitFactor"),
        test.get("profitFactor"),
    )
    sample_count = int(_first_float(walk_forward.get("oosSampleCount"), row.get("oosSampleCount"), test.get("sampleCount")) or 0)
    reason = _oos_evidence_reason(sample_count, win_rate, profit_factor)
    return _evidence_payload(
        source="oos",
        sample_count=sample_count,
        win_rate=win_rate,
        profit_factor=profit_factor,
        reason=reason,
    )


def _oos_evidence_reason(
    sample_count: int,
    win_rate: float | None,
    profit_factor: float | None,
) -> str:
    if sample_count <= 0 and win_rate is None and profit_factor is None:
        return "oos_evidence_missing"
    if sample_count < LIVE_MIN_OOS_SAMPLE_COUNT:
        return "oos_sample_count_below_min"
    if win_rate is None:
        return "oos_win_rate_missing"
    if profit_factor is None:
        return "oos_profit_factor_missing"
    return "passed"


def _evidence_payload(
    *,
    source: str,
    sample_count: int,
    win_rate: float | None,
    profit_factor: float | None,
    reason: str,
) -> dict[str, Any]:
    return {
        "source": source,
        "sampleCount": sample_count,
        "winRate": win_rate,
        "profitFactor": profit_factor,
        "reason": reason,
    }


def walk_forward_reason(row: dict[str, Any]) -> str | None:
    if is_high_winrate_combo_name(str(row.get("factorName") or "")):
        return None
    passed = row.get("walkForwardPassed")
    if passed is True or passed == 1:
        return None
    if passed is False or passed == 0:
        return str(row.get("walkForwardFailureReason") or "walk_forward_failed")
    return "walk_forward_missing"


def min_total_periods(row: dict[str, Any]) -> float:
    value = finite_float(row.get("minTrades"))
    if value is None:
        return float(LIVE_MIN_TOTAL_PERIODS)
    if value <= 0:
        raise ValueError(f"combination row has invalid minTrades: {row.get('factorName')}")
    return value


def threshold_reason_for(row: dict[str, Any], score: float) -> str:
    if abs(score) < signal_threshold(row):
        return "signal_threshold_not_met"
    return "passed"


def signal_threshold(row: dict[str, Any]) -> float:
    threshold = finite_float(row.get("threshold"))
    if threshold is None:
        return DEFAULT_SIGNAL_THRESHOLD
    if threshold < 0:
        raise ValueError(f"combination row has negative threshold: {row.get('factorName')}")
    return threshold


def entry_window_passed(window: EntryWindow) -> bool | None:
    if window.open_time is None or window.grace_ms is None:
        return None
    return is_within_entry_grace(int(window.open_time), grace_ms=int(window.grace_ms))


def quality_reason(
    *,
    metric_reason: str,
    threshold_reason: str,
    entry_passed: bool | None,
    timing: FactorSignalTiming,
    regime: RegimeGate,
) -> str:
    if metric_reason != "passed":
        return metric_reason
    if threshold_reason != "passed":
        return threshold_reason
    if not timing.passed:
        return timing.reason
    if not regime.passed:
        return regime.reason
    if entry_passed is False:
        return "entry_window_closed"
    return "passed"


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if isfinite(number) else None


def _first_float(*values: Any) -> float | None:
    for value in values:
        number = finite_float(value)
        if number is not None:
            return number
    return None
