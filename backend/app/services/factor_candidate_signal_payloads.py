from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.services.factor_backtest_gate import meets_backtest_gate
from app.services.factor_candidate_signal_keys import factor_candidate_signal_key
from app.services.factor_candidate_signal_utils import oos_win_rate, signal_rule_reasons, utc_now

SIGNAL_RULE_NAME = "factor_candidate_signal_v1"
LIVE_STABLE_SIGNAL_RULE_NAME = "factor_candidate_paper_live_stable_v1"
LIVE_ADMISSION_KEY = "_liveCandidateAdmission"
PROBABILITY_DECIMALS = 4
SCORE_DECIMALS = 6


@dataclass(frozen=True)
class FactorSignal:
    score: float
    median: float
    direction: str
    confidence: float
    entry_price: float
    index: Any
    orientation: int
    regime: dict[str, Any]
    regime_passed: bool
    regime_reason: str
    regime_min_win_rate: float | None


def factor_candidate_prediction_payload(
    row: dict[str, Any],
    signal: FactorSignal,
    *,
    symbol: str,
    duration: str,
    entry_open_time: int,
) -> dict[str, Any]:
    signal_key = factor_candidate_signal_key(str(row["factorName"]))
    probability_up = signal.confidence if signal.direction == "up" else 1.0 - signal.confidence
    return {
        **_core_payload(signal_key, row, signal, symbol=symbol, duration=duration, entry_open_time=entry_open_time),
        **_model_payload(row, duration),
        "rule_score": round(signal.score, SCORE_DECIMALS),
        "rule_reasons": _rule_reasons(row, signal),
        "signal_source": "factor_candidate_signal",
        "probability_up": round(probability_up, PROBABILITY_DECIMALS),
    }


def _core_payload(
    signal_key: str,
    row: dict[str, Any],
    signal: FactorSignal,
    *,
    symbol: str,
    duration: str,
    entry_open_time: int,
) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "signal_key": signal_key,
        "strategy_key": signal_key,
        "duration": duration,
        "open_time": int(entry_open_time),
        "entry_price": signal.entry_price,
        "direction": signal.direction,
        "confidence": round(signal.confidence, PROBABILITY_DECIMALS),
        "certainty_label": "FACTOR_CANDIDATE_OBSERVE",
        "trade_quality_score": _quality_score(row, signal),
        "trade_quality_passed": _trade_quality_passed(row, signal),
        "trade_quality_gate": _trade_quality_gate(row),
        "high_winrate_gate": SIGNAL_RULE_NAME,
        "high_winrate_rule": str(row["factorName"]),
        "high_winrate_gate_passed": True,
        "high_winrate_gate_value": _high_winrate_gate_value(row),
        "high_winrate_gate_min": None,
    }


def _model_payload(row: dict[str, Any], duration: str) -> dict[str, Any]:
    return {
        "expected_return": None,
        "model_version": str(row["factorName"]),
        "model_family": "factor",
        "model_duration": duration,
        "model_trained_at": utc_now(),
        "oos_win_rate": oos_win_rate(row),
        "walk_forward_result": row.get("walkForward"),
        "recent_rolling_result": row.get("recentRollingResult"),
        "data_freshness_status": "fresh",
        "missing_feature_status": "complete",
    }


def _rule_reasons(row: dict[str, Any], signal: FactorSignal) -> list[str]:
    return [
        *signal_rule_reasons(row, signal, rule_name=SIGNAL_RULE_NAME, decimals=SCORE_DECIMALS),
        f"live_admission={_live_admission_reason(row)}",
        f"regime={signal.regime.get('regimeLabel')}",
        f"regime_passed={signal.regime_passed}",
        f"regime_reason={signal.regime_reason}",
        f"regime_min_win_rate={signal.regime_min_win_rate}",
    ]


def _trade_quality_passed(row: dict[str, Any], signal: FactorSignal) -> bool:
    if _live_admission_stable(row):
        return signal.regime_passed
    return meets_backtest_gate(row) and signal.regime_passed


def _trade_quality_gate(row: dict[str, Any]) -> str:
    return LIVE_STABLE_SIGNAL_RULE_NAME if _live_admission_stable(row) else SIGNAL_RULE_NAME


def _quality_score(row: dict[str, Any], signal: FactorSignal) -> float:
    admission = row.get(LIVE_ADMISSION_KEY)
    metrics = admission.get("metrics") if isinstance(admission, dict) else {}
    win_rate = metrics.get("winRate") if isinstance(metrics, dict) else None
    value = win_rate if win_rate is not None else signal.confidence
    return round(float(value), PROBABILITY_DECIMALS)


def _high_winrate_gate_value(row: dict[str, Any]) -> Any:
    admission = row.get(LIVE_ADMISSION_KEY)
    metrics = admission.get("metrics") if isinstance(admission, dict) else {}
    return metrics.get("winRate") if isinstance(metrics, dict) and metrics.get("winRate") is not None else row.get("winRate")


def _live_admission_stable(row: dict[str, Any]) -> bool:
    admission = row.get(LIVE_ADMISSION_KEY)
    return isinstance(admission, dict) and admission.get("status") == "paper_stable"


def _live_admission_reason(row: dict[str, Any]) -> str:
    admission = row.get(LIVE_ADMISSION_KEY)
    if not isinstance(admission, dict):
        return "not_live_enabled"
    return str(admission.get("reason") or admission.get("status") or "unknown")
