from __future__ import annotations

from typing import Any

from app.services.factor_combination_signal_payloads import combo_regime_rule_reasons


def factor_combo_rule_reasons(signal: dict[str, Any], cache_reason: str, rule_name: str) -> list[str]:
    member_names = ",".join(str(member["name"]) for member in signal["members"])
    reasons = [
        f"rule={rule_name}",
        f"combo={signal['factorName']}",
        f"combo_rank={signal.get('comboRank') or 1}",
        f"combo_cache={cache_reason}",
        f"members={member_names}",
        f"method={signal['method']}",
        f"historical_win_rate={signal['historicalWinRate']}",
        f"historical_profit_factor={signal['historicalProfitFactor']}",
        f"walk_forward_passed={signal.get('walkForwardPassed')}",
        f"walk_forward_failure={signal.get('walkForwardFailureReason')}",
        f"score={signal['score']}",
        f"signal_threshold={signal.get('signalThreshold')}",
        f"signal_threshold_passed={signal.get('qualityThresholdPassed')}",
        f"quality_gate={signal['qualityGateReason']}",
        f"factor_timing={signal.get('factorTimingReason')}",
        f"factor_timing_passed={signal.get('factorTimingPassed')}",
        f"factor_timing_blocked={','.join(signal.get('factorTimingBlockedMembers') or [])}",
        *combo_regime_rule_reasons(signal),
        f"quality_min_win_rate={signal['qualityMinWinRate']}",
        f"quality_min_profit_factor={signal['qualityMinProfitFactor']}",
    ]
    learning = signal.get("factorLearning")
    if isinstance(learning, dict):
        reasons.extend(_factor_learning_reasons(learning))
    return reasons


def _factor_learning_reasons(learning: dict[str, Any]) -> list[str]:
    matches = learning.get("lossPatternMatches") or []
    return [
        f"factor_learning={learning.get('state')}",
        f"factor_learning_filter_passed={learning.get('filterPassed')}",
        f"factor_learning_confirmations={learning.get('confirmationCount')}",
        f"factor_learning_loss_matches={len(matches) if isinstance(matches, list) else 0}",
    ]
