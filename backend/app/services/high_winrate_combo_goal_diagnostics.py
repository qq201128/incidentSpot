from __future__ import annotations

from typing import Any

from app.services.factor_performance_metrics import BACKTEST_MIN_PERIODS

DEFAULT_SEARCH_CANDIDATE_LIMIT = 30


def candidate_diagnostics(
    numeric_column_count: int,
    eligible_count: int,
    rejected: list[tuple[str, int]],
    max_valid_pairs: int,
) -> dict[str, Any]:
    reason = None
    if numeric_column_count == 0:
        reason = "no_numeric_factor_columns"
    elif eligible_count == 0:
        reason = "no_candidate_factors_met_min_periods"
    return {
        "stage": "candidate_factor_filter",
        "reason": reason,
        "minPeriods": BACKTEST_MIN_PERIODS,
        "numericFactorColumns": numeric_column_count,
        "eligibleCandidateFactors": eligible_count,
        "rejectedCandidateFactors": len(rejected),
        "maxValidPairs": max_valid_pairs,
        "topRejectedByValidPairs": top_rejected_candidate_payloads(rejected),
    }


def top_rejected_candidate_payloads(rejected: list[tuple[str, int]]) -> list[dict[str, Any]]:
    top = sorted(rejected, key=lambda row: row[1], reverse=True)[:10]
    return [{"name": name, "validPairs": valid_pairs} for name, valid_pairs in top]


def ranked_search_diagnostics(names: list[str]) -> dict[str, Any]:
    return {
        "stage": "combo_threshold_gates",
        "searchCandidateLimit": DEFAULT_SEARCH_CANDIDATE_LIMIT,
        "selectedCandidateFactors": len(names),
        "testedCombinations": 0,
        "testedThresholdEvaluations": 0,
        "gateFailures": {
            "min_trades_below_min": 0,
            "win_rate_below_min": 0,
            "profit_factor_below_min": 0,
        },
        "passedThresholdEvaluations": 0,
        "bestRejected": None,
        "failureReason": None,
    }


def ranked_failure_reason(
    has_scores: bool,
    hit_count: int,
    selected_count: int,
    min_combo_size: int,
    diagnostics: dict[str, Any],
) -> str | None:
    if hit_count > 0:
        return None
    if not has_scores:
        return "no_candidate_factors"
    if selected_count < min_combo_size:
        return "not_enough_selected_candidate_factors"
    if diagnostics["testedThresholdEvaluations"] == diagnostics["gateFailures"]["min_trades_below_min"]:
        return "all_combo_thresholds_below_min_trades"
    return "no_combo_met_target_gates"


def record_combo_gate_result(
    diagnostics: dict[str, Any],
    hit: Any,
    rejected: dict[str, Any] | None,
) -> None:
    if hit is not None:
        diagnostics["passedThresholdEvaluations"] += 1
        return
    if rejected is None:
        return
    diagnostics["gateFailures"][rejected["reason"]] += 1
    record_best_rejected(diagnostics, rejected)


def record_best_rejected(diagnostics: dict[str, Any], rejected: dict[str, Any]) -> None:
    current = diagnostics["bestRejected"]
    if current is not None and rejected_rank_key(current) >= rejected_rank_key(rejected):
        return
    diagnostics["bestRejected"] = rejected_payload(rejected)


def rejected_rank_key(row: dict[str, Any]) -> tuple[float, float, float, int]:
    return (float(row["winRate"]), float(row["profitFactor"]), float(row["avgReturn"]), int(row["trades"]))


def rejected_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "members": list(row["members"]),
        "threshold": row["threshold"],
        "reason": row["reason"],
        "trades": row["trades"],
        "winRate": round(row["winRate"], 4),
        "profitFactor": round(row["profitFactor"], 4),
        "avgReturn": round(row["avgReturn"], 8),
    }
