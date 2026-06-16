from __future__ import annotations

from dataclasses import dataclass
from typing import Any

STATUS_STABLE = "paper_stable"
STATUS_COLLECTING = "paper_collecting"
STATUS_BACKTEST = "backtest_candidate"
SORT_SAMPLE_PRIOR = 50.0
WILSON_Z = 1.96
SORT_WEIGHTS = {
    "winRateLowerBound": 0.30,
    "recentWindowStability": 0.20,
    "rollingWindowStability": 0.15,
    "returnQuality": 0.15,
    "regimeConsistency": 0.10,
    "sampleConfidence": 0.10,
}


@dataclass(frozen=True)
class ValidationMetadata:
    walk_forward_result: Any
    recent_rolling_result: Any


def candidate_rank_key(candidate: dict[str, Any]) -> tuple[float, float, float, float, float, float, float, float, float]:
    metrics = candidate["metrics"]
    robust = candidate.get("robustScore")
    robust_score = _num(robust) if robust is not None else candidate_robust_score(candidate)["score"]
    return (
        _status_priority(str(candidate.get("status") or "")),
        robust_score,
        _num(candidate.get("oosWinRate")),
        _walk_forward_score(candidate.get("walkForwardResult")),
        _recent_rolling_score(candidate.get("recentRollingResult")),
        _stability_score(metrics.get("paperStability") or {}),
        _num(metrics.get("profitFactor")),
        _num(metrics.get("avgReturn")),
        float(metrics.get("sampleCount") or 0),
    )


def performance_comparison(
    candidate: dict[str, Any],
    metrics: dict[str, Any],
    metadata: ValidationMetadata,
) -> dict[str, Any]:
    backtest = candidate.get("high_winrate_gate_value")
    paper = metrics.get("winRate")
    return {
        "policy": "backtest_oos_walk_forward_recent_rolling_are_prefilter_only",
        "backtestWinRate": backtest,
        "oosWinRate": candidate.get("oos_win_rate"),
        "walkForwardResult": metadata.walk_forward_result,
        "recentRollingResult": metadata.recent_rolling_result,
        "validationWinRate": candidate.get("validation_win_rate"),
        "paperLiveWinRate": paper,
        "paperLiveSampleCount": metrics.get("sampleCount"),
        "winRateGap": _gap(backtest, paper),
        "paperLiveStatus": None,
    }


def candidate_robust_score(candidate: dict[str, Any]) -> dict[str, Any]:
    metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
    components = {
        "winRateLowerBound": _win_rate_lower_bound(candidate, metrics),
        "recentWindowStability": _recent_window_stability_score(metrics, candidate.get("paperLiveWinRate")),
        "rollingWindowStability": _rolling_window_stability_score(metrics, candidate.get("paperLiveWinRate")),
        "returnQuality": _return_quality_score(metrics),
        "regimeConsistency": _regime_consistency_score(candidate.get("regimeValidation")),
        "sampleConfidence": _sample_confidence(metrics, candidate.get("paperLiveSampleCount")),
    }
    penalties = {
        "backtestGap": _backtest_gap_penalty(candidate, metrics),
        "lossStreak": _loss_streak_penalty(metrics),
    }
    weighted = sum(SORT_WEIGHTS[key] * components[key] for key in SORT_WEIGHTS)
    score = weighted - sum(penalties.values())
    return {
        "score": round(score, 6),
        "policy": "robust_paper_live_score_v1",
        "weights": dict(SORT_WEIGHTS),
        "components": {key: round(value, 6) for key, value in components.items()},
        "penalties": {key: round(value, 6) for key, value in penalties.items()},
    }


def focus_pool(candidates: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    eligible = [row for row in candidates if row.get("status") not in {"paper_failed", "invalid_data_leakage"}]
    return eligible[:limit]


def _status_priority(status: str) -> float:
    if status == STATUS_STABLE:
        return 3.0
    if status == STATUS_COLLECTING:
        return 2.0
    if status == STATUS_BACKTEST:
        return 1.0
    return 0.0


def _stability_score(stability: dict[str, Any]) -> float:
    rolling = stability.get("rollingWindows") if isinstance(stability, dict) else None
    if not isinstance(rolling, list) or not rolling:
        return float("-inf")
    rates = [_num(row.get("winRate")) for row in rolling if isinstance(row, dict)]
    return min(rates) if rates else float("-inf")


def _walk_forward_score(value: Any) -> float:
    if not isinstance(value, dict):
        return float("-inf")
    if value.get("status") == "passed" or value.get("passed") is True:
        return max(_num(value.get("score")), 0.0)
    return _num(value.get("score"))


def _recent_rolling_score(value: Any) -> float:
    if isinstance(value, dict):
        stability = value.get("paperStability") or value
        return _stability_score(stability if isinstance(stability, dict) else {})
    return float("-inf")


def _gap(first: Any, second: Any) -> float | None:
    if first is None or second is None:
        return None
    return round(float(first) - float(second), 4)


def _win_rate_lower_bound(candidate: dict[str, Any], metrics: dict[str, Any]) -> float:
    win_rate = metrics.get("winRate")
    if win_rate is None:
        win_rate = candidate.get("paperLiveWinRate")
    return _wilson_lower_bound(win_rate, _effective_sample_count(candidate, metrics))


def _recent_window_stability_score(metrics: dict[str, Any], fallback: Any) -> float:
    windows = metrics.get("paperLiveWindows") if isinstance(metrics.get("paperLiveWindows"), dict) else {}
    rates = [
        _window_win_rate(windows.get("recent30")),
        _window_win_rate(windows.get("recent60")),
        _window_win_rate(windows.get("recent100")),
    ]
    values = [value for value in rates if value is not None]
    if not values:
        return _win_rate_lower_bound({"paperLiveWinRate": fallback}, metrics)
    return _stable_win_rate_score(values, fallback)


def _rolling_window_stability_score(metrics: dict[str, Any], fallback: Any) -> float:
    stability = metrics.get("paperStability") if isinstance(metrics.get("paperStability"), dict) else {}
    windows = stability.get("rollingWindows") if isinstance(stability.get("rollingWindows"), list) else []
    values = [_window_win_rate(window) for window in windows if isinstance(window, dict)]
    values = [value for value in values if value is not None]
    if not values:
        return _recent_window_stability_score(metrics, fallback)
    return _stable_win_rate_score(values, fallback)


def _return_quality_score(metrics: dict[str, Any]) -> float:
    pf = metrics.get("profitFactor")
    avg_return = metrics.get("avgReturn")
    pf_score = 0.0 if pf is None else _clamp((float(pf) - 1.0) / 1.0)
    avg_return_score = 0.0 if avg_return is None else _clamp(0.5 + float(avg_return) / 0.01)
    return 0.7 * pf_score + 0.3 * avg_return_score


def _regime_consistency_score(regime_validation: Any) -> float:
    if not isinstance(regime_validation, dict):
        return 0.5
    buckets = _regime_buckets(regime_validation)
    if not buckets:
        return 0.5
    total_samples = sum(bucket["sampleCount"] for bucket in buckets)
    if total_samples <= 0:
        return 0.5
    weighted_lower = sum(
        _wilson_lower_bound(bucket["winRate"], bucket["sampleCount"]) * bucket["sampleCount"]
        for bucket in buckets
    ) / total_samples
    rates = [bucket["winRate"] for bucket in buckets]
    spread = max(rates) - min(rates)
    coverage_confidence = (len(buckets) / (len(buckets) + 3.0)) ** 0.5
    return _clamp(weighted_lower - spread * 0.35 + coverage_confidence * 0.1)


def _sample_confidence(metrics: dict[str, Any], fallback_sample_count: Any) -> float:
    sample_count = _effective_sample_count({"paperLiveSampleCount": fallback_sample_count}, metrics)
    if sample_count <= 0:
        return 0.0
    return (sample_count / (sample_count + SORT_SAMPLE_PRIOR)) ** 0.5


def _backtest_gap_penalty(candidate: dict[str, Any], metrics: dict[str, Any]) -> float:
    backtest = candidate.get("backtestWinRate", candidate.get("high_winrate_gate_value"))
    paper = metrics.get("winRate")
    if paper is None:
        paper = candidate.get("paperLiveWinRate")
    if backtest is None or paper is None:
        return 0.0
    gap = float(backtest) - float(paper)
    return 0.0 if gap <= 0 else _clamp(gap, 0.0, 0.25)


def _loss_streak_penalty(metrics: dict[str, Any]) -> float:
    losses = metrics.get("maxConsecutiveLosses") or metrics.get("consecutiveLosses") or 0
    return max(float(losses) - 2.0, 0.0) * 0.03


def _effective_sample_count(candidate: dict[str, Any], metrics: dict[str, Any]) -> float:
    return max(float(metrics.get("sampleCount") or 0), float(candidate.get("paperLiveSampleCount") or 0))


def _window_win_rate(window: Any) -> float | None:
    if not isinstance(window, dict):
        return None
    if int(window.get("sampleCount") or 0) <= 0 or window.get("winRate") is None:
        return None
    return float(window["winRate"])


def _stable_win_rate_score(values: list[float], fallback: Any) -> float:
    if not values:
        return 0.0 if fallback is None else _clamp(float(fallback))
    average = sum(values) / len(values)
    reference = average if fallback is None else float(fallback)
    average_deviation = sum(abs(value - reference) for value in values) / len(values)
    return _clamp(average - average_deviation)


def _regime_buckets(regime_validation: dict[str, Any]) -> list[dict[str, float]]:
    buckets: list[dict[str, float]] = []
    for payload in regime_validation.values():
        if not isinstance(payload, dict):
            continue
        rate = payload.get("winRate", payload.get("accuracy"))
        sample_count = payload.get("sampleCount", payload.get("n", 0))
        if rate is None or float(sample_count or 0) <= 0:
            continue
        buckets.append({"winRate": float(rate), "sampleCount": float(sample_count)})
    return buckets


def _wilson_lower_bound(win_rate: Any, sample_count: Any) -> float:
    n = float(sample_count or 0)
    if n <= 0 or win_rate is None:
        return 0.0
    p = _clamp(float(win_rate))
    z2 = WILSON_Z * WILSON_Z
    denominator = 1 + z2 / n
    centre = p + z2 / (2 * n)
    margin = WILSON_Z * ((p * (1 - p) + z2 / (4 * n)) / n) ** 0.5
    return _clamp((centre - margin) / denominator)


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return min(max(float(value), minimum), maximum)


def _num(value: Any) -> float:
    return float(value) if value is not None else float("-inf")
