from __future__ import annotations

from fastapi import HTTPException

from app.services.prediction_policy import trade_confidence_threshold_for_duration, trade_policy_payload
from app.services.strategy_registry import MANUAL_STRATEGY_KEY


def validate_ai_trade_probability(
    p_up: float | None,
    event_interval: str,
    quality_score: float | None = None,
    quality_passed: bool | int | None = None,
    high_winrate_passed: bool | int | None = None,
    strategy_key: str | None = None,
) -> None:
    if p_up is None:
        return

    try:
        threshold = trade_confidence_threshold_for_duration(event_interval)
        policy = trade_policy_payload(
            event_interval,
            strategy_key=_policy_strategy_key(strategy_key),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    side_probability = max(float(p_up), 1 - float(p_up))
    target = policy.get("productionTarget") or {}
    if target.get("passed") is not True:
        raise HTTPException(
            status_code=400,
            detail="规则回测未达标：需要每日胜率 >= 70%，当前禁止规则自动下单",
        )
    if side_probability < threshold:
        raise HTTPException(
            status_code=400,
            detail=f"规则最高方向置信需 >= {int(threshold * 100)}%，当前不建议下单",
        )
    if bool(policy.get("highWinrateGateEnabled")):
        _validate_high_winrate_gate(high_winrate_passed)
        return

    score_min = float(policy.get("tradeQualityScoreMin") or 0)
    score = float(quality_score or 0)
    if bool(quality_passed) and score >= score_min:
        return
    raise HTTPException(
        status_code=400,
        detail=f"规则质量分需 >= {int(score_min * 100)}%，当前不建议下单",
    )


def _validate_high_winrate_gate(high_winrate_passed: bool | int | None) -> None:
    if bool(high_winrate_passed):
        return
    raise HTTPException(
        status_code=400,
        detail="规则高胜率门控未通过，当前不建议下单",
    )


def _policy_strategy_key(strategy_key: str | None) -> str | None:
    if strategy_key == MANUAL_STRATEGY_KEY:
        return None
    return strategy_key
