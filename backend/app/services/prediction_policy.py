from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.services.rule_config import (
    RULE_DURATION,
    RULE_GATE_NAME,
    SUPPORTED_RULE_DURATIONS,
    RULE_MIN_CONFIDENCE,
    RULE_MIN_QUALITY_SCORE,
    RULE_TARGET_WIN_RATE,
)
from app.services.strategy_registry import DEFAULT_STRATEGY_KEY, StrategyDefinition, strategy_definition

RULE_ARTIFACT_DIR = Path(__file__).resolve().parent.parent.parent / "rules"


def trade_confidence_threshold_for_duration(duration: str) -> float:
    _assert_supported_duration(duration)
    return RULE_MIN_CONFIDENCE


def trade_policy_payload(
    duration: str,
    *,
    strategy_key: str | None = DEFAULT_STRATEGY_KEY,
) -> dict[str, Any]:
    _assert_supported_duration(duration)
    strategy = strategy_definition(strategy_key)
    backtest = _load_rule_backtest(duration, strategy.key)
    overall = backtest.get("overall", {}) if backtest else {}
    return {
        "strategyKey": strategy.key,
        "tradeConfidenceThreshold": RULE_MIN_CONFIDENCE,
        "qualityTradeConfidenceThreshold": RULE_MIN_CONFIDENCE,
        "effectiveTradeConfidenceThreshold": RULE_MIN_CONFIDENCE,
        "tradePolicySource": strategy.signal_source,
        "tradePolicyGatesEnabled": strategy.uses_trade_policy_gates,
        "productionTarget": _production_target(backtest, strategy),
        "highWinrateGate": RULE_GATE_NAME,
        "highWinrateGateEnabled": strategy.requires_high_winrate_gate,
        "highWinrateMinConfidence": RULE_MIN_CONFIDENCE,
        "highWinrateRules": [_rule_payload(strategy)],
        "highWinrateExpectedWinRate": overall.get("winRate"),
        "highWinrateExpectedTradesPerDay": overall.get("tradesPerDay"),
        "highWinrateBacktestTrades": overall.get("trades"),
        "tradeQualityScoreMin": RULE_MIN_QUALITY_SCORE,
        "tradeQualityGate": RULE_GATE_NAME,
        "targetEventWinRate": RULE_TARGET_WIN_RATE,
        "expectedEventWinRate": overall.get("winRate"),
        "expectedDirectionHitRate": overall.get("winRate"),
        "expectedTestTrades": overall.get("trades"),
        "expectedTradesPerDay": overall.get("tradesPerDay"),
        "backtestSource": backtest.get("source") if backtest else "missing_rule_backtest",
    }


def _production_target(backtest: dict[str, Any] | None, strategy: StrategyDefinition) -> dict[str, Any]:
    if not strategy.uses_trade_policy_gates:
        return {
            "targetWinRate": None,
            "targetTradesPerDay": None,
            "winRate": None,
            "tradesPerDay": None,
            "passed": True,
            "source": "trade_policy_gates_disabled",
        }
    overall = backtest.get("overall", {}) if backtest else {}
    return {
        "targetWinRate": RULE_TARGET_WIN_RATE,
        "targetTradesPerDay": strategy.min_daily_trades,
        "winRate": overall.get("winRate"),
        "tradesPerDay": overall.get("tradesPerDay"),
        "passed": backtest.get("passed") if backtest else None,
        "source": backtest.get("source") if backtest else "missing_rule_backtest",
    }


def _rule_payload(strategy: StrategyDefinition) -> dict[str, Any]:
    conditions = _rule_conditions(strategy)
    return {
        "name": RULE_GATE_NAME,
        "strategyKey": strategy.key,
        "direction": "both",
        "min_confidence": RULE_MIN_CONFIDENCE,
        "min_quality_score": RULE_MIN_QUALITY_SCORE,
        "conditions": conditions,
    }


def _rule_conditions(strategy: StrategyDefinition) -> list[dict[str, Any]]:
    if not strategy.uses_trade_policy_gates:
        return []
    conditions = [
        {"feature": "orderbook_imbalance", "operator": "directional_alignment"},
        {"feature": "1d_4h_1h_30m_trend", "operator": "supports_10m"},
    ]
    if strategy.rule_names is not None:
        conditions.append({"feature": "optimized_rule_subset", "operator": "in", "value": list(strategy.rule_names)})
    if strategy.min_daily_trades is not None:
        conditions.append({"feature": "min_daily_trades", "operator": ">=", "value": strategy.min_daily_trades})
    return conditions


def _load_rule_backtest(duration: str, strategy_key: str) -> dict[str, Any] | None:
    suffix = "" if strategy_key == DEFAULT_STRATEGY_KEY else f"_{strategy_key}"
    path = RULE_ARTIFACT_DIR / f"rule_backtest_{duration}{suffix}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _assert_supported_duration(duration: str) -> None:
    if duration not in SUPPORTED_RULE_DURATIONS:
        raise ValueError(f"rule policy supports only {sorted(SUPPORTED_RULE_DURATIONS)}, got {duration}")
