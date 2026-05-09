from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"
DEFAULT_GATE_NAME = "disabled_missing_high_winrate_gate"
EPSILON = 1e-12


def load_high_winrate_gate(duration: str) -> dict[str, Any]:
    path = MODEL_DIR / f"high_winrate_gate_{duration}.json"
    if not path.exists():
        return {
            "enabled": False,
            "gate_name": DEFAULT_GATE_NAME,
            "min_confidence": None,
            "rules": [],
            "backtest": {},
        }
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def high_winrate_confidence_threshold(duration: str) -> float | None:
    config = load_high_winrate_gate(duration)
    if not bool(config.get("enabled")):
        return None
    confidences = [float(rule["min_confidence"]) for rule in _rules(config)]
    return min(confidences) if confidences else None


def high_winrate_gate_policy_payload(duration: str) -> dict[str, Any]:
    config = load_high_winrate_gate(duration)
    backtest = config.get("backtest", {})
    return {
        "highWinrateGate": config.get("gate_name"),
        "highWinrateGateEnabled": bool(config.get("enabled")),
        "highWinrateMinConfidence": high_winrate_confidence_threshold(duration),
        "highWinrateRules": _rules(config),
        "highWinrateExpectedWinRate": backtest.get("win_rate"),
        "highWinrateExpectedTradesPerDay": backtest.get("trades_per_day"),
        "highWinrateBacktestTrades": backtest.get("test_trades"),
    }


def evaluate_high_winrate_gate(
    feature_row: dict[str, Any],
    direction: str,
    confidence: float,
    duration: str,
) -> dict[str, Any]:
    config = load_high_winrate_gate(duration)
    if not bool(config.get("enabled")):
        return _gate_result(config, None, False, None, "gate_disabled")
    for rule in _rules(config):
        if _passes_rule(feature_row, direction, float(confidence), rule):
            return _gate_result(config, rule, True, _first_value(feature_row, direction, rule), "passed")
    rule = _first_direction_rule(config, direction)
    return _gate_result(config, rule, False, _first_value(feature_row, direction, rule), "rule_not_met")


def _passes_rule(
    feature_row: dict[str, Any],
    direction: str,
    confidence: float,
    rule: dict[str, Any],
) -> bool:
    if rule.get("direction") and direction != rule.get("direction"):
        return False
    if confidence + EPSILON < float(rule["min_confidence"]):
        return False
    return all(_passes_condition(feature_row, direction, item) for item in rule.get("conditions", []))


def _passes_condition(feature_row: dict[str, Any], direction: str, condition: dict[str, Any]) -> bool:
    value = _condition_value(feature_row, direction, condition)
    threshold = float(condition["value"])
    if condition.get("operator") == "<=":
        return value <= threshold + EPSILON
    return value + EPSILON >= threshold


def _condition_value(feature_row: dict[str, Any], direction: str, condition: dict[str, Any]) -> float:
    feature = condition["feature"]
    if feature not in feature_row:
        raise KeyError(f"high winrate gate feature missing: {feature}")
    raw_value = float(feature_row[feature])
    if condition.get("transform") != "signed":
        return raw_value
    side = 1.0 if direction == "up" else -1.0
    return raw_value * side


def _rules(config: dict[str, Any]) -> list[dict[str, Any]]:
    rules = config.get("rules")
    if isinstance(rules, list) and rules:
        return rules
    if config.get("feature") is None:
        return []
    return [_legacy_rule(config)]


def _legacy_rule(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": config.get("gate_name", DEFAULT_GATE_NAME),
        "direction": config.get("direction"),
        "min_confidence": config.get("min_confidence"),
        "conditions": [
            {
                "feature": config.get("feature"),
                "transform": config.get("transform"),
                "operator": ">=",
                "value": config.get("feature_min"),
            }
        ],
    }


def _first_direction_rule(config: dict[str, Any], direction: str) -> dict[str, Any] | None:
    for rule in _rules(config):
        if rule.get("direction") == direction:
            return rule
    return _rules(config)[0] if _rules(config) else None


def _first_value(feature_row: dict[str, Any], direction: str, rule: dict[str, Any] | None) -> float | None:
    conditions = (rule or {}).get("conditions", [])
    if not conditions:
        return None
    return _condition_value(feature_row, direction, conditions[0])


def _gate_result(
    config: dict[str, Any],
    rule: dict[str, Any] | None,
    passed: bool,
    value: float | None,
    reason: str,
) -> dict[str, Any]:
    first_condition = ((rule or {}).get("conditions") or [{}])[0]
    return {
        "high_winrate_gate": config.get("gate_name", DEFAULT_GATE_NAME),
        "high_winrate_rule": (rule or {}).get("name"),
        "high_winrate_gate_passed": passed,
        "high_winrate_gate_value": None if value is None else round(float(value), 8),
        "high_winrate_gate_min": first_condition.get("value"),
        "high_winrate_gate_reason": reason,
    }
