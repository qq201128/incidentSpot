from __future__ import annotations

import pytest

from app.services.daily_trade_floor_tree_rule import evaluate_daily_trade_floor_rule
from app.services.rule_backtest_metrics import passed
from app.services.strategy_registry import (
    DAILY_TRADE_FLOOR_MIN_DAILY_TRADES,
    DAILY_TRADE_FLOOR_RULE_NAME,
    DAILY_TRADE_FLOOR_TREE_STRATEGY_KEY,
    strategy_definition,
)


def test_daily_trade_floor_strategy_declares_new_rule_and_trade_floor() -> None:
    strategy = strategy_definition(DAILY_TRADE_FLOOR_TREE_STRATEGY_KEY)

    assert strategy.rule_names == (DAILY_TRADE_FLOOR_RULE_NAME,)
    assert strategy.min_daily_trades == DAILY_TRADE_FLOOR_MIN_DAILY_TRADES
    assert strategy.requires_vegas_confirmation is False


def test_backtest_passed_rejects_low_trade_days() -> None:
    overall = {"trades": 100, "winRate": 0.80}
    low_trade_days = [{"day": "2026-05-01", "trades": 9}]

    assert passed(overall, failed_days=[], low_trade_days=low_trade_days) is False


def test_daily_trade_floor_rule_exposes_missing_features() -> None:
    with pytest.raises(KeyError, match="daily trade floor feature missing"):
        evaluate_daily_trade_floor_rule({})
