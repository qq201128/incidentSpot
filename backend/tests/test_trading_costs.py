from __future__ import annotations

import pytest

from app.services.trading_costs import (
    BacktestCostConfig,
    default_backtest_cost_config,
    roundtrip_cost_rate,
    validate_backtest_cost_config,
)


def test_default_backtest_cost_is_explicitly_zero() -> None:
    config = default_backtest_cost_config()

    assert config.fee_rate_per_side == pytest.approx(0.0)
    assert config.slippage_rate_per_side == pytest.approx(0.0)
    assert roundtrip_cost_rate(config) == pytest.approx(0.0)


def test_invalid_backtest_cost_config_raises() -> None:
    config = BacktestCostConfig(fee_rate_per_side=-0.1, slippage_rate_per_side=0.0)

    with pytest.raises(ValueError, match="fee_rate_per_side"):
        validate_backtest_cost_config(config)
