from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BacktestCostConfig:
    fee_rate_per_side: float
    slippage_rate_per_side: float
    min_trade_gap_minutes: int = 0

    @property
    def roundtrip_cost_rate(self) -> float:
        return 2.0 * (self.fee_rate_per_side + self.slippage_rate_per_side)


DEFAULT_BACKTEST_COST_CONFIG = BacktestCostConfig(
    fee_rate_per_side=0.0004,
    slippage_rate_per_side=0.0001,
    min_trade_gap_minutes=0,
)
ROUNDTRIP_COST_RATE = DEFAULT_BACKTEST_COST_CONFIG.roundtrip_cost_rate


def default_backtest_cost_config() -> BacktestCostConfig:
    return DEFAULT_BACKTEST_COST_CONFIG


def validate_backtest_cost_config(config: BacktestCostConfig) -> BacktestCostConfig:
    if config.fee_rate_per_side < 0:
        raise ValueError("fee_rate_per_side must be >= 0")
    if config.slippage_rate_per_side < 0:
        raise ValueError("slippage_rate_per_side must be >= 0")
    if config.min_trade_gap_minutes < 0:
        raise ValueError("min_trade_gap_minutes must be >= 0")
    return config


def roundtrip_cost_rate(config: BacktestCostConfig | None = None) -> float:
    selected = validate_backtest_cost_config(config or DEFAULT_BACKTEST_COST_CONFIG)
    return selected.roundtrip_cost_rate
