from __future__ import annotations

from app.services.factor_registry_core import FactorCategory, FactorDefinition, FactorDirection


# =============================================================================
# 均线因子 (Moving Average Factors)
# =============================================================================
MA_FACTORS = (
    FactorDefinition(
        name="ma_ratio_3",
        category=FactorCategory.MOVING_AVERAGE,
        description="3周期均线偏离",
        formula="close / sma(3) - 1",
        direction=FactorDirection.NEUTRAL,
    ),
    FactorDefinition(
        name="ma_ratio_5",
        category=FactorCategory.MOVING_AVERAGE,
        description="5周期均线偏离",
        formula="close / sma(5) - 1",
        direction=FactorDirection.NEUTRAL,
    ),
    FactorDefinition(
        name="ma_ratio_10",
        category=FactorCategory.MOVING_AVERAGE,
        description="10周期均线偏离",
        formula="close / sma(10) - 1",
        direction=FactorDirection.NEUTRAL,
    ),
    FactorDefinition(
        name="ma_ratio_20",
        category=FactorCategory.MOVING_AVERAGE,
        description="20周期均线偏离",
        formula="close / sma(20) - 1",
        direction=FactorDirection.NEUTRAL,
    ),
    FactorDefinition(
        name="ma_ratio_60",
        category=FactorCategory.MOVING_AVERAGE,
        description="60周期均线偏离",
        formula="close / sma(60) - 1",
        direction=FactorDirection.NEUTRAL,
    ),
    FactorDefinition(
        name="ma_ratio_120",
        category=FactorCategory.MOVING_AVERAGE,
        description="120周期均线偏离",
        formula="close / sma(120) - 1",
        direction=FactorDirection.NEUTRAL,
    ),
    FactorDefinition(
        name="ma_ratio_240",
        category=FactorCategory.MOVING_AVERAGE,
        description="240周期均线偏离",
        formula="close / sma(240) - 1",
        direction=FactorDirection.NEUTRAL,
    ),
    FactorDefinition(
        name="ema_cross",
        category=FactorCategory.MOVING_AVERAGE,
        description="EMA交叉（12/26）",
        formula="(ema(12) - ema(26)) / close",
        direction=FactorDirection.NEUTRAL,
    ),
)

# =============================================================================
# 动量因子 (Momentum Factors)
# =============================================================================
MOMENTUM_FACTORS = (
    FactorDefinition(
        name="rsi_14",
        category=FactorCategory.MOMENTUM,
        description="RSI（14周期）",
        formula="100 - 100 / (1 + avg_gain / avg_loss)",
        direction=FactorDirection.NEUTRAL,
        parameters={"period": 14},
    ),
    FactorDefinition(
        name="rsi_14_chg_3",
        category=FactorCategory.MOMENTUM,
        description="RSI变化（3周期）",
        formula="rsi_14.diff(3)",
        direction=FactorDirection.NEUTRAL,
    ),
    FactorDefinition(
        name="macd",
        category=FactorCategory.MOMENTUM,
        description="MACD线",
        formula="ema(12) - ema(26)",
        direction=FactorDirection.NEUTRAL,
    ),
    FactorDefinition(
        name="macd_signal",
        category=FactorCategory.MOMENTUM,
        description="MACD信号线",
        formula="ema(macd, 9)",
        direction=FactorDirection.NEUTRAL,
    ),
    FactorDefinition(
        name="macd_hist",
        category=FactorCategory.MOMENTUM,
        description="MACD直方图",
        formula="macd - macd_signal",
        direction=FactorDirection.NEUTRAL,
    ),
    FactorDefinition(
        name="macd_hist_chg_3",
        category=FactorCategory.MOMENTUM,
        description="MACD直方图变化（3周期）",
        formula="macd_hist.diff(3)",
        direction=FactorDirection.NEUTRAL,
    ),
    FactorDefinition(
        name="mom_10_norm",
        category=FactorCategory.MOMENTUM,
        description="标准化动量（10周期）",
        formula="(close - close.shift(10)) / close",
        direction=FactorDirection.NEUTRAL,
    ),
    FactorDefinition(
        name="mom_20_norm",
        category=FactorCategory.MOMENTUM,
        description="标准化动量（20周期）",
        formula="(close - close.shift(20)) / close",
        direction=FactorDirection.NEUTRAL,
    ),
    FactorDefinition(
        name="efficiency_ratio_10",
        category=FactorCategory.MOMENTUM,
        description="效率比率（10周期）",
        formula="abs(close - close.shift(10)) / sum(abs(close.diff()), 10)",
        direction=FactorDirection.HIGHER_BETTER,
    ),
    FactorDefinition(
        name="efficiency_ratio_20",
        category=FactorCategory.MOMENTUM,
        description="效率比率（20周期）",
        formula="abs(close - close.shift(20)) / sum(abs(close.diff()), 20)",
        direction=FactorDirection.HIGHER_BETTER,
    ),
)

# =============================================================================
# 成交量因子 (Volume Factors)
# =============================================================================
VOLUME_FACTORS = (
    FactorDefinition(
        name="vol_chg",
        category=FactorCategory.VOLUME,
        description="成交量变化率",
        formula="volume.pct_change(1)",
        direction=FactorDirection.NEUTRAL,
    ),
    FactorDefinition(
        name="vol_ma_20",
        category=FactorCategory.VOLUME,
        description="20周期成交量均值",
        formula="volume.rolling(20).mean()",
        direction=FactorDirection.NEUTRAL,
    ),
    FactorDefinition(
        name="vol_ma_60",
        category=FactorCategory.VOLUME,
        description="60周期成交量均值",
        formula="volume.rolling(60).mean()",
        direction=FactorDirection.NEUTRAL,
    ),
    FactorDefinition(
        name="vol_ma_120",
        category=FactorCategory.VOLUME,
        description="120周期成交量均值",
        formula="volume.rolling(120).mean()",
        direction=FactorDirection.NEUTRAL,
    ),
    FactorDefinition(
        name="vol_ma_240",
        category=FactorCategory.VOLUME,
        description="240周期成交量均值",
        formula="volume.rolling(240).mean()",
        direction=FactorDirection.NEUTRAL,
    ),
    FactorDefinition(
        name="vol_ratio_20",
        category=FactorCategory.VOLUME,
        description="20周期成交量比率",
        formula="volume / vol_ma_20",
        direction=FactorDirection.NEUTRAL,
    ),
    FactorDefinition(
        name="vol_ratio_60",
        category=FactorCategory.VOLUME,
        description="60周期成交量比率",
        formula="volume / vol_ma_60",
        direction=FactorDirection.NEUTRAL,
    ),
    FactorDefinition(
        name="vol_ratio_120",
        category=FactorCategory.VOLUME,
        description="120周期成交量比率",
        formula="volume / vol_ma_120",
        direction=FactorDirection.NEUTRAL,
    ),
    FactorDefinition(
        name="vol_ratio_240",
        category=FactorCategory.VOLUME,
        description="240周期成交量比率",
        formula="volume / vol_ma_240",
        direction=FactorDirection.NEUTRAL,
    ),
    FactorDefinition(
        name="vol_median_ratio_10",
        category=FactorCategory.VOLUME,
        description="10周期成交量中位数比率",
        formula="volume / volume.rolling(10).median()",
        direction=FactorDirection.NEUTRAL,
    ),
    FactorDefinition(
        name="vol_median_ratio_20",
        category=FactorCategory.VOLUME,
        description="20周期成交量中位数比率",
        formula="volume / volume.rolling(20).median()",
        direction=FactorDirection.NEUTRAL,
    ),
    FactorDefinition(
        name="vol_median_ratio_60",
        category=FactorCategory.VOLUME,
        description="60周期成交量中位数比率",
        formula="volume / volume.rolling(60).median()",
        direction=FactorDirection.NEUTRAL,
    ),
)

