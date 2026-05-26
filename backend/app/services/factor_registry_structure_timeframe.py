from __future__ import annotations

from app.services.factor_registry_core import (
    FactorCategory,
    FactorDefinition,
    FactorDirection,
    _timeframes_with_rule_align,
)


# =============================================================================
# 结构因子 (Structure Factors)
# =============================================================================
STRUCTURE_FACTORS = (
    FactorDefinition(
        name="upper_shadow",
        category=FactorCategory.STRUCTURE,
        description="上影线",
        formula="(high - max(close, open)) / close",
        direction=FactorDirection.NEUTRAL,
    ),
    FactorDefinition(
        name="lower_shadow",
        category=FactorCategory.STRUCTURE,
        description="下影线",
        formula="(min(close, open) - low) / close",
        direction=FactorDirection.NEUTRAL,
    ),
    FactorDefinition(
        name="wick_imbalance",
        category=FactorCategory.STRUCTURE,
        description="影线不平衡",
        formula="lower_shadow - upper_shadow",
        direction=FactorDirection.NEUTRAL,
    ),
    FactorDefinition(
        name="donchian_pos_20",
        category=FactorCategory.STRUCTURE,
        description="Donchian位置（20周期）",
        formula="(close - min(low, 20)) / (max(high, 20) - min(low, 20))",
        direction=FactorDirection.NEUTRAL,
    ),
    FactorDefinition(
        name="donchian_pos_60",
        category=FactorCategory.STRUCTURE,
        description="Donchian位置（60周期）",
        formula="(close - min(low, 60)) / (max(high, 60) - min(low, 60))",
        direction=FactorDirection.NEUTRAL,
    ),
    FactorDefinition(
        name="vwap_dev_20",
        category=FactorCategory.STRUCTURE,
        description="VWAP偏离（20周期）",
        formula="close / vwap(20) - 1",
        direction=FactorDirection.NEUTRAL,
    ),
    FactorDefinition(
        name="vwap_dev_60",
        category=FactorCategory.STRUCTURE,
        description="VWAP偏离（60周期）",
        formula="close / vwap(60) - 1",
        direction=FactorDirection.NEUTRAL,
    ),
)

# =============================================================================
# 多时间框架因子 (Multi-Timeframe Factors)
# =============================================================================
MULTI_TIMEFRAME_FACTORS = (
    # 5分钟时间框架
    FactorDefinition(
        name="tf_5m_ret_5",
        category=FactorCategory.MULTI_TIMEFRAME,
        description="5分钟-5周期收益率",
        formula="close_5m.pct_change(5)",
        source_file="enhanced_timeframes.py",
        timeframes=_timeframes_with_rule_align("5m"),
        direction=FactorDirection.NEUTRAL,
    ),
    FactorDefinition(
        name="tf_5m_ma_ratio_12",
        category=FactorCategory.MULTI_TIMEFRAME,
        description="5分钟-12周期均线偏离",
        formula="close_5m / sma(12) - 1",
        source_file="enhanced_timeframes.py",
        timeframes=_timeframes_with_rule_align("5m"),
        direction=FactorDirection.NEUTRAL,
    ),
    FactorDefinition(
        name="tf_5m_ret_vol_12",
        category=FactorCategory.MULTI_TIMEFRAME,
        description="5分钟-12周期波动率",
        formula="ret_5m.rolling(12).std()",
        source_file="enhanced_timeframes.py",
        timeframes=_timeframes_with_rule_align("5m"),
        direction=FactorDirection.NEUTRAL,
    ),
    FactorDefinition(
        name="tf_5m_intrabar_pos",
        category=FactorCategory.MULTI_TIMEFRAME,
        description="5分钟-K线内位置",
        formula="(close - low) / (high - low)",
        source_file="enhanced_timeframes.py",
        timeframes=_timeframes_with_rule_align("5m"),
        direction=FactorDirection.NEUTRAL,
    ),
    # 15分钟时间框架
    FactorDefinition(
        name="tf_15m_ret_4",
        category=FactorCategory.MULTI_TIMEFRAME,
        description="15分钟-4周期收益率",
        formula="close_15m.pct_change(4)",
        source_file="enhanced_timeframes.py",
        timeframes=_timeframes_with_rule_align("15m"),
        direction=FactorDirection.NEUTRAL,
    ),
    FactorDefinition(
        name="tf_15m_ma_ratio_8",
        category=FactorCategory.MULTI_TIMEFRAME,
        description="15分钟-8周期均线偏离",
        formula="close_15m / sma(8) - 1",
        source_file="enhanced_timeframes.py",
        timeframes=_timeframes_with_rule_align("15m"),
        direction=FactorDirection.NEUTRAL,
    ),
    FactorDefinition(
        name="tf_15m_ret_vol_8",
        category=FactorCategory.MULTI_TIMEFRAME,
        description="15分钟-8周期波动率",
        formula="ret_15m.rolling(8).std()",
        source_file="enhanced_timeframes.py",
        timeframes=_timeframes_with_rule_align("15m"),
        direction=FactorDirection.NEUTRAL,
    ),
    # 1小时时间框架
    FactorDefinition(
        name="tf_1h_ret_4",
        category=FactorCategory.MULTI_TIMEFRAME,
        description="1小时-4周期收益率",
        formula="close_1h.pct_change(4)",
        source_file="enhanced_timeframes.py",
        timeframes=_timeframes_with_rule_align("1h"),
        direction=FactorDirection.NEUTRAL,
    ),
    FactorDefinition(
        name="tf_1h_ret_vol_4",
        category=FactorCategory.MULTI_TIMEFRAME,
        description="1小时-4周期波动率",
        formula="ret_1h.rolling(4).std()",
        source_file="enhanced_timeframes.py",
        timeframes=_timeframes_with_rule_align("1h"),
        direction=FactorDirection.NEUTRAL,
    ),
    FactorDefinition(
        name="tf_1h_ret_vol_12",
        category=FactorCategory.MULTI_TIMEFRAME,
        description="1小时-12周期波动率",
        formula="ret_1h.rolling(12).std()",
        source_file="enhanced_timeframes.py",
        timeframes=_timeframes_with_rule_align("1h"),
        direction=FactorDirection.NEUTRAL,
    ),
    # 4小时时间框架
    FactorDefinition(
        name="tf_4h_ret_vol_3",
        category=FactorCategory.MULTI_TIMEFRAME,
        description="4小时-3周期波动率",
        formula="ret_4h.rolling(3).std()",
        source_file="enhanced_timeframes.py",
        timeframes=_timeframes_with_rule_align("4h"),
        direction=FactorDirection.NEUTRAL,
    ),
    FactorDefinition(
        name="tf_4h_intrabar_pos",
        category=FactorCategory.MULTI_TIMEFRAME,
        description="4小时-K线内位置",
        formula="(close - low_4h) / (high_4h - low_4h)",
        source_file="enhanced_timeframes.py",
        timeframes=_timeframes_with_rule_align("4h"),
        direction=FactorDirection.NEUTRAL,
    ),
    # 日线时间框架
    FactorDefinition(
        name="tf_1d_intrabar_pos",
        category=FactorCategory.MULTI_TIMEFRAME,
        description="日线-K线内位置",
        formula="(close - low_1d) / (high_1d - low_1d)",
        source_file="enhanced_timeframes.py",
        timeframes=_timeframes_with_rule_align("1d"),
        direction=FactorDirection.NEUTRAL,
    ),
    FactorDefinition(
        name="tf_1d_volume_share",
        category=FactorCategory.MULTI_TIMEFRAME,
        description="日线-成交量占比",
        formula="volume / daily_volume",
        source_file="enhanced_timeframes.py",
        timeframes=_timeframes_with_rule_align("1d"),
        direction=FactorDirection.NEUTRAL,
    ),
)

