from __future__ import annotations

from app.services.factor_registry_core import FactorCategory, FactorDefinition, FactorDirection


# =============================================================================
# 收益率因子 (Return Factors)
# =============================================================================
RETURN_FACTORS = (
    FactorDefinition(
        name="ret_1",
        category=FactorCategory.RETURN,
        description="单周期收益率",
        formula="close.pct_change(1)",
        direction=FactorDirection.NEUTRAL,
    ),
    FactorDefinition(
        name="ret_3",
        category=FactorCategory.RETURN,
        description="3周期收益率",
        formula="close.pct_change(3)",
        direction=FactorDirection.NEUTRAL,
    ),
    FactorDefinition(
        name="ret_5",
        category=FactorCategory.RETURN,
        description="5周期收益率",
        formula="close.pct_change(5)",
        direction=FactorDirection.NEUTRAL,
    ),
    FactorDefinition(
        name="ret_10",
        category=FactorCategory.RETURN,
        description="10周期收益率",
        formula="close.pct_change(10)",
        direction=FactorDirection.NEUTRAL,
    ),
    FactorDefinition(
        name="ret_20",
        category=FactorCategory.RETURN,
        description="20周期收益率",
        formula="close.pct_change(20)",
        direction=FactorDirection.NEUTRAL,
    ),
    FactorDefinition(
        name="ret_60",
        category=FactorCategory.RETURN,
        description="60周期收益率",
        formula="close.pct_change(60)",
        direction=FactorDirection.NEUTRAL,
    ),
    FactorDefinition(
        name="ret_120",
        category=FactorCategory.RETURN,
        description="120周期收益率",
        formula="close.pct_change(120)",
        direction=FactorDirection.NEUTRAL,
    ),
    FactorDefinition(
        name="ret_240",
        category=FactorCategory.RETURN,
        description="240周期收益率",
        formula="close.pct_change(240)",
        direction=FactorDirection.NEUTRAL,
    ),
    FactorDefinition(
        name="hl_range",
        category=FactorCategory.RETURN,
        description="高低价范围",
        formula="(high - low) / close",
        direction=FactorDirection.NEUTRAL,
    ),
    FactorDefinition(
        name="oc_body",
        category=FactorCategory.RETURN,
        description="开收价实体",
        formula="(close - open) / close",
        direction=FactorDirection.NEUTRAL,
    ),
    FactorDefinition(
        name="ret_10_60",
        category=FactorCategory.RETURN,
        description="收益率加速度（短-长）",
        formula="ret_10 - ret_60",
        direction=FactorDirection.NEUTRAL,
    ),
)

# =============================================================================
# 波动率因子 (Volatility Factors)
# =============================================================================
VOLATILITY_FACTORS = (
    FactorDefinition(
        name="vol_std_3",
        category=FactorCategory.VOLATILITY,
        description="3周期收益率标准差",
        formula="ret_1.rolling(3).std()",
        direction=FactorDirection.NEUTRAL,
    ),
    FactorDefinition(
        name="vol_std_5",
        category=FactorCategory.VOLATILITY,
        description="5周期收益率标准差",
        formula="ret_1.rolling(5).std()",
        direction=FactorDirection.NEUTRAL,
    ),
    FactorDefinition(
        name="vol_std_10",
        category=FactorCategory.VOLATILITY,
        description="10周期收益率标准差",
        formula="ret_1.rolling(10).std()",
        direction=FactorDirection.NEUTRAL,
    ),
    FactorDefinition(
        name="vol_std_20",
        category=FactorCategory.VOLATILITY,
        description="20周期收益率标准差",
        formula="ret_1.rolling(20).std()",
        direction=FactorDirection.NEUTRAL,
    ),
    FactorDefinition(
        name="vol_std_60",
        category=FactorCategory.VOLATILITY,
        description="60周期收益率标准差",
        formula="ret_1.rolling(60).std()",
        direction=FactorDirection.NEUTRAL,
    ),
    FactorDefinition(
        name="vol_std_120",
        category=FactorCategory.VOLATILITY,
        description="120周期收益率标准差",
        formula="ret_1.rolling(120).std()",
        direction=FactorDirection.NEUTRAL,
    ),
    FactorDefinition(
        name="vol_std_240",
        category=FactorCategory.VOLATILITY,
        description="240周期收益率标准差",
        formula="ret_1.rolling(240).std()",
        direction=FactorDirection.NEUTRAL,
    ),
    FactorDefinition(
        name="atr_14",
        category=FactorCategory.VOLATILITY,
        description="14周期ATR（标准化）",
        formula="ATR(14) / close",
        direction=FactorDirection.NEUTRAL,
        parameters={"period": 14},
    ),
    FactorDefinition(
        name="atr_ratio",
        category=FactorCategory.VOLATILITY,
        description="ATR比率",
        formula="atr_14 / atr_14.rolling(20).mean()",
        direction=FactorDirection.NEUTRAL,
    ),
    FactorDefinition(
        name="bb_width_20",
        category=FactorCategory.VOLATILITY,
        description="布林带宽度（20周期）",
        formula="4 * std(20) / sma(20)",
        direction=FactorDirection.NEUTRAL,
        parameters={"period": 20},
    ),
    FactorDefinition(
        name="bb_z_20",
        category=FactorCategory.VOLATILITY,
        description="布林带Z分数（20周期）",
        formula="(close - sma(20)) / std(20)",
        direction=FactorDirection.NEUTRAL,
        parameters={"period": 20},
    ),
    FactorDefinition(
        name="adx_14",
        category=FactorCategory.VOLATILITY,
        description="ADX趋势强度（14周期）",
        formula="ADX(14)",
        direction=FactorDirection.HIGHER_BETTER,
        parameters={"period": 14},
    ),
    FactorDefinition(
        name="chop_14",
        category=FactorCategory.VOLATILITY,
        description="Choppiness指数（14周期）",
        formula="100 * log10(sum(ATR,14) / range) / log10(14)",
        direction=FactorDirection.NEUTRAL,
        parameters={"period": 14},
    ),
)

