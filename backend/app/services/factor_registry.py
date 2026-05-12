"""
Factor Registry - Central definition of all trading factors.

Each factor has:
- name: unique identifier
- category: grouping for UI display
- description: human-readable explanation
- formula: calculation description
- source_file: where the factor is computed
- timeframes: rule horizons used in UI/backtest (10m/30m/60m/1d); multi-TF rows also list feature bar intervals (e.g. 5m)
- direction: "higher_better", "lower_better", or "neutral"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.services.factor_extensions import ADDITIONAL_FACTORS
from app.services.factor_external_extensions import EXTERNAL_DATA_FACTORS
from app.services.rule_config import SUPPORTED_RULE_DURATIONS

# 与规则 / 因子排名回测周期一致（特征多在 1m 序列上构造，此处表示「适用规则周期」）
_RULE_UI_TIME_ORDER = ("10m", "30m", "60m", "1d")
RULE_FACTOR_TIMEFRAMES: tuple[str, ...] = tuple(
    d for d in _RULE_UI_TIME_ORDER if d in SUPPORTED_RULE_DURATIONS
)


def _timeframes_with_rule_align(*feature_bar_intervals: str) -> tuple[str, ...]:
    """先列规则周期，再追加特征所用 K 线周期（去重）。"""
    ordered: list[str] = []
    seen: set[str] = set()
    for x in (*RULE_FACTOR_TIMEFRAMES, *feature_bar_intervals):
        if x in seen:
            continue
        seen.add(x)
        ordered.append(x)
    return tuple(ordered)


class FactorCategory(str, Enum):
    RETURN = "return"
    VOLATILITY = "volatility"
    MOVING_AVERAGE = "moving_average"
    MOMENTUM = "momentum"
    VOLUME = "volume"
    STRUCTURE = "structure"
    MULTI_TIMEFRAME = "multi_timeframe"
    ORDERBOOK = "orderbook"
    FUNDING = "funding"
    POSITIONING = "positioning"
    TAKER_FLOW = "taker_flow"
    SMC = "smc"
    SENTIMENT = "sentiment"
    STATISTIC = "statistic"
    ONCHAIN = "onchain"
    PERFORMANCE = "performance"


class FactorDirection(str, Enum):
    HIGHER_BETTER = "higher_better"
    LOWER_BETTER = "lower_better"
    NEUTRAL = "neutral"


@dataclass(frozen=True)
class FactorDefinition:
    name: str
    category: FactorCategory
    description: str
    formula: str
    source_file: str = "kline_features.py"
    timeframes: tuple[str, ...] = RULE_FACTOR_TIMEFRAMES
    direction: FactorDirection = FactorDirection.NEUTRAL
    parameters: dict[str, Any] = field(default_factory=dict)


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

# =============================================================================
# 订单簿因子 (Orderbook Factors)
# =============================================================================
ORDERBOOK_FACTORS = (
    FactorDefinition(
        name="imbalance",
        category=FactorCategory.ORDERBOOK,
        description="订单簿不平衡",
        formula="(bid_qty - ask_qty) / total_qty",
        source_file="orderbook_feature_service.py",
        direction=FactorDirection.NEUTRAL,
    ),
    FactorDefinition(
        name="spread_bps",
        category=FactorCategory.ORDERBOOK,
        description="买卖价差（基点）",
        formula="(best_ask - best_bid) / mid * 10000",
        source_file="rule_orderbook_service.py",
        direction=FactorDirection.LOWER_BETTER,
    ),
    FactorDefinition(
        name="microprice_bps",
        category=FactorCategory.ORDERBOOK,
        description="微价格偏差（基点）",
        formula="(microprice - mid) / mid * 10000",
        source_file="orderbook_feature_service.py",
        direction=FactorDirection.NEUTRAL,
    ),
    FactorDefinition(
        name="ofi_ratio",
        category=FactorCategory.ORDERBOOK,
        description="订单流失衡比率",
        formula="ofi / (best_bid_qty + best_ask_qty)",
        source_file="orderbook_feature_service.py",
        direction=FactorDirection.NEUTRAL,
    ),
)

# =============================================================================
# 资金费率因子 (Funding Rate Factors)
# =============================================================================
FUNDING_FACTORS = (
    FactorDefinition(
        name="funding_rate",
        category=FactorCategory.FUNDING,
        description="资金费率",
        formula="funding_rate",
        source_file="enhanced_features.py",
        direction=FactorDirection.NEUTRAL,
    ),
    FactorDefinition(
        name="funding_ma_8",
        category=FactorCategory.FUNDING,
        description="资金费率均值（8周期）",
        formula="funding_rate.rolling(8).mean()",
        source_file="enhanced_features.py",
        direction=FactorDirection.NEUTRAL,
    ),
    FactorDefinition(
        name="funding_z_20",
        category=FactorCategory.FUNDING,
        description="资金费率Z分数（20周期）",
        formula="(funding_rate - mean) / std",
        source_file="enhanced_features.py",
        direction=FactorDirection.NEUTRAL,
    ),
)


def _factor_from_payload(payload: dict[str, Any]) -> FactorDefinition:
    data = dict(payload)
    data["category"] = FactorCategory(str(data["category"]))
    data["direction"] = FactorDirection(str(data.get("direction", FactorDirection.NEUTRAL)))
    return FactorDefinition(**data)


EXTENDED_KLINE_FACTORS = tuple(
    _factor_from_payload(payload) for payload in (*ADDITIONAL_FACTORS, *EXTERNAL_DATA_FACTORS)
)

# =============================================================================
# All Factors Combined
# =============================================================================
ALL_FACTORS: tuple[FactorDefinition, ...] = (
    *RETURN_FACTORS,
    *VOLATILITY_FACTORS,
    *MA_FACTORS,
    *MOMENTUM_FACTORS,
    *VOLUME_FACTORS,
    *STRUCTURE_FACTORS,
    *EXTENDED_KLINE_FACTORS,
    *MULTI_TIMEFRAME_FACTORS,
    *ORDERBOOK_FACTORS,
    *FUNDING_FACTORS,
)


def _factor_index(factors: tuple[FactorDefinition, ...]) -> dict[str, FactorDefinition]:
    indexed: dict[str, FactorDefinition] = {}
    for factor in factors:
        if factor.name in indexed:
            raise ValueError(f"duplicate factor definition: {factor.name}")
        indexed[factor.name] = factor
    return indexed


FACTOR_BY_NAME: dict[str, FactorDefinition] = _factor_index(ALL_FACTORS)


def get_factor(name: str) -> FactorDefinition | None:
    """Get factor definition by name."""
    return FACTOR_BY_NAME.get(name)


def list_factors(category: FactorCategory | None = None) -> list[FactorDefinition]:
    """List all factors, optionally filtered by category."""
    if category is None:
        return list(ALL_FACTORS)
    return [f for f in ALL_FACTORS if f.category == category]


def list_factor_categories() -> list[dict[str, Any]]:
    """List all factor categories with counts."""
    from collections import Counter
    counts = Counter(f.category.value for f in ALL_FACTORS)
    return [
        {
            "key": cat.value,
            "name": _category_display_name(cat),
            "count": counts.get(cat.value, 0),
        }
        for cat in FactorCategory
    ]


def _category_display_name(cat: FactorCategory) -> str:
    """Get display name for category."""
    names = {
        FactorCategory.RETURN: "收益率因子",
        FactorCategory.VOLATILITY: "波动率因子",
        FactorCategory.MOVING_AVERAGE: "均线因子",
        FactorCategory.MOMENTUM: "动量因子",
        FactorCategory.VOLUME: "成交量因子",
        FactorCategory.STRUCTURE: "结构因子",
        FactorCategory.MULTI_TIMEFRAME: "多时间框架因子",
        FactorCategory.ORDERBOOK: "订单簿因子",
        FactorCategory.FUNDING: "资金费率因子",
        FactorCategory.POSITIONING: "持仓因子",
        FactorCategory.TAKER_FLOW: "主动成交因子",
        FactorCategory.SMC: "SMC因子",
        FactorCategory.SENTIMENT: "情绪因子",
        FactorCategory.STATISTIC: "统计因子",
        FactorCategory.ONCHAIN: "链上因子",
        FactorCategory.PERFORMANCE: "绩效因子",
    }
    return names.get(cat, cat.value)


def factor_payload(factor: FactorDefinition) -> dict[str, Any]:
    """Convert factor definition to API response payload."""
    return {
        "name": factor.name,
        "category": factor.category.value,
        "categoryName": _category_display_name(factor.category),
        "displayName": factor.description,
        "description": factor.description,
        "formula": factor.formula,
        "sourceFile": factor.source_file,
        "timeframes": list(factor.timeframes),
        "direction": factor.direction.value,
        "parameters": dict(factor.parameters) if factor.parameters else {},
    }


def list_factor_payloads(category: str | None = None) -> list[dict[str, Any]]:
    """List all factor payloads for API response."""
    cat = FactorCategory(category) if category else None
    return [factor_payload(f) for f in list_factors(cat)]
