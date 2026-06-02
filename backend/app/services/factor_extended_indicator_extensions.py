from __future__ import annotations

from typing import Any

SOURCE_FILE = "kline_extended_indicators.py"

_ROWS: tuple[tuple[str, str, str, str, str], ...] = (
    ("bb_percent_b_20", "volatility", "布林带%B位置（20周期）", "(close - lower_band) / (upper_band - lower_band)", "neutral"),
    ("bb_percent_b_z_20", "volatility", "布林带%B位置Z分数（20周期）", "zscore(bb_percent_b_20, 20)", "neutral"),
    ("dpo_20", "momentum", "去趋势价格振荡器DPO（20周期）", "close.shift(11) - sma(close,20)", "neutral"),
    ("pmo_35_20", "momentum", "DecisionPoint PMO动量线（35/20）", "smooth(10 * smooth(roc1,35),20)", "neutral"),
    ("pmo_signal_10", "momentum", "DecisionPoint PMO信号线（10周期）", "ema(pmo_35_20,10)", "neutral"),
    ("pmo_diff", "momentum", "DecisionPoint PMO动量差", "pmo_35_20 - pmo_signal_10", "neutral"),
    ("stoch_rsi_14", "momentum", "StochRSI位置（14周期）", "stoch(rsi_14,14)", "neutral"),
    ("stoch_rsi_signal_3", "momentum", "StochRSI信号线（3周期）", "sma(stoch_rsi_14,3)", "neutral"),
    ("rsi_14_sma_5", "momentum", "RSI平滑线（5周期）", "sma(rsi_14,5)", "neutral"),
    ("rsi_14_slope_3", "momentum", "RSI斜率（3周期）", "rsi_14.diff(3)", "neutral"),
    ("awesome_osc_5_34", "momentum", "Awesome Oscillator（5/34）", "sma(median,5) - sma(median,34)", "neutral"),
    ("accelerator_osc", "momentum", "Accelerator Oscillator动量加速度", "awesome_osc_5_34 - sma(awesome_osc_5_34,5)", "neutral"),
    ("fisher_10", "momentum", "Fisher Transform价格位置（10周期）", "0.5 * log((1 + x) / (1 - x))", "neutral"),
    ("fisher_signal_1", "momentum", "Fisher Transform前值信号", "fisher_10.shift(1)", "neutral"),
    ("ichimoku_tenkan_9", "structure", "一目均衡转换线偏离（9周期）", "midrange(high,low,9) / close - 1", "neutral"),
    ("ichimoku_kijun_26", "structure", "一目均衡基准线偏离（26周期）", "midrange(high,low,26) / close - 1", "neutral"),
    ("ichimoku_tenkan_kijun_spread", "structure", "一目转换线与基准线差", "tenkan / kijun - 1", "neutral"),
    ("ichimoku_cloud_pos", "structure", "一目云层内位置", "(close - cloud_bottom) / (cloud_top - cloud_bottom)", "neutral"),
    ("ichimoku_cloud_width", "volatility", "一目云层宽度", "(cloud_top - cloud_bottom) / close", "neutral"),
    ("ichimoku_chikou_mom_26", "momentum", "一目迟行跨度动量（26周期）", "close.pct_change(26)", "neutral"),
    ("donchian_width_20", "volatility", "Donchian通道宽度（20周期）", "(rolling_high_20 - rolling_low_20) / close", "neutral"),
    ("donchian_breakout_20", "structure", "Donchian向上突破（20周期）", "close > rolling_high_20.shift(1)", "neutral"),
    ("donchian_breakdown_20", "structure", "Donchian向下突破（20周期）", "close < rolling_low_20.shift(1)", "neutral"),
    ("donchian_width_55", "volatility", "Donchian通道宽度（55周期）", "(rolling_high_55 - rolling_low_55) / close", "neutral"),
    ("qstick_10", "structure", "QStick实体均值（10周期）", "sma(close - open,10) / close", "neutral"),
    ("qstick_20", "structure", "QStick实体均值（20周期）", "sma(close - open,20) / close", "neutral"),
    ("qstick_spread_10_20", "structure", "QStick快慢差（10/20）", "qstick_10 - qstick_20", "neutral"),
    ("elder_bull_power_13", "structure", "Elder Bull Power（13周期）", "high - ema(close,13)", "neutral"),
    ("elder_bear_power_13", "structure", "Elder Bear Power（13周期）", "low - ema(close,13)", "neutral"),
    ("elder_ray_spread_13", "volatility", "Elder Ray多空力度差", "elder_bull_power_13 - elder_bear_power_13", "neutral"),
    ("sma_50_200_spread", "moving_average", "50/200均线差", "sma(close,50) / sma(close,200) - 1", "neutral"),
    ("close_sma_50_ratio", "moving_average", "收盘价相对50周期均线偏离", "close / sma(close,50) - 1", "neutral"),
    ("nvi_slope_20", "volume", "负成交量指数斜率（20周期）", "nvi.pct_change(20)", "neutral"),
    ("pvi_slope_20", "volume", "正成交量指数斜率（20周期）", "pvi.pct_change(20)", "neutral"),
    ("nvi_pvi_spread_20", "volume", "NVI/PVI斜率差（20周期）", "nvi_slope_20 - pvi_slope_20", "neutral"),
    ("klinger_osc_34_55", "volume", "Klinger成交量振荡器（34/55）", "ema(volume_force,34) - ema(volume_force,55)", "neutral"),
    ("klinger_signal_13", "volume", "Klinger信号线（13周期）", "ema(klinger_osc_34_55,13)", "neutral"),
    ("klinger_diff", "volume", "Klinger振荡器差", "klinger_osc_34_55 - klinger_signal_13", "neutral"),
    ("relative_vigor_10", "momentum", "相对活力指数RVI（10周期）", "sma(close - open,10) / sma(high - low,10)", "neutral"),
    ("relative_vigor_signal_4", "momentum", "RVI信号线（4周期）", "sma(relative_vigor_10,4)", "neutral"),
    ("force_index_z_20", "volume", "Force Index Z分数（20周期）", "zscore(force_index_13,20)", "neutral"),
    ("emv_z_20", "volume", "EMV Z分数（20周期）", "zscore(emv_14,20)", "neutral"),
    ("mfi_7", "volume", "资金流量指标MFI（7周期）", "MFI(7)", "neutral"),
    ("mfi_21", "volume", "资金流量指标MFI（21周期）", "MFI(21)", "neutral"),
    ("mfi_spread_7_21", "volume", "MFI快慢差（7/21）", "mfi_7 - mfi_21", "neutral"),
)


def _factor(row: tuple[str, str, str, str, str]) -> dict[str, Any]:
    name, category, description, formula, direction = row
    return {
        "name": name,
        "category": category,
        "description": description,
        "formula": formula,
        "source_file": SOURCE_FILE,
        "direction": direction,
    }


EXTENDED_INDICATOR_FACTORS: tuple[dict[str, Any], ...] = tuple(_factor(row) for row in _ROWS)
