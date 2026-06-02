from __future__ import annotations

from app.services.kline_web_factor_specs import WEB_FACTOR_COLUMNS

BASE_WINDOWS = (3, 5, 10, 20, 60, 120, 240)
VOLUME_WINDOWS = (20, 60, 120, 240)
VOLUME_MEDIAN_WINDOWS = (10, 20, 60)
BOLLINGER_WINDOW = 20
ADX_PERIOD = 14
CHOP_PERIOD = 14
RSI_FAST_PERIOD = 7
RSI_SLOW_PERIOD = 21
ROC_PERIOD = 12
STOCH_PERIOD = 14
STOCH_SIGNAL_PERIOD = 3
CCI_PERIOD = 20
MFI_PERIOD = 14
CMF_PERIOD = 20
ZSCORE_WINDOW = 20
VOL_OF_VOL_WINDOW = 20
SKEW_KURT_WINDOW = 20
EPSILON = 1e-12

RETURN_FEATURE_COLUMNS = (
    "ret_1", "log_ret_1", "hl_range", "oc_body", "ret_3", "ret_5",
    "ret_10", "ret_20", "ret_60", "ret_120", "ret_240", "ret_10_60",
    "ret_1_z_20",
)
VOLATILITY_FEATURE_COLUMNS = (
    "vol_std_3", "vol_std_5", "vol_std_10", "vol_std_20", "vol_std_60",
    "vol_std_120", "vol_std_240", "downside_vol_20", "downside_vol_60",
    "upside_vol_20", "upside_vol_60", "vol_of_vol_20", "realized_skew_20",
    "realized_kurt_20", "atr_14", "atr_ratio", "bb_width_20", "bb_z_20",
    "adx_14", "chop_14", "range_ma_20", "range_z_20",
)
MA_FEATURE_COLUMNS = (
    "ma_ratio_3", "ma_ratio_5", "ma_ratio_10", "ma_ratio_20",
    "ma_ratio_60", "ma_ratio_120", "ma_ratio_240", "ema_ratio_12",
    "ema_ratio_26", "ema_cross", "sma_slope_20", "sma_slope_60",
)
MOMENTUM_FEATURE_COLUMNS = (
    "rsi_7", "rsi_14", "rsi_21", "rsi_14_chg_3", "macd", "macd_signal",
    "macd_hist", "macd_hist_chg_3", "mom_10_norm", "mom_20_norm", "roc_12",
    "ppo_12_26", "stochastic_k_14", "stochastic_d_3", "williams_r_14",
    "cci_20", "efficiency_ratio_10", "efficiency_ratio_20",
)
VOLUME_FEATURE_COLUMNS = (
    "vol_chg", "vol_ma_20", "vol_ma_60", "vol_ma_120", "vol_ma_240",
    "vol_ratio_20", "vol_ratio_60", "vol_ratio_120", "vol_ratio_240",
    "vol_median_ratio_10", "vol_median_ratio_20", "vol_median_ratio_60",
    "volume_z_20", "dollar_volume_ma_20", "mfi_14", "cmf_20", "obv_slope_20",
)
STRUCTURE_FEATURE_COLUMNS = (
    "upper_shadow", "lower_shadow", "wick_imbalance", "body_to_range",
    "gap_1", "donchian_pos_20", "donchian_pos_60", "close_to_high_20",
    "close_to_low_20", "vwap_dev_20", "vwap_dev_60",
)
SMC_FEATURE_COLUMNS = (
    "fvg_up_3", "fvg_down_3", "liquidity_sweep_high_20",
    "liquidity_sweep_low_20", "breakout_high_20", "breakdown_low_20",
)
STATISTIC_FEATURE_COLUMNS = (
    "ret_autocorr_20", "price_volume_corr_20", "vol_ret_corr_20",
)
PERFORMANCE_FEATURE_COLUMNS = (
    "rolling_sharpe_60", "win_rate_60", "profit_factor_60",
)
TECHNICAL_INDICATOR_FEATURE_COLUMNS = (
    "aroon_up_25", "aroon_down_25", "aroon_osc_25", "dmi_spread_14",
    "keltner_width_20", "keltner_pos_20", "trix_15", "tsi_25_13",
    "ultimate_osc_7_14_28", "vortex_pos_14", "vortex_neg_14",
    "vortex_spread_14", "mass_index_25", "ulcer_index_14", "cmo_14",
    "coppock_10_14_11", "kst_10_15_20_30", "kst_signal_9", "kst_diff",
    "force_index_13", "emv_14", "chaikin_osc_3_10", "adl_slope_20",
    "pvt_slope_20", "pvo_12_26", "vwma_ratio_20", "bop_14",
    "bb_percent_b_20", "bb_percent_b_z_20", "dpo_20", "pmo_35_20",
    "pmo_signal_10", "pmo_diff", "stoch_rsi_14", "stoch_rsi_signal_3",
    "rsi_14_sma_5", "rsi_14_slope_3", "awesome_osc_5_34",
    "accelerator_osc", "fisher_10", "fisher_signal_1", "ichimoku_tenkan_9",
    "ichimoku_kijun_26", "ichimoku_tenkan_kijun_spread",
    "ichimoku_cloud_pos", "ichimoku_cloud_width", "ichimoku_chikou_mom_26",
    "donchian_width_20", "donchian_breakout_20", "donchian_breakdown_20",
    "donchian_width_55", "qstick_10", "qstick_20", "qstick_spread_10_20",
    "elder_bull_power_13", "elder_bear_power_13", "elder_ray_spread_13",
    "sma_50_200_spread", "close_sma_50_ratio", "nvi_slope_20",
    "pvi_slope_20", "nvi_pvi_spread_20", "klinger_osc_34_55",
    "klinger_signal_13", "klinger_diff", "relative_vigor_10",
    "relative_vigor_signal_4", "force_index_z_20", "emv_z_20", "mfi_7",
    "mfi_21", "mfi_spread_7_21",
)

FEATURE_COLUMNS = [
    *RETURN_FEATURE_COLUMNS,
    *VOLATILITY_FEATURE_COLUMNS,
    *MA_FEATURE_COLUMNS,
    *MOMENTUM_FEATURE_COLUMNS,
    *VOLUME_FEATURE_COLUMNS,
    *STRUCTURE_FEATURE_COLUMNS,
    *SMC_FEATURE_COLUMNS,
    *STATISTIC_FEATURE_COLUMNS,
    *PERFORMANCE_FEATURE_COLUMNS,
    *TECHNICAL_INDICATOR_FEATURE_COLUMNS,
    *WEB_FACTOR_COLUMNS,
]
