from __future__ import annotations

from dataclasses import dataclass
from typing import Any

RULE_DURATION = "10m"
RULE_HORIZON_MINUTES = 10

# 预测 / 自动下单 / 事件合约共用周期（与 train_10m.DURATION_TO_HORIZON、Binance 事件周期一致）
DURATION_TO_MINUTES: dict[str, int] = {"10m": 10, "30m": 30, "60m": 60, "1d": 1440}
SUPPORTED_RULE_DURATIONS = frozenset(DURATION_TO_MINUTES.keys())


def horizon_minutes_for_duration(duration: str) -> int:
    if duration not in DURATION_TO_MINUTES:
        raise ValueError(f"unsupported duration: {duration}")
    return DURATION_TO_MINUTES[duration]
RULE_GATE_NAME = "kline_boundary_vegas_fib_orderbook_10m"
RULE_TARGET_WIN_RATE = 0.70
RULE_MIN_CONFIDENCE = 0.70
RULE_MIN_QUALITY_SCORE = 0.70
ORDERBOOK_LIMIT = 500
MAX_SPREAD_BPS = 8.0
ORDERBOOK_IMBALANCE_SCALE = 0.30
ORDERBOOK_OFI_SCALE = 0.50
ORDERBOOK_MICROPRICE_BPS_SCALE = 2.0
ORDERBOOK_IMBALANCE_WEIGHT = 0.40
ORDERBOOK_OFI_WEIGHT = 0.35
ORDERBOOK_MICROPRICE_WEIGHT = 0.25
NEUTRAL_PROBABILITY = 0.50
OBSERVATION_MAX_EDGE = 0.18
MIN_SIGNAL_SCORE = -1.0
MAX_SIGNAL_SCORE = 1.0
MAX_RULE_CONFIDENCE = 0.99
BPS_DIVISOR = 10_000.0
INTRABAR_CENTER_SCALE = 2.0
MIN_ALIGNMENT_RATIO = 0.66
MS_PER_MINUTE = 60_000
WALK_FORWARD_MIN_TRAIN_DAYS = 14
WALK_FORWARD_TEST_DAYS = 7
WALK_FORWARD_PURGE_MINUTES = RULE_HORIZON_MINUTES


def walk_forward_purge_minutes_for_duration(duration: str) -> int:
    """Walk-forward purge 分钟数 = 该周期的 horizon 分钟数（防止前瞻偏差）。"""
    return horizon_minutes_for_duration(duration)

HIGHER_TIMEFRAME_WEIGHT = 0.35
TEN_MINUTE_WEIGHT = 0.25
ORDERBOOK_WEIGHT = 0.20
VEGAS_RESONANCE_WEIGHT = 0.20
VEGAS_SCORE_SCALE = 100.0
VEGAS_MIN_DIRECTION_SCORE = 0.15
VEGAS_MIN_RESONANCE_SCORE = 40.0
MOMENTUM_WEIGHT = 0.55
MA_BIAS_WEIGHT = 0.35
BODY_WEIGHT = 0.10
QUALITY_CONFIDENCE_WEIGHT = 0.45
QUALITY_SPREAD_WEIGHT = 0.20
QUALITY_ALIGNMENT_WEIGHT = 0.35


@dataclass(frozen=True)
class TimeframeRule:
    interval: str
    lookback: int
    ma_window: int
    scale_bps: float
    weight: float


TEN_MINUTE_RULE = TimeframeRule("10m", 3, 6, 45.0, 1.0)
HIGHER_TIMEFRAME_RULES = (
    TimeframeRule("30m", 3, 8, 80.0, 0.25),
    TimeframeRule("1h", 4, 8, 110.0, 0.25),
    TimeframeRule("4h", 3, 6, 180.0, 0.30),
    TimeframeRule("1d", 2, 5, 300.0, 0.20),
)

OPTIMIZED_EVENT_RULES: tuple[dict[str, Any], ...] = (
    {
        "name": "deep_pullback_4h_volatility",
        "direction": "up",
        "win_rate": 0.8773584905660378,
        "min_daily_win_rate": 0.7142857142857143,
        "conditions": (
            ("ma_ratio_60", "<=", -0.007640377101417159),
            ("tf_4h_ret_vol_3", ">=", 0.010658846290432187),
        ),
    },
    {
        "name": "five_minute_pullback_quiet_1h",
        "direction": "up",
        "win_rate": 0.9777777777777777,
        "min_daily_win_rate": 0.9090909090909091,
        "conditions": (
            ("tf_5m_ma_ratio_12", "<=", -0.0036991330602377413),
            ("tf_1h_ret_vol_12", "<=", 0.0019944489552506917),
        ),
    },
    {
        "name": "daily_low_position_volume",
        "direction": "up",
        "win_rate": 0.8,
        "min_daily_win_rate": 0.7142857142857143,
        "conditions": (
            ("tf_1d_intrabar_pos", "<=", 0.04253326511439651),
            ("vol_ma_60", ">=", 412.3079393333335),
        ),
    },
    {
        "name": "strong_adx_low_volume_ratio",
        "direction": "up",
        "win_rate": 1.0,
        "min_daily_win_rate": 1.0,
        "conditions": (
            ("adx_14", ">=", 55.91049351461014),
            ("vol_ratio_240", "<=", 0.39234044622088776),
        ),
    },
    {
        "name": "five_minute_pullback_early_day",
        "direction": "up",
        "win_rate": 0.9193548387096774,
        "min_daily_win_rate": 0.8571428571428571,
        "conditions": (
            ("tf_5m_ma_ratio_12", "<=", -0.0036991330602377413),
            ("tf_1d_volume_share", "<=", 0.029861111111111113),
        ),
    },
    {
        "name": "macd_washout_1h_rebound",
        "direction": "up",
        "win_rate": 0.8555555555555555,
        "min_daily_win_rate": 0.7727272727272727,
        "conditions": (
            ("macd_signal", "<=", -129.87204055373286),
            ("tf_1h_ret_4", ">=", 0.004533698716392198),
        ),
    },
    {
        "name": "ema_cross_capitulation_low_volume",
        "direction": "up",
        "win_rate": 1.0,
        "min_daily_win_rate": 1.0,
        "conditions": (
            ("ema_cross", "<=", -0.0013289884837553703),
            ("vol_ma_120", "<=", 71.72723666666666),
        ),
    },
    {
        "name": "asian_session_15m_flush",
        "direction": "up",
        "win_rate": 0.8688524590163934,
        "min_daily_win_rate": 0.7142857142857143,
        "conditions": (
            ("tf_15m_ret_4", "<=", -0.015131006882729071),
            ("tod_bucket", "<=", 16.0),
        ),
    },
    {
        "name": "daily_floor_rsi_reclaim",
        "direction": "up",
        "win_rate": 1.0,
        "min_daily_win_rate": 1.0,
        "conditions": (
            ("tf_4h_intrabar_pos", "<=", 0.012045000633879243),
            ("rsi_14", ">=", 64.19098707792983),
        ),
    },
    {
        "name": "compressed_1h_low_volume_mean_revert",
        "direction": "up",
        "win_rate": 0.7964071856287425,
        "min_daily_win_rate": 0.7142857142857143,
        "conditions": (
            ("tf_1h_ret_vol_4", "<=", 0.0004922446102912526),
            ("vol_ma_120", "<=", 20.442336416666667),
        ),
    },
    {
        "name": "daily_bottom_midday",
        "direction": "up",
        "win_rate": 1.0,
        "min_daily_win_rate": 1.0,
        "conditions": (
            ("tf_1d_intrabar_pos", "<=", 0.05662481855090655),
            ("tod_bucket", "==", 12),
        ),
    },
    {
        "name": "macd_washout_late_session",
        "direction": "up",
        "win_rate": 0.8333333333333334,
        "min_daily_win_rate": 0.7142857142857143,
        "conditions": (
            ("macd_signal", "<=", -103.3112048830281),
            ("tod_bucket", "==", 41),
        ),
    },
    {
        "name": "low_volatility_1h_and_15m",
        "direction": "up",
        "win_rate": 0.835820895522388,
        "min_daily_win_rate": 0.708994708994709,
        "conditions": (
            ("tf_1h_ret_vol_4", "<=", 0.000413053985189405),
            ("tf_15m_ret_vol_8", "<=", 0.0009508018369897207),
        ),
    },
    {
        "name": "deep_ma120_afternoon",
        "direction": "up",
        "win_rate": 0.8333333333333334,
        "min_daily_win_rate": 0.7142857142857143,
        "conditions": (
            ("ma_ratio_120", "<=", -0.00571154556797977),
            ("tod_bucket", "==", 31),
        ),
    },
    {
        "name": "fifteen_minute_pullback_morning",
        "direction": "up",
        "win_rate": 0.8636363636363636,
        "min_daily_win_rate": 0.7142857142857143,
        "conditions": (
            ("tf_15m_ma_ratio_8", "<=", -0.004870111692453083),
            ("tod_bucket", "==", 6),
        ),
    },
)
