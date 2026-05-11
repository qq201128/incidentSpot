from __future__ import annotations

from dataclasses import dataclass

from app.services.kline_timing import KLINE_ENTRY_GRACE_MS, N_BAR_10M_RM_ENTRY_GRACE_MS

DEFAULT_STRATEGY_KEY = "vegas_fib_resonance"
MANUAL_STRATEGY_KEY = "manual"
DAILY_TRADE_FLOOR_TREE_STRATEGY_KEY = "daily_trade_floor_tree"
DAILY_TRADE_FLOOR_RULE_NAME = "daily_trade_floor_tree_v1"
DAILY_TRADE_FLOOR_MIN_DAILY_TRADES = 10
ORDERBOOK_NOTIONAL_STRATEGY_KEY = "orderbook_notional_40m"
ORDERBOOK_NOTIONAL_RULE_NAME = "orderbook_notional_value_delta_8m"
ORDERBOOK_NOTIONAL_ENTRY_GRACE_MS = KLINE_ENTRY_GRACE_MS

ORDERBOOK_NOTIONAL_MG_STRATEGY_KEY = "orderbook_notional_40m_mg"
ORDERBOOK_NOTIONAL_MG_RULE_NAME = "orderbook_notional_value_delta_8m_mg"

ORDERBOOK_TRADE_FLOW_STRATEGY_KEY = "orderbook_trade_flow_1k"
ORDERBOOK_TRADE_FLOW_RULE_NAME = "orderbook_depth_trade_flow_v1"
ORDERBOOK_TRADE_FLOW_ENTRY_GRACE_MS = KLINE_ENTRY_GRACE_MS

ORDERBOOK_TRADE_FLOW_INVERT_MG_STRATEGY_KEY = "orderbook_trade_flow_1k_invert_mg"
ORDERBOOK_TRADE_FLOW_INVERT_RULE_NAME = "orderbook_depth_trade_flow_v1_invert_mg"

BLIND_REVERSE_MARTINGALE_STRATEGY_KEY = "blind_reverse_martingale_v1"
BLIND_REVERSE_MARTINGALE_RULE_NAME = "blind_reverse_martingale_v1"

THREE_BAR_10M_RM_STRATEGY_KEY = "three_bar_10m_reverse_martingale_v1"
THREE_BAR_10M_RM_RULE_NAME = "three_bar_10m_reverse_martingale_v1"

FOUR_BAR_10M_RM_STRATEGY_KEY = "four_bar_10m_reverse_martingale_v1"
FOUR_BAR_10M_RM_RULE_NAME = "four_bar_10m_reverse_martingale_v1"

FIVE_BAR_10M_RM_STRATEGY_KEY = "five_bar_10m_reverse_martingale_v1"
FIVE_BAR_10M_RM_RULE_NAME = "five_bar_10m_reverse_martingale_v1"

# 与 blind RM 相同 10/20/45 倍投链；自动下单用 load_blind_rm_settlement_state(strategy_key, symbol)
N_BAR_10M_RM_STRATEGY_KEYS: frozenset[str] = frozenset(
    {
        THREE_BAR_10M_RM_STRATEGY_KEY,
        FOUR_BAR_10M_RM_STRATEGY_KEY,
        FIVE_BAR_10M_RM_STRATEGY_KEY,
    }
)

CONTINUOUS_ORDERBOOK_STRATEGY_KEYS: frozenset[str] = frozenset(
    {
        ORDERBOOK_NOTIONAL_STRATEGY_KEY,
        ORDERBOOK_NOTIONAL_MG_STRATEGY_KEY,
        ORDERBOOK_TRADE_FLOW_STRATEGY_KEY,
        ORDERBOOK_TRADE_FLOW_INVERT_MG_STRATEGY_KEY,
        BLIND_REVERSE_MARTINGALE_STRATEGY_KEY,
    }
)


def is_continuous_orderbook_strategy(strategy_key: str | None) -> bool:
    if not strategy_key:
        return False
    return strategy_key in CONTINUOUS_ORDERBOOK_STRATEGY_KEYS


@dataclass(frozen=True)
class StrategyDefinition:
    key: str
    name: str
    description: str
    requires_vegas_confirmation: bool
    signal_source: str
    requires_high_winrate_gate: bool = False
    requires_trade_quality_gate: bool = False
    rule_names: tuple[str, ...] | None = None
    tradable: bool = True
    disabled_reason: str | None = None
    backtest_summary: dict | None = None
    min_daily_trades: int | None = None
    requires_kline_features: bool = True
    uses_trade_policy_gates: bool = True
    entry_grace_ms: int = KLINE_ENTRY_GRACE_MS


PURE_RULE_PRECISION_RULE_NAMES = (
    "strong_adx_low_volume_ratio",
    "five_minute_pullback_early_day",
    "macd_washout_1h_rebound",
    "ema_cross_capitulation_low_volume",
    "macd_washout_late_session",
    "deep_ma120_afternoon",
    "fifteen_minute_pullback_morning",
)

WIN70_TRADE_MAX_RULE_NAMES = (
    "strong_adx_low_volume_ratio",
    "five_minute_pullback_early_day",
    "macd_washout_1h_rebound",
    "ema_cross_capitulation_low_volume",
    "asian_session_15m_flush",
    "macd_washout_late_session",
    "deep_ma120_afternoon",
    "fifteen_minute_pullback_morning",
)


STRATEGIES = (
    StrategyDefinition(
        key=DEFAULT_STRATEGY_KEY,
        name="Vegas/Fib 共振",
        description="10m 边界入场，EMA Vegas 通道 + 斐波那契 + 多周期共振确认。",
        requires_vegas_confirmation=True,
        signal_source="kline_boundary_vegas_fib_rule_engine",
    ),
    StrategyDefinition(
        key="high_winrate_rules",
        name="高胜率规则",
        description="10m 边界入场，仅使用已优化的多周期高胜率规则。",
        requires_vegas_confirmation=False,
        signal_source="kline_boundary_high_winrate_rule_engine",
        requires_high_winrate_gate=True,
    ),
    StrategyDefinition(
        key="pure_rule_precision",
        name="纯规则高精度",
        description="10m 边界入场，不使用模型，只启用历史验证通过的高精度规则组合。",
        requires_vegas_confirmation=False,
        signal_source="kline_boundary_pure_rule_precision_engine",
        requires_high_winrate_gate=True,
        rule_names=PURE_RULE_PRECISION_RULE_NAMES,
    ),
    StrategyDefinition(
        key="win70_trade_max_rules",
        name="70胜率高频",
        description="10m 边界入场，筛选历史日胜率守住 70% 且单数更多的规则组合。",
        requires_vegas_confirmation=False,
        signal_source="kline_boundary_win70_trade_max_rule_engine",
        requires_high_winrate_gate=True,
        rule_names=WIN70_TRADE_MAX_RULE_NAMES,
    ),
    StrategyDefinition(
        key=ORDERBOOK_NOTIONAL_STRATEGY_KEY,
        name="订单簿8M差额",
        description="10m K线刷新后30秒内抓取每侧1000挡，统计数量>1的名义价值；多空差额大于8M时跟随大额方向。",
        requires_vegas_confirmation=False,
        signal_source="kline_refresh_orderbook_notional_delta",
        rule_names=(ORDERBOOK_NOTIONAL_RULE_NAME,),
        requires_kline_features=False,
        uses_trade_policy_gates=False,
        entry_grace_ms=ORDERBOOK_NOTIONAL_ENTRY_GRACE_MS,
    ),
    StrategyDefinition(
        key=ORDERBOOK_NOTIONAL_MG_STRATEGY_KEY,
        name="订单簿8M差额·倍投",
        description=(
            "与「订单簿8M差额」同一套规则（千档名义差额>8M 跟随大额方向）；"
            "自动下单时若上一笔已结算且猜错，下一笔名义加倍，最多连续倍投 3 档后恢复为基础数量，单笔封顶 20 USDT。"
        ),
        requires_vegas_confirmation=False,
        signal_source="kline_refresh_orderbook_notional_delta_mg",
        rule_names=(ORDERBOOK_NOTIONAL_MG_RULE_NAME,),
        requires_kline_features=False,
        uses_trade_policy_gates=False,
        entry_grace_ms=ORDERBOOK_NOTIONAL_ENTRY_GRACE_MS,
    ),
    StrategyDefinition(
        key=ORDERBOOK_TRADE_FLOW_STRATEGY_KEY,
        name="挂单波动+成交流向",
        description=(
            "10m K 线打开后 30 秒内：两侧各 1000 挡快照间隔采样，用挂单量不平衡的变化刻画波动；"
            "并结合近期聚合成交的主动买/卖名义金额判断方向。"
        ),
        requires_vegas_confirmation=False,
        signal_source="kline_refresh_orderbook_trade_flow",
        rule_names=(ORDERBOOK_TRADE_FLOW_RULE_NAME,),
        requires_kline_features=False,
        uses_trade_policy_gates=False,
        entry_grace_ms=ORDERBOOK_TRADE_FLOW_ENTRY_GRACE_MS,
    ),
    StrategyDefinition(
        key=ORDERBOOK_TRADE_FLOW_INVERT_MG_STRATEGY_KEY,
        name="挂单波动+成交流向·反向倍投",
        description=(
            "与「挂单波动+成交流向」同一套计算，但交易方向取反；自动下单时若上一笔已结算且猜错，"
            "下一笔名义金额加倍（以上一笔下单量为基准），封顶 20 USDT，猜对后恢复为基础数量。"
        ),
        requires_vegas_confirmation=False,
        signal_source="kline_refresh_orderbook_trade_flow_invert_mg",
        rule_names=(ORDERBOOK_TRADE_FLOW_INVERT_RULE_NAME,),
        requires_kline_features=False,
        uses_trade_policy_gates=False,
        entry_grace_ms=ORDERBOOK_TRADE_FLOW_ENTRY_GRACE_MS,
    ),
    StrategyDefinition(
        key=BLIND_REVERSE_MARTINGALE_STRATEGY_KEY,
        name="随意首单·反向倍投(10/20/45)",
        description=(
            "不做信号判断：新一轮随机选涨或跌；若结算为亏，则固定押本轮首单的反向，依次倍投 10 / 20 / 45 USDT。"
            "任意一笔盈利（首单或倍投过程中）即结束本轮，下一笔重新随机首单并用基础数量。"
            "若押 45 USDT 仍亏（连同首单共连亏 4 笔），同样结束本轮，下一笔重新随机首单并用基础数量。"
        ),
        requires_vegas_confirmation=False,
        signal_source="blind_reverse_martingale_v1",
        rule_names=(BLIND_REVERSE_MARTINGALE_RULE_NAME,),
        requires_kline_features=False,
        uses_trade_policy_gates=False,
        entry_grace_ms=ORDERBOOK_NOTIONAL_ENTRY_GRACE_MS,
    ),
    StrategyDefinition(
        key=THREE_BAR_10M_RM_STRATEGY_KEY,
        name="三连10m反向·倍投(10/20/45)",
        description=(
            "仅在「新开一轮、当前无连亏记录」时：若前 3 根已收盘 10m **指数 K**（indexPriceKlines 与界面指数 10m 一致）为连续三阳或连续三阴，则押反向。"
            "一旦出现亏损进入倍投链：之后每一根 10m 均继续自动下注；名义按 10→20→45 USDT 递进。"
            "倍投方向规则：每一档押注方向与「本轮第一笔已结算亏损单」预测方向**相同**（同向加码，例如首轮押跌则 Recovery 仍押跌），"
            "直到猜对止盈、或连亏满 4 笔（含 45 档仍未中）后结束本轮并恢复基础数量；倍投阶段不再要求三连 K 形态。"
        ),
        requires_vegas_confirmation=False,
        signal_source="three_bar_10m_reverse_martingale_v1",
        rule_names=(THREE_BAR_10M_RM_RULE_NAME,),
        requires_kline_features=False,
        uses_trade_policy_gates=False,
        entry_grace_ms=N_BAR_10M_RM_ENTRY_GRACE_MS,
    ),
    StrategyDefinition(
        key=FOUR_BAR_10M_RM_STRATEGY_KEY,
        name="四连10m反向·倍投(10/20/45)",
        description=(
            "仅在「新开一轮、当前无连亏记录」时：若前 4 根已收盘 10m **指数 K**（indexPriceKlines 与界面指数 10m 一致）为连续四阳或连续四阴，则押反向。"
            "一旦出现亏损进入倍投链：之后每一根 10m 均继续自动下注（方向与本轮第一笔已结算亏损单的预测方向相同），名义按 10→20→45 USDT 递进，"
            "直到猜对止盈、或连亏满 4 笔（含 45 档仍未中）后结束本轮并恢复基础数量；倍投阶段不再要求四连 K 形态。"
        ),
        requires_vegas_confirmation=False,
        signal_source="four_bar_10m_reverse_martingale_v1",
        rule_names=(FOUR_BAR_10M_RM_RULE_NAME,),
        requires_kline_features=False,
        uses_trade_policy_gates=False,
        entry_grace_ms=N_BAR_10M_RM_ENTRY_GRACE_MS,
    ),
    StrategyDefinition(
        key=FIVE_BAR_10M_RM_STRATEGY_KEY,
        name="五连10m反向·倍投(10/20/45)",
        description=(
            "仅在「新开一轮、当前无连亏记录」时：若前 5 根已收盘 10m **指数 K**（indexPriceKlines 与界面指数 10m 一致）为连续五阳或连续五阴，则押反向。"
            "一旦出现亏损进入倍投链：之后每一根 10m 均继续自动下注（方向与本轮第一笔已结算亏损单的预测方向相同），名义按 10→20→45 USDT 递进，"
            "直到猜对止盈、或连亏满 4 笔（含 45 档仍未中）后结束本轮并恢复基础数量；倍投阶段不再要求五连 K 形态。"
        ),
        requires_vegas_confirmation=False,
        signal_source="five_bar_10m_reverse_martingale_v1",
        rule_names=(FIVE_BAR_10M_RM_RULE_NAME,),
        requires_kline_features=False,
        uses_trade_policy_gates=False,
        entry_grace_ms=N_BAR_10M_RM_ENTRY_GRACE_MS,
    ),
    StrategyDefinition(
        key=DAILY_TRADE_FLOOR_TREE_STRATEGY_KEY,
        name="70胜率日频树规则",
        description="10m 边界入场，新建树形规则；仅交易历史叶子胜率大于 70% 的信号，每日不少于 10 单。",
        requires_vegas_confirmation=False,
        signal_source="kline_boundary_daily_trade_floor_tree_rule_engine",
        requires_high_winrate_gate=True,
        rule_names=(DAILY_TRADE_FLOOR_RULE_NAME,),
        min_daily_trades=DAILY_TRADE_FLOOR_MIN_DAILY_TRADES,
    ),
)


def strategy_definition(strategy_key: str | None) -> StrategyDefinition:
    key = strategy_key or DEFAULT_STRATEGY_KEY
    for strategy in STRATEGIES:
        if strategy.key == key:
            return strategy
    raise ValueError(f"unsupported strategy: {key}")


def strategy_payloads() -> list[dict]:
    return [
        {
            "key": strategy.key,
            "name": strategy.name,
            "description": strategy.description,
            "requiresVegasConfirmation": strategy.requires_vegas_confirmation,
            "requiresHighWinrateGate": strategy.requires_high_winrate_gate,
            "requiresTradeQualityGate": strategy.requires_trade_quality_gate,
            "signalSource": strategy.signal_source,
            "ruleNames": strategy.rule_names,
            "tradable": strategy.tradable,
            "disabledReason": strategy.disabled_reason,
            "backtestSummary": strategy.backtest_summary,
            "minDailyTrades": strategy.min_daily_trades,
            "requiresKlineFeatures": strategy.requires_kline_features,
            "usesTradePolicyGates": strategy.uses_trade_policy_gates,
            "entryGraceMs": strategy.entry_grace_ms,
        }
        for strategy in STRATEGIES
    ]


def strategy_requires_kline_features(strategy_key: str | None) -> bool:
    return strategy_definition(strategy_key).requires_kline_features


def strategy_uses_trade_policy_gates(strategy_key: str | None) -> bool:
    return strategy_definition(strategy_key).uses_trade_policy_gates


def strategy_entry_grace_ms(strategy_key: str | None) -> int:
    return strategy_definition(strategy_key).entry_grace_ms
