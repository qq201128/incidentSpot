from __future__ import annotations

from dataclasses import dataclass

from app.services.kline_timing import KLINE_ENTRY_GRACE_MS, N_BAR_10M_RM_ENTRY_GRACE_MS

DEFAULT_STRATEGY_KEY = "orderbook_notional_40m"
MANUAL_STRATEGY_KEY = "manual"
ORDERBOOK_NOTIONAL_STRATEGY_KEY = "orderbook_notional_40m"
ORDERBOOK_NOTIONAL_RULE_NAME = "orderbook_notional_value_delta_8m"
ORDERBOOK_NOTIONAL_ENTRY_GRACE_MS = KLINE_ENTRY_GRACE_MS

ORDERBOOK_NOTIONAL_MG_STRATEGY_KEY = "orderbook_notional_40m_mg"
ORDERBOOK_NOTIONAL_MG_RULE_NAME = "orderbook_notional_value_delta_8m_mg"

ORDERBOOK_NOTIONAL_MG_5102045_STRATEGY_KEY = "orderbook_notional_10m_mg_5102045"
ORDERBOOK_NOTIONAL_MG_5102045_RULE_NAME = "orderbook_notional_value_delta_8m_mg_5102045"

ORDERBOOK_NOTIONAL_10M_STRATEGY_KEY = "orderbook_notional_10m"
ORDERBOOK_NOTIONAL_10M_RULE_NAME = "orderbook_notional_value_delta_10m"

ORDERBOOK_NOTIONAL_15M_STRATEGY_KEY = "orderbook_notional_15m"
ORDERBOOK_NOTIONAL_15M_RULE_NAME = "orderbook_notional_value_delta_15m"

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

# 三连/四连/五连 10m：自动下单用面板基础数量；仅在指数 n 连形态再现时下注，无 Recovery 补单。
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
        ORDERBOOK_NOTIONAL_10M_STRATEGY_KEY,
        ORDERBOOK_NOTIONAL_15M_STRATEGY_KEY,
        ORDERBOOK_NOTIONAL_MG_STRATEGY_KEY,
        ORDERBOOK_NOTIONAL_MG_5102045_STRATEGY_KEY,
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


OPTIMIZED_RULES_BACKTEST_META_KEY = "optimized_rules_10m"


STRATEGIES = (
    StrategyDefinition(
        key=OPTIMIZED_RULES_BACKTEST_META_KEY,
        name="优化规则集（回测）",
        description="历史 K 线优化规则目录回测；仅用于 /api/rules/backtest。",
        requires_vegas_confirmation=False,
        signal_source="optimized_rules_catalog_simulation",
        rule_names=None,
        tradable=False,
        disabled_reason="仅供规则回测 API",
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
        key=ORDERBOOK_NOTIONAL_MG_5102045_STRATEGY_KEY,
        name="订单簿10M差额·倍投",
        description=(
            "与「订单簿8M差额」同一套信号（千档名义差额>8M 跟随大额方向）；"
            "自动下单名义固定阶梯：首单 5 USDT，连亏后依次为 10 / 20 / 45 USDT，最多连续倍投 3 档；"
            "连亏满 4 笔（含 45 档仍未中）后下一笔回到 5 USDT。面板数量对该策略名义无影响。"
        ),
        requires_vegas_confirmation=False,
        signal_source="kline_refresh_orderbook_notional_delta_mg_5102045",
        rule_names=(ORDERBOOK_NOTIONAL_MG_5102045_RULE_NAME,),
        requires_kline_features=False,
        uses_trade_policy_gates=False,
        entry_grace_ms=ORDERBOOK_NOTIONAL_ENTRY_GRACE_MS,
    ),
    StrategyDefinition(
        key=ORDERBOOK_NOTIONAL_10M_STRATEGY_KEY,
        name="订单簿10M差额",
        description=(
            "与「订单簿8M差额」同一套计算（10m K 线刷新后窗口内千档，统计数量>1 的名义价值）；"
            "多空名义差额大于 10M USDT 时跟随大额方向；自动下单使用面板基础数量，无倍投。"
        ),
        requires_vegas_confirmation=False,
        signal_source="kline_refresh_orderbook_notional_delta_10m",
        rule_names=(ORDERBOOK_NOTIONAL_10M_RULE_NAME,),
        requires_kline_features=False,
        uses_trade_policy_gates=False,
        entry_grace_ms=ORDERBOOK_NOTIONAL_ENTRY_GRACE_MS,
    ),
    StrategyDefinition(
        key=ORDERBOOK_NOTIONAL_15M_STRATEGY_KEY,
        name="订单簿15M差额",
        description=(
            "与「订单簿8M差额」同一套计算；多空名义差额大于 15M USDT 时跟随大额方向；"
            "自动下单使用面板基础数量，无倍投。"
        ),
        requires_vegas_confirmation=False,
        signal_source="kline_refresh_orderbook_notional_delta_15m",
        rule_names=(ORDERBOOK_NOTIONAL_15M_RULE_NAME,),
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
        name="三连10m反向·不倍投",
        description=(
            "若前 3 根已收盘 10m **指数 K**（indexPriceKlines 与界面指数 10m 一致）为连续三阳或连续三阴，"
            "则对**当前新开**这根 10m 押与前 3 根趋势相反（涨押跌、跌押涨）。"
            "连亏后不再自动补单；须再次满足三连形态才下注。上一笔已结算（不论输赢）之后，用于判定的 n 根 K 须全部落在该笔对应 10m 桶之后，避免形态 K 与上一笔重叠。"
            "自动下单名义为面板基础数量，无倍投。"
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
        name="四连10m反向·不倍投",
        description=(
            "若前 4 根已收盘 10m **指数 K** 为连续四阳或连续四阴，则对当前新开 10m 押反向。"
            "连亏后不补单；须再次满足四连形态才下注；形态 K 须在上一笔已结算事件对应桶之后（规则同三连）。名义为面板基础数量。"
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
        name="五连10m反向·不倍投",
        description=(
            "若前 5 根已收盘 10m **指数 K** 为连续五阳或连续五阴，则对当前新开 10m 押反向。"
            "连亏后不补单；须再次满足五连形态才下注；形态 K 须在上一笔已结算事件对应桶之后（规则同三连）。名义为面板基础数量。"
        ),
        requires_vegas_confirmation=False,
        signal_source="five_bar_10m_reverse_martingale_v1",
        rule_names=(FIVE_BAR_10M_RM_RULE_NAME,),
        requires_kline_features=False,
        uses_trade_policy_gates=False,
        entry_grace_ms=N_BAR_10M_RM_ENTRY_GRACE_MS,
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
        if strategy.tradable
    ]


def strategy_requires_kline_features(strategy_key: str | None) -> bool:
    return strategy_definition(strategy_key).requires_kline_features


def strategy_uses_trade_policy_gates(strategy_key: str | None) -> bool:
    return strategy_definition(strategy_key).uses_trade_policy_gates


def strategy_entry_grace_ms(strategy_key: str | None) -> int:
    return strategy_definition(strategy_key).entry_grace_ms
