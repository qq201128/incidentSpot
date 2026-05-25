from __future__ import annotations

from dataclasses import dataclass

from app.services.lstm_config import (
    LSTM_RULE_NAME,
    is_lstm_shadow_strategy,
    lstm_shadow_strategy_key,
    lstm_strategy_duration,
)
from app.services.kline_timing import KLINE_ENTRY_GRACE_MS
from app.services.model_family_config import (
    MODEL_FAMILIES,
    is_model_family_shadow_strategy,
    model_family_rule_name,
    model_family_strategy_key,
    parse_model_family_strategy,
)
from app.services.ensemble_judge_constants import (
    ENSEMBLE_RANKER_RULE_NAME,
    ENSEMBLE_RANKER_STRATEGY_KEY,
)
from app.services.factor_candidate_signal_keys import is_factor_candidate_signal_key
from app.services.rule_config import DURATION_TO_MINUTES, SUPPORTED_RULE_DURATIONS

FACTOR_COMBO_STRATEGY_KEY = "factor_combo_ranker_v1"
FACTOR_COMBO_RULE_NAME = "factor_combo_cached_ranking_v1"
HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY = "high_winrate_factor_combo_v1"
HIGH_WINRATE_FACTOR_COMBO_RULE_NAME = "high_winrate_factor_combo_goal_v1"
DEFAULT_STRATEGY_KEY = FACTOR_COMBO_STRATEGY_KEY
MANUAL_STRATEGY_KEY = "manual"
MODEL_SHADOW_DURATIONS = tuple(DURATION_TO_MINUTES)


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
    supported_durations: frozenset[str] = SUPPORTED_RULE_DURATIONS


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
        key=FACTOR_COMBO_STRATEGY_KEY,
        name="多因子组合执行",
        description=(
            "读取因子页缓存的最佳组合因子；每个结算周期独立使用该周期胜率最高的组合，"
            "用当前组合分数给出多空方向，可在自动执行中按周期开启模拟或实盘。"
        ),
        requires_vegas_confirmation=False,
        signal_source="factor_combination_ranking",
        rule_names=(FACTOR_COMBO_RULE_NAME,),
        requires_kline_features=True,
        uses_trade_policy_gates=False,
    ),
    StrategyDefinition(
        key=HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY,
        name="高胜率目标组合执行",
        description=(
            "只读取 high-winrate goal 搜索写入的 goal_combo 组合；"
            "已并入综合裁判信号层，不再作为独立执行项。"
        ),
        requires_vegas_confirmation=False,
        signal_source="high_winrate_factor_combo_goal",
        rule_names=(HIGH_WINRATE_FACTOR_COMBO_RULE_NAME,),
        tradable=False,
        disabled_reason="已并入综合裁判信号层，请使用多因子组合执行 + 综合裁判模拟。",
        requires_kline_features=True,
        uses_trade_policy_gates=False,
    ),
    StrategyDefinition(
        key=ENSEMBLE_RANKER_STRATEGY_KEY,
        name="综合裁判模拟",
        description="按候选信号裁判层建议权重综合投票；后端强制仅允许模拟下单。",
        requires_vegas_confirmation=False,
        signal_source="ensemble_judge",
        rule_names=(ENSEMBLE_RANKER_RULE_NAME,),
        tradable=True,
        requires_kline_features=False,
        uses_trade_policy_gates=False,
    ),
)


def strategy_definition(strategy_key: str | None) -> StrategyDefinition:
    key = strategy_key or DEFAULT_STRATEGY_KEY
    for strategy in STRATEGIES:
        if strategy.key == key:
            return strategy
    if key.startswith(f"{FACTOR_COMBO_STRATEGY_KEY}_combo_"):
        return _batch_combo_strategy_definition(key, "批量多因子组合模拟", "factor_combination_ranking")
    if key.startswith(f"{HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY}_combo_"):
        return _batch_combo_strategy_definition(key, "批量高胜率组合模拟", "high_winrate_factor_combo_goal")
    if key.startswith(f"{FACTOR_COMBO_STRATEGY_KEY}_top"):
        return _factor_combo_shadow_strategy_definition(key)
    if key.startswith(f"{HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY}_top"):
        return _high_winrate_combo_shadow_strategy_definition(key)
    if is_factor_candidate_signal_key(key):
        return _factor_candidate_signal_definition(key)
    if is_model_family_shadow_strategy(key):
        return _model_family_shadow_strategy_definition(key)
    raise ValueError(f"unsupported strategy: {key}")


def _batch_combo_strategy_definition(strategy_key: str, name: str, source: str) -> StrategyDefinition:
    return StrategyDefinition(
        key=strategy_key,
        name=name,
        description="按组合因子独立执行键批量开模拟仓，用于快速验证因子组合。",
        requires_vegas_confirmation=False,
        signal_source=source,
        rule_names=(FACTOR_COMBO_RULE_NAME,),
        tradable=True,
        requires_kline_features=True,
        uses_trade_policy_gates=False,
    )


def _factor_combo_shadow_strategy_definition(strategy_key: str) -> StrategyDefinition:
    rank = strategy_key.removeprefix(f"{FACTOR_COMBO_STRATEGY_KEY}_top")
    return StrategyDefinition(
        key=strategy_key,
        name=f"多因子组合执行·Top{rank}",
        description="多因子组合执行影子排名，仅用于模拟实盘对比。",
        requires_vegas_confirmation=False,
        signal_source="factor_combination_ranking",
        rule_names=(FACTOR_COMBO_RULE_NAME,),
        tradable=False,
        requires_kline_features=True,
        uses_trade_policy_gates=False,
    )


def _high_winrate_combo_shadow_strategy_definition(strategy_key: str) -> StrategyDefinition:
    rank = strategy_key.removeprefix(f"{HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY}_top")
    return StrategyDefinition(
        key=strategy_key,
        name=f"高胜率目标组合·Top{rank}",
        description="高胜率目标组合影子排名，仅用于模拟实盘对比。",
        requires_vegas_confirmation=False,
        signal_source="high_winrate_factor_combo_goal",
        rule_names=(HIGH_WINRATE_FACTOR_COMBO_RULE_NAME,),
        tradable=False,
        requires_kline_features=True,
        uses_trade_policy_gates=False,
    )


def _factor_candidate_signal_definition(strategy_key: str) -> StrategyDefinition:
    suffix = strategy_key.rsplit("_", 1)[-1]
    return StrategyDefinition(
        key=strategy_key,
        name=f"因子候选信号·{suffix}",
        description="单因子、Agent 因子和技术指标发布的独立候选信号，用于综合裁判观察加权。",
        requires_vegas_confirmation=False,
        signal_source="factor_candidate_signal",
        rule_names=("factor_candidate_signal_v1",),
        tradable=False,
        requires_kline_features=True,
        uses_trade_policy_gates=False,
    )


def _lstm_shadow_strategy_definition(strategy_key: str) -> StrategyDefinition:
    duration = lstm_strategy_duration(strategy_key)
    return StrategyDefinition(
        key=strategy_key,
        name=f"LSTM模拟执行·{duration}",
        description="LSTM 候选算法可开启模拟执行，真实下单仍由后端禁止。",
        requires_vegas_confirmation=False,
        signal_source="factor_lstm_shadow",
        rule_names=(LSTM_RULE_NAME,),
        tradable=True,
        requires_kline_features=True,
        uses_trade_policy_gates=False,
        supported_durations=frozenset({duration}),
    )


def _model_family_shadow_strategy_definition(strategy_key: str) -> StrategyDefinition:
    parsed = parse_model_family_strategy(strategy_key)
    if parsed is None:
        raise ValueError(f"not a model family shadow strategy: {strategy_key}")
    family, duration = parsed
    label = family.upper() if family != "random_forest" else "RandomForest"
    return StrategyDefinition(
        key=strategy_key,
        name=f"{label}模拟执行·{duration}",
        description=f"{family} 候选算法可开启模拟执行，每个算法族独立训练、预测和缓存。",
        requires_vegas_confirmation=False,
        signal_source=f"factor_{family}_shadow",
        rule_names=(model_family_rule_name(family),),
        tradable=True,
        requires_kline_features=True,
        uses_trade_policy_gates=False,
        supported_durations=frozenset({duration}),
    )


def strategy_payloads() -> list[dict]:
    return [
        _strategy_payload(strategy)
        for strategy in _visible_strategy_definitions()
    ]


def _visible_strategy_definitions() -> tuple[StrategyDefinition, ...]:
    base = tuple(
        item
        for item in STRATEGIES
        if item.tradable and item.key != ENSEMBLE_RANKER_STRATEGY_KEY
    )
    return base + _model_family_shadow_strategy_definitions()


def _model_family_shadow_strategy_definitions() -> tuple[StrategyDefinition, ...]:
    return tuple(
        _model_family_shadow_strategy_definition(model_family_strategy_key(family, duration))
        for family in MODEL_FAMILIES
        for duration in MODEL_SHADOW_DURATIONS
    )


def _strategy_payload(strategy: StrategyDefinition) -> dict:
    return {
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
        "supportedDurations": sorted(strategy.supported_durations),
    }


def strategy_requires_kline_features(strategy_key: str | None) -> bool:
    return strategy_definition(strategy_key).requires_kline_features


def strategy_uses_trade_policy_gates(strategy_key: str | None) -> bool:
    return strategy_definition(strategy_key).uses_trade_policy_gates


def strategy_entry_grace_ms(strategy_key: str | None) -> int:
    return strategy_definition(strategy_key).entry_grace_ms


def strategy_supported_durations(strategy_key: str | None) -> frozenset[str]:
    return strategy_definition(strategy_key).supported_durations


def strategy_supports_duration(strategy_key: str | None, duration: str) -> bool:
    return duration in strategy_supported_durations(strategy_key)
