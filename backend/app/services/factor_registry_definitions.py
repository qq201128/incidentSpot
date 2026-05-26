from __future__ import annotations

from collections import Counter
from typing import Any

from app.services.factor_registry_core import (
    RULE_FACTOR_TIMEFRAMES,
    FactorCategory,
    FactorDefinition,
    FactorDirection,
)
from app.services.factor_registry_external import EXTENDED_KLINE_FACTORS
from app.services.factor_registry_order_funding import FUNDING_FACTORS, ORDERBOOK_FACTORS
from app.services.factor_registry_return_volatility import RETURN_FACTORS, VOLATILITY_FACTORS
from app.services.factor_registry_structure_timeframe import MULTI_TIMEFRAME_FACTORS, STRUCTURE_FACTORS
from app.services.factor_registry_trend_volume import MA_FACTORS, MOMENTUM_FACTORS, VOLUME_FACTORS


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
    return FACTOR_BY_NAME.get(name)


def list_factors(category: FactorCategory | None = None) -> list[FactorDefinition]:
    if category is None:
        return list(ALL_FACTORS)
    return [factor for factor in ALL_FACTORS if factor.category == category]


def list_factor_categories() -> list[dict[str, Any]]:
    counts = Counter(factor.category.value for factor in ALL_FACTORS)
    return [
        {"key": cat.value, "name": _category_display_name(cat), "count": counts.get(cat.value, 0)}
        for cat in FactorCategory
    ]


def factor_payload(factor: FactorDefinition) -> dict[str, Any]:
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
    cat = FactorCategory(category) if category else None
    return [factor_payload(factor) for factor in list_factors(cat)]


def _category_display_name(cat: FactorCategory) -> str:
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
