from __future__ import annotations

from app.services.factor_registry_core import FactorCategory, FactorDefinition, FactorDirection


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


