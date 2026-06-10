from __future__ import annotations

from collections import Counter
from typing import Any

DERIVATIVES_CATEGORY = "derivatives"
UNKNOWN_CATEGORY = "other"
CATEGORY_BUDGETS = {
    DERIVATIVES_CATEGORY: {"maxShare": 0.4, "minExistingTotal": 5, "maxIdeasPerRound": 1},
    "microstructure": {"maxIdeasPerRound": 1},
    "volume_price": {"maxIdeasPerRound": 1},
    "risk_shape": {"maxIdeasPerRound": 1},
    "price_action": {"maxIdeasPerRound": 2},
}
SATURATION_SCORE_MULTIPLIER = 0.25

_CATEGORY_TERMS = (
    (
        DERIVATIVES_CATEGORY,
        (
            "fundingz",
            "funding_rate",
            "openinterestz",
            "open_interest",
            "longshortratioz",
            "long_short",
            "taker_buy",
            "taker_sell",
            "taker_buy_sell",
            "资金费率",
            "持仓",
            "多空",
            "主动性买",
            "主动性卖",
        ),
    ),
    (
        "microstructure",
        (
            "orderbook",
            "imbalance",
            "spread_bps",
            "microprice",
            "ofi",
            "bid_qty",
            "ask_qty",
            "盘口",
            "价差",
            "订单流",
        ),
    ),
    (
        "volume_price",
        ("vwap", "volume", "amount", "obv", "mfi", "cmf", "成交量", "量价"),
    ),
    (
        "risk_shape",
        (
            "atr",
            "truerange",
            "drawdown",
            "donchian",
            "realized_vol",
            "range_z",
            "volatility",
            "波动",
            "回撤",
        ),
    ),
    (
        "price_action",
        ("close", "open", "high", "low", "ret_", "return", "slope", "trend", "动量", "趋势"),
    ),
    (
        "onchain_sentiment",
        ("netflow", "active_addresses", "fear_greed", "stablecoin", "onchain", "链上", "情绪"),
    ),
)


def factor_category(payload: dict[str, Any]) -> str:
    text = _payload_text(payload).lower()
    for category, terms in _CATEGORY_TERMS:
        if any(term.lower() in text for term in terms):
            return category
    return UNKNOWN_CATEGORY


def category_share(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    categories = [factor_category(row) for row in rows]
    total = len(categories)
    counts = Counter(categories)
    return [
        {"category": category, "count": count, "share": count / total if total else 0.0}
        for category, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def category_budget_payload() -> dict[str, Any]:
    return {
        "budgets": CATEGORY_BUDGETS,
        "saturationScoreMultiplier": SATURATION_SCORE_MULTIPLIER,
        "policy": "cap over-represented derivatives and require cross-category exploration",
    }


def category_saturation(category: str, existing_rows: list[dict[str, Any]]) -> dict[str, Any]:
    budget = CATEGORY_BUDGETS.get(category) or {}
    max_share = budget.get("maxShare")
    min_existing_total = int(budget.get("minExistingTotal") or 0)
    total = len(existing_rows)
    if max_share is None or total < min_existing_total:
        return _saturation_payload(category, False, total, 0, max_share)
    count = sum(1 for row in existing_rows if factor_category(row) == category)
    next_share = (count + 1) / (total + 1)
    saturated = next_share > float(max_share)
    return _saturation_payload(category, saturated, total, count, max_share, next_share)


def adjusted_score(score: float, saturation: dict[str, Any]) -> float:
    if saturation.get("saturated"):
        return score * SATURATION_SCORE_MULTIPLIER
    return score


def _saturation_payload(
    category: str,
    saturated: bool,
    total: int,
    count: int,
    max_share: object,
    next_share: float | None = None,
) -> dict[str, Any]:
    return {
        "category": category,
        "saturated": saturated,
        "existingTotal": total,
        "existingCategoryCount": count,
        "maxShare": max_share,
        "nextShare": next_share,
    }


def _payload_text(payload: dict[str, Any]) -> str:
    idea = payload.get("idea") if isinstance(payload.get("idea"), dict) else {}
    values = [
        payload.get("factorName"),
        payload.get("factorDisplayName"),
        payload.get("displayName"),
        payload.get("formula"),
        idea.get("formulaHint"),
        idea.get("nameHint"),
        idea.get("displayNameZh"),
        " ".join(str(item) for item in idea.get("operatorTrace") or []),
        " ".join(str(item) for item in idea.get("requiredColumns") or []),
    ]
    return " ".join(str(value or "") for value in values)
