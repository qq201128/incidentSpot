"""
因子分类和标签系统
"""
from __future__ import annotations

import re
from typing import Any


# 因子类别定义
FACTOR_CATEGORIES = {
    "trend": {
        "label": "趋势跟踪",
        "keywords": ["ema", "sma", "ma", "trend", "adx", "slope"],
        "description": "捕捉市场趋势的因子"
    },
    "mean_reversion": {
        "label": "均值回归",
        "keywords": ["mean", "reversion", "rsi", "bb", "bollinger", "deviation"],
        "description": "预期价格回归均值的因子"
    },
    "momentum": {
        "label": "动量",
        "keywords": ["momentum", "roc", "rate of change", "macd"],
        "description": "基于价格变化速度的因子"
    },
    "volatility": {
        "label": "波动率",
        "keywords": ["std", "atr", "volatility", "variance", "keltner"],
        "description": "基于市场波动的因子"
    },
    "volume": {
        "label": "成交量",
        "keywords": ["volume", "vol", "obv", "vwap"],
        "description": "基于成交量的因子"
    },
    "pattern": {
        "label": "形态识别",
        "keywords": ["pattern", "candlestick", "engulfing", "doji", "hammer"],
        "description": "识别K线形态的因子"
    },
    "cross": {
        "label": "交叉信号",
        "keywords": ["cross", "crossover", "golden", "death"],
        "description": "基于均线交叉的因子"
    },
}

# 市场环境定义
MARKET_REGIMES = {
    "trending": {
        "label": "趋势环境",
        "description": "市场有明确方向"
    },
    "ranging": {
        "label": "震荡环境",
        "description": "市场横盘整理"
    },
    "high_volatility": {
        "label": "高波动",
        "description": "价格波动剧烈"
    },
    "low_volatility": {
        "label": "低波动",
        "description": "价格波动平缓"
    },
}


def auto_classify_factor(factor: dict[str, Any]) -> dict[str, Any]:
    """
    自动分类因子

    分析：
    - 公式关键词
    - 因子名称
    - 描述文本

    Returns:
        带分类标签的因子
    """
    formula = factor.get("formula", "").lower()
    name = factor.get("name", "").lower()
    description = factor.get("description", "").lower()

    searchable = f"{formula} {name} {description}"

    # 识别类别
    categories = []
    for cat_key, cat_info in FACTOR_CATEGORIES.items():
        if any(keyword in searchable for keyword in cat_info["keywords"]):
            categories.append(cat_key)

    # 如果没有匹配到类别，标记为未分类
    if not categories:
        categories = ["uncategorized"]

    # 自动检测适用环境（如果有历史数据）
    regimes = auto_detect_regimes(factor)

    # 生成标签
    tags = generate_tags(factor, categories, regimes)

    return {
        **factor,
        "categories": categories,
        "regimes": regimes,
        "tags": tags,
        "classified_at": None,  # 可以记录分类时间
    }


def auto_detect_regimes(factor: dict[str, Any]) -> list[str]:
    """
    自动检测因子适用的市场环境

    基于：
    - 历史回测表现
    - 因子类型
    """
    regimes = []

    # 基于分类推断
    categories = factor.get("categories", [])

    if "trend" in categories or "momentum" in categories:
        regimes.append("trending")

    if "mean_reversion" in categories:
        regimes.append("ranging")

    if "volatility" in categories:
        regimes.extend(["high_volatility", "low_volatility"])

    # 基于历史数据（如果有）
    if "regime_performance" in factor:
        perf = factor["regime_performance"]
        for regime, metrics in perf.items():
            if metrics.get("win_rate", 0) > 0.55:
                if regime not in regimes:
                    regimes.append(regime)

    return regimes


def generate_tags(
    factor: dict[str, Any],
    categories: list[str],
    regimes: list[str]
) -> list[str]:
    """
    生成因子标签

    包括：
    - 类别标签
    - 环境标签
    - 性能标签
    """
    tags = []

    # 类别标签
    for cat in categories:
        if cat in FACTOR_CATEGORIES:
            tags.append(FACTOR_CATEGORIES[cat]["label"])

    # 环境标签
    for regime in regimes:
        if regime in MARKET_REGIMES:
            tags.append(MARKET_REGIMES[regime]["label"])

    # 性能标签
    win_rate = factor.get("win_rate", 0)
    if win_rate >= 0.60:
        tags.append("高胜率")
    elif win_rate >= 0.55:
        tags.append("中等胜率")

    ir = factor.get("ir", 0)
    if ir >= 0.5:
        tags.append("高IR")
    elif ir >= 0.3:
        tags.append("中等IR")

    # 特征标签
    if factor.get("trades", 0) > 100:
        tags.append("高频")
    elif factor.get("trades", 0) < 20:
        tags.append("低频")

    return tags


def extract_parameters_from_formula(formula: str) -> dict[str, Any]:
    """
    从公式中提取参数

    例如：
    - "EMA(close, 20)" -> {"ema_period": 20}
    - "RSI(14)" -> {"rsi_period": 14}
    """
    params = {}

    # 提取所有数字
    numbers = re.findall(r'\b\d+\b', formula)
    if numbers:
        params["periods"] = [int(n) for n in numbers]

    # 提取常见指标及参数
    indicators = {
        "ema": r'ema\s*\(\s*\w+\s*,\s*(\d+)\s*\)',
        "sma": r'sma\s*\(\s*\w+\s*,\s*(\d+)\s*\)',
        "rsi": r'rsi\s*\(\s*(\d+)\s*\)',
        "atr": r'atr\s*\(\s*(\d+)\s*\)',
        "std": r'std\s*\(\s*\w+\s*,\s*(\d+)\s*\)',
    }

    for indicator, pattern in indicators.items():
        matches = re.findall(pattern, formula.lower())
        if matches:
            params[f"{indicator}_period"] = [int(m) for m in matches]

    return params


def search_factors_by_tags(
    factors: list[dict[str, Any]],
    required_tags: list[str] | None = None,
    excluded_tags: list[str] | None = None,
    categories: list[str] | None = None,
    regimes: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    按标签搜索因子

    Args:
        factors: 因子列表
        required_tags: 必须包含的标签
        excluded_tags: 必须不包含的标签
        categories: 必须属于的类别
        regimes: 必须适用的市场环境

    Returns:
        过滤后的因子列表
    """
    filtered = factors

    # 按类别过滤
    if categories:
        filtered = [
            f for f in filtered
            if any(cat in f.get("categories", []) for cat in categories)
        ]

    # 按环境过滤
    if regimes:
        filtered = [
            f for f in filtered
            if any(regime in f.get("regimes", []) for regime in regimes)
        ]

    # 按必需标签过滤
    if required_tags:
        filtered = [
            f for f in filtered
            if all(tag in f.get("tags", []) for tag in required_tags)
        ]

    # 按排除标签过滤
    if excluded_tags:
        filtered = [
            f for f in filtered
            if not any(tag in f.get("tags", []) for tag in excluded_tags)
        ]

    return filtered


def get_factor_taxonomy_summary(factors: list[dict[str, Any]]) -> dict[str, Any]:
    """
    获取因子库分类汇总

    Returns:
        分类统计信息
    """
    from collections import Counter

    category_counts = Counter()
    regime_counts = Counter()
    tag_counts = Counter()

    for factor in factors:
        for cat in factor.get("categories", []):
            category_counts[cat] += 1

        for regime in factor.get("regimes", []):
            regime_counts[regime] += 1

        for tag in factor.get("tags", []):
            tag_counts[tag] += 1

    return {
        "total_factors": len(factors),
        "by_category": dict(category_counts),
        "by_regime": dict(regime_counts),
        "top_tags": dict(tag_counts.most_common(10)),
        "uncategorized_count": category_counts.get("uncategorized", 0),
    }
