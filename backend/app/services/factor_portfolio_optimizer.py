"""
因子组合优化器 - 降低相关性，提高多样化
"""
from __future__ import annotations

from typing import Any

import numpy as np


def optimize_factor_portfolio(
    factors: list[dict[str, Any]],
    max_correlation: float = 0.7,
    min_factors: int = 3,
    max_factors: int = 10,
) -> dict[str, Any]:
    """
    优化因子组合

    目标：
    - 降低因子间相关性
    - 提高多样化
    - 覆盖多种市场环境

    Args:
        factors: 因子列表
        max_correlation: 最大允许相关性
        min_factors: 最少因子数
        max_factors: 最多因子数

    Returns:
        优化建议
    """
    if len(factors) < 2:
        return {
            "status": "insufficient_factors",
            "message": "因子数量不足，至少需要2个因子"
        }

    # 1. 计算相关性矩阵
    corr_matrix = compute_correlation_matrix(factors)

    # 2. 识别高相关因子对
    high_corr_pairs = find_high_correlation_pairs(factors, corr_matrix, max_correlation)

    # 3. 检查环境覆盖
    regime_coverage = check_regime_coverage(factors)

    # 4. 检查类别多样性
    category_diversity = check_category_diversity(factors)

    # 5. 生成优化建议
    suggestions = generate_optimization_suggestions(
        factors,
        high_corr_pairs,
        regime_coverage,
        category_diversity
    )

    # 6. 计算组合得分
    portfolio_score = calculate_portfolio_score(
        factors,
        corr_matrix,
        regime_coverage,
        category_diversity
    )

    return {
        "status": "success",
        "current_factors": len(factors),
        "portfolio_score": portfolio_score,
        "high_correlation_pairs": len(high_corr_pairs),
        "regime_coverage": regime_coverage,
        "category_diversity": category_diversity,
        "suggestions": suggestions,
        "metrics": {
            "avg_correlation": float(np.mean(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])),
            "max_correlation": float(np.max(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])),
            "diversification_score": calculate_diversification_score(corr_matrix),
        }
    }


def compute_correlation_matrix(factors: list[dict[str, Any]]) -> np.ndarray:
    """
    计算因子间相关性矩阵

    基于因子的回测信号序列
    """
    n = len(factors)
    corr_matrix = np.eye(n)

    for i in range(n):
        for j in range(i + 1, n):
            corr = compute_factor_correlation(factors[i], factors[j])
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr

    return corr_matrix


def compute_factor_correlation(factor1: dict, factor2: dict) -> float:
    """计算两个因子的相关性"""
    # 如果有回测信号，使用信号相关性
    if "backtest_signals" in factor1 and "backtest_signals" in factor2:
        s1 = factor1["backtest_signals"]
        s2 = factor2["backtest_signals"]

        if len(s1) == len(s2) and len(s1) > 0:
            try:
                corr = np.corrcoef(s1, s2)[0, 1]
                return abs(corr) if not np.isnan(corr) else 0.0
            except:
                pass

    # 否则基于公式相似度
    from app.services.factor_similarity_detector import compute_formula_similarity
    return compute_formula_similarity(
        factor1.get("formula", ""),
        factor2.get("formula", "")
    )


def find_high_correlation_pairs(
    factors: list[dict],
    corr_matrix: np.ndarray,
    threshold: float
) -> list[dict]:
    """识别高相关因子对"""
    pairs = []
    n = len(factors)

    for i in range(n):
        for j in range(i + 1, n):
            if corr_matrix[i, j] > threshold:
                pairs.append({
                    "factor1": factors[i],
                    "factor2": factors[j],
                    "correlation": float(corr_matrix[i, j]),
                    "better_factor": _select_better_factor(factors[i], factors[j])
                })

    return sorted(pairs, key=lambda p: p["correlation"], reverse=True)


def _select_better_factor(factor1: dict, factor2: dict) -> dict:
    """选择表现更好的因子"""
    # 综合评分：IR + 胜率
    score1 = (factor1.get("ir", 0) * 0.6 + factor1.get("win_rate", 0) * 0.4)
    score2 = (factor2.get("ir", 0) * 0.6 + factor2.get("win_rate", 0) * 0.4)

    return factor1 if score1 > score2 else factor2


def check_regime_coverage(factors: list[dict]) -> dict[str, Any]:
    """检查市场环境覆盖"""
    from app.services.factor_taxonomy import MARKET_REGIMES

    covered_regimes = set()
    regime_factors = {regime: [] for regime in MARKET_REGIMES.keys()}

    for factor in factors:
        for regime in factor.get("regimes", []):
            covered_regimes.add(regime)
            regime_factors[regime].append(factor["name"])

    coverage_ratio = len(covered_regimes) / len(MARKET_REGIMES)

    return {
        "covered_regimes": list(covered_regimes),
        "missing_regimes": [r for r in MARKET_REGIMES.keys() if r not in covered_regimes],
        "coverage_ratio": coverage_ratio,
        "regime_factors": regime_factors,
    }


def check_category_diversity(factors: list[dict]) -> dict[str, Any]:
    """检查因子类别多样性"""
    from collections import Counter
    from app.services.factor_taxonomy import FACTOR_CATEGORIES

    category_counts = Counter()
    for factor in factors:
        for cat in factor.get("categories", []):
            category_counts[cat] += 1

    diversity_score = len(category_counts) / len(FACTOR_CATEGORIES)

    return {
        "category_counts": dict(category_counts),
        "diversity_score": diversity_score,
        "dominant_category": category_counts.most_common(1)[0][0] if category_counts else None,
    }


def generate_optimization_suggestions(
    factors: list[dict],
    high_corr_pairs: list[dict],
    regime_coverage: dict,
    category_diversity: dict
) -> list[dict]:
    """生成优化建议"""
    suggestions = []

    # 1. 处理高相关因子
    for pair in high_corr_pairs:
        better = pair["better_factor"]
        worse = pair["factor1"] if better == pair["factor2"] else pair["factor2"]

        suggestions.append({
            "type": "remove_redundant",
            "severity": "high" if pair["correlation"] > 0.9 else "medium",
            "factor": worse,
            "reason": f"与 {better['name']} 高度相关 ({pair['correlation']:.2%})",
            "action": f"移除 {worse['name']}",
            "expected_impact": "降低冗余，提高组合效率"
        })

    # 2. 处理环境覆盖不足
    if regime_coverage["missing_regimes"]:
        suggestions.append({
            "type": "add_regime_coverage",
            "severity": "high",
            "reason": f"缺少 {', '.join(regime_coverage['missing_regimes'])} 环境的因子",
            "action": "添加覆盖缺失环境的因子",
            "expected_impact": "提高环境适应性",
            "recommended_regimes": regime_coverage["missing_regimes"]
        })

    # 3. 处理类别多样性不足
    if category_diversity["diversity_score"] < 0.5:
        suggestions.append({
            "type": "improve_diversity",
            "severity": "medium",
            "reason": f"类别多样性不足 ({category_diversity['diversity_score']:.1%})",
            "action": "添加不同类型的因子",
            "expected_impact": "提高策略鲁棒性",
            "dominant_category": category_diversity["dominant_category"]
        })

    # 4. 检查因子数量
    if len(factors) < 3:
        suggestions.append({
            "type": "add_factors",
            "severity": "high",
            "reason": f"因子数量过少 ({len(factors)}个)",
            "action": "增加因子数量到至少3个",
            "expected_impact": "提高策略稳定性"
        })
    elif len(factors) > 15:
        suggestions.append({
            "type": "reduce_factors",
            "severity": "low",
            "reason": f"因子数量过多 ({len(factors)}个)",
            "action": "考虑精简到10-15个高质量因子",
            "expected_impact": "简化策略，降低维护成本"
        })

    return sorted(suggestions, key=lambda s: {"high": 3, "medium": 2, "low": 1}[s["severity"]], reverse=True)


def calculate_portfolio_score(
    factors: list[dict],
    corr_matrix: np.ndarray,
    regime_coverage: dict,
    category_diversity: dict
) -> float:
    """
    计算组合综合得分

    权重：
    - 多样化 30%
    - 环境覆盖 30%
    - 类别多样性 20%
    - 平均质量 20%
    """
    # 1. 多样化得分（相关性越低越好）
    avg_corr = np.mean(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])
    diversification_score = max(0, (1 - avg_corr) * 100)

    # 2. 环境覆盖得分
    coverage_score = regime_coverage["coverage_ratio"] * 100

    # 3. 类别多样性得分
    diversity_score = category_diversity["diversity_score"] * 100

    # 4. 平均质量得分
    avg_ir = np.mean([f.get("ir", 0) for f in factors])
    avg_wr = np.mean([f.get("win_rate", 0) for f in factors])
    quality_score = (avg_ir * 100 + avg_wr * 100) / 2

    # 综合得分
    portfolio_score = (
        0.3 * diversification_score +
        0.3 * coverage_score +
        0.2 * diversity_score +
        0.2 * quality_score
    )

    return float(portfolio_score)


def calculate_diversification_score(corr_matrix: np.ndarray) -> float:
    """计算多样化得分"""
    avg_corr = np.mean(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])
    return float((1 - avg_corr) * 100)
