"""
因子相似度检测器 - 避免重复因子入库
"""
from __future__ import annotations

import numpy as np
from difflib import SequenceMatcher
from typing import Any


def detect_similar_factors(
    new_factor: dict[str, Any],
    existing_factors: list[dict[str, Any]],
    formula_threshold: float = 0.8,
    signal_threshold: float = 0.9,
) -> list[dict[str, Any]]:
    """
    检测相似因子

    策略：
    1. 公式文本相似度（编辑距离）
    2. 回测信号相关性

    Args:
        new_factor: 新因子
        existing_factors: 现有因子列表
        formula_threshold: 公式相似度阈值
        signal_threshold: 信号相关性阈值

    Returns:
        相似因子列表
    """
    similar = []

    for existing in existing_factors:
        # 1. 公式相似度
        formula_sim = compute_formula_similarity(
            new_factor.get("formula", ""),
            existing.get("formula", "")
        )

        # 2. 信号相关性（如果有回测数据）
        signal_corr = 0.0
        if "backtest_signals" in new_factor and "backtest_signals" in existing:
            signal_corr = compute_signal_correlation(
                new_factor["backtest_signals"],
                existing["backtest_signals"]
            )

        # 3. 判断是否相似
        if formula_sim > formula_threshold or signal_corr > signal_threshold:
            recommendation = _get_similarity_recommendation(formula_sim, signal_corr)

            similar.append({
                "existing_factor": existing,
                "formula_similarity": formula_sim,
                "signal_correlation": signal_corr,
                "recommendation": recommendation,
                "reason": _get_similarity_reason(formula_sim, signal_corr, recommendation)
            })

    return similar


def compute_formula_similarity(formula1: str, formula2: str) -> float:
    """
    计算公式文本相似度（基于编辑距离）

    Returns:
        相似度 [0, 1]，1表示完全相同
    """
    if not formula1 or not formula2:
        return 0.0

    # 标准化：移除空格、统一大小写
    f1 = formula1.replace(" ", "").lower()
    f2 = formula2.replace(" ", "").lower()

    # 完全相同
    if f1 == f2:
        return 1.0

    # 使用SequenceMatcher计算相似度
    return SequenceMatcher(None, f1, f2).ratio()


def compute_signal_correlation(signals1: list, signals2: list) -> float:
    """
    计算信号相关性

    Args:
        signals1: 因子1的信号序列
        signals2: 因子2的信号序列

    Returns:
        相关系数绝对值 [0, 1]
    """
    if not signals1 or not signals2 or len(signals1) != len(signals2):
        return 0.0

    try:
        # 转换为numpy数组
        s1 = np.array(signals1, dtype=float)
        s2 = np.array(signals2, dtype=float)

        # 计算皮尔逊相关系数
        corr = np.corrcoef(s1, s2)[0, 1]

        # 返回绝对值（正相关和负相关都算相似）
        return abs(corr) if not np.isnan(corr) else 0.0
    except Exception:
        return 0.0


def _get_similarity_recommendation(formula_sim: float, signal_corr: float) -> str:
    """
    给出相似度建议

    Returns:
        - duplicate: 重复，建议不入库
        - similar: 相似，建议作为变体版本
        - distinct: 不同，可以独立入库
    """
    if formula_sim > 0.95 and signal_corr > 0.95:
        return "duplicate"
    elif formula_sim > 0.8 or signal_corr > 0.9:
        return "similar"
    else:
        return "distinct"


def _get_similarity_reason(formula_sim: float, signal_corr: float, recommendation: str) -> str:
    """生成相似性原因说明"""
    if recommendation == "duplicate":
        return f"公式相似度 {formula_sim:.2%}，信号相关性 {signal_corr:.2%}，高度重复"
    elif recommendation == "similar":
        if formula_sim > 0.8:
            return f"公式相似度 {formula_sim:.2%}，可能是变体版本"
        else:
            return f"信号相关性 {signal_corr:.2%}，行为模式相似"
    else:
        return "与现有因子明显不同"


def check_factor_before_add(
    factor: dict[str, Any],
    existing_factors: list[dict[str, Any]]
) -> dict[str, Any]:
    """
    添加因子前的检查

    Returns:
        检查结果，包含 status 和建议
    """
    similar = detect_similar_factors(factor, existing_factors)

    if not similar:
        return {
            "status": "ok",
            "message": "因子独特，可以入库",
            "similar_factors": []
        }

    # 检查是否有重复
    duplicates = [s for s in similar if s["recommendation"] == "duplicate"]
    if duplicates:
        return {
            "status": "duplicate",
            "message": f"因子与 {duplicates[0]['existing_factor'].get('name', 'unknown')} 高度重复",
            "similar_factors": duplicates,
            "suggestion": "建议不入库，或替换现有因子"
        }

    # 检查是否有相似
    variants = [s for s in similar if s["recommendation"] == "similar"]
    if variants:
        return {
            "status": "similar",
            "message": f"因子与 {len(variants)} 个现有因子相似",
            "similar_factors": variants,
            "suggestion": "建议作为变体版本入库"
        }

    return {
        "status": "ok",
        "message": "因子可以入库",
        "similar_factors": similar
    }
