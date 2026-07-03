"""
因子库管理API - 集成所有优化功能
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.services.factor_similarity_detector import check_factor_before_add, detect_similar_factors
from app.services.factor_performance_monitor import get_performance_monitor
from app.services.factor_taxonomy import (
    auto_classify_factor,
    get_factor_taxonomy_summary,
    search_factors_by_tags,
)
from app.services.factor_portfolio_optimizer import optimize_factor_portfolio
from app.services.factor_genetic_evolution import evolve_factors_from_best

router = APIRouter(prefix="/api/factors/library", tags=["factors"])


@router.post("/check-similarity")
def check_factor_similarity(
    factor: dict,
    existing_factors: list[dict] | None = None
) -> dict:
    """
    检查因子相似度（添加前检查）

    Args:
        factor: 待检查的因子
        existing_factors: 现有因子列表（如果为None，从数据库加载）

    Returns:
        相似度检查结果
    """
    if existing_factors is None:
        # TODO: 从数据库加载现有因子
        existing_factors = []

    try:
        result = check_factor_before_add(factor, existing_factors)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/classify")
def classify_factor(factor: dict) -> dict:
    """
    自动分类因子

    Returns:
        带分类标签的因子
    """
    try:
        classified = auto_classify_factor(factor)
        return classified
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/taxonomy-summary")
def get_taxonomy_summary() -> dict:
    """
    获取因子库分类汇总

    Returns:
        分类统计信息
    """
    try:
        # TODO: 从数据库加载所有因子
        factors = []
        summary = get_factor_taxonomy_summary(factors)
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search-by-tags")
def search_by_tags(
    required_tags: list[str] | None = None,
    excluded_tags: list[str] | None = None,
    categories: list[str] | None = None,
    regimes: list[str] | None = None,
) -> dict:
    """
    按标签搜索因子

    Returns:
        匹配的因子列表
    """
    try:
        # TODO: 从数据库加载所有因子
        all_factors = []

        filtered = search_factors_by_tags(
            all_factors,
            required_tags=required_tags,
            excluded_tags=excluded_tags,
            categories=categories,
            regimes=regimes,
        )

        return {
            "total": len(filtered),
            "factors": filtered
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance-trends")
def get_performance_trends() -> dict:
    """
    获取因子性能趋势

    Returns:
        改进和衰减的因子列表
    """
    try:
        monitor = get_performance_monitor()
        trends = monitor.get_trending_factors()
        return trends
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health/{factor_id}")
def get_factor_health(factor_id: str) -> dict:
    """
    获取因子健康度评分

    Returns:
        健康度评分和详情
    """
    try:
        monitor = get_performance_monitor()
        health_score = monitor.get_factor_health_score(factor_id)
        degradation = monitor.check_degradation(factor_id)

        return {
            "factor_id": factor_id,
            "health_score": health_score,
            "degradation_warning": degradation,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize-portfolio")
def optimize_portfolio(
    factor_ids: list[str] | None = None,
    max_correlation: float = Query(0.7, ge=0, le=1)
) -> dict:
    """
    优化因子组合

    Args:
        factor_ids: 因子ID列表（如果为None，使用所有活跃因子）
        max_correlation: 最大允许相关性

    Returns:
        优化建议
    """
    try:
        # TODO: 从数据库加载因子
        if factor_ids:
            factors = []  # 根据IDs加载
        else:
            factors = []  # 加载所有活跃因子

        result = optimize_factor_portfolio(
            factors,
            max_correlation=max_correlation
        )

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evolve")
def evolve_new_factors(
    top_k: int = Query(10, ge=3, le=20),
    generations: int = Query(10, ge=5, le=50)
) -> dict:
    """
    通过遗传算法进化新因子

    Args:
        top_k: 选择前k个最优因子作为父代
        generations: 进化代数

    Returns:
        进化后的因子列表
    """
    try:
        # TODO: 从数据库加载所有因子
        all_factors = []

        evolved_factors = evolve_factors_from_best(
            all_factors,
            top_k=top_k,
            generations=generations
        )

        return {
            "status": "success",
            "generations": generations,
            "parent_count": top_k,
            "evolved_count": len(evolved_factors),
            "evolved_factors": evolved_factors
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
