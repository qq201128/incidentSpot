"""
因子排名过滤和排序工具
"""
from __future__ import annotations

from typing import Any, Literal


SortKey = Literal["ir", "winRate", "sharpe", "trades", "recentWinRate"]


def filter_ranking(
    ranking: list[dict[str, Any]],
    search: str | None = None,
    min_win_rate: float | None = None,
    min_ir: float | None = None,
    regime: str | None = None,
) -> list[dict[str, Any]]:
    """
    过滤因子排名

    Args:
        ranking: 排名列表
        search: 搜索关键词
        min_win_rate: 最低胜率（0-100）
        min_ir: 最低IR
        regime: 市场环境过滤 (trending/ranging)

    Returns:
        过滤后的排名列表
    """
    filtered = ranking

    # 搜索过滤
    if search and search.strip():
        search_lower = search.strip().lower()
        filtered = [
            row for row in filtered
            if _matches_search(row, search_lower)
        ]

    # 胜率过滤
    if min_win_rate is not None:
        filtered = [
            row for row in filtered
            if _get_win_rate(row) >= min_win_rate
        ]

    # IR过滤
    if min_ir is not None:
        filtered = [
            row for row in filtered
            if _get_ir(row) >= min_ir
        ]

    # 环境过滤
    if regime and regime != "all":
        filtered = [
            row for row in filtered
            if _matches_regime(row, regime)
        ]

    return filtered


def sort_ranking(
    ranking: list[dict[str, Any]],
    sort_by: SortKey = "ir",
    ascending: bool = False,
) -> list[dict[str, Any]]:
    """
    排序因子排名

    Args:
        ranking: 排名列表
        sort_by: 排序字段
        ascending: 是否升序（默认降序）

    Returns:
        排序后的排名列表
    """
    if sort_by == "ir":
        key_func = _get_ir
    elif sort_by == "winRate":
        key_func = _get_win_rate
    elif sort_by == "sharpe":
        key_func = _get_sharpe
    elif sort_by == "trades":
        key_func = _get_trades
    elif sort_by == "recentWinRate":
        key_func = _get_recent_win_rate
    else:
        key_func = _get_ir

    return sorted(ranking, key=key_func, reverse=not ascending)


def _matches_search(row: dict[str, Any], search: str) -> bool:
    """检查是否匹配搜索关键词"""
    searchable = " ".join([
        str(row.get("factorName") or ""),
        str(row.get("factorDisplayName") or ""),
        str(row.get("description") or ""),
        _get_members_text(row.get("members")),
    ]).lower()

    return search in searchable


def _get_members_text(members: Any) -> str:
    """获取成员文本"""
    if not isinstance(members, list):
        return ""

    return " ".join(
        f"{m.get('name', '')} {m.get('displayName', '')}"
        for m in members
        if isinstance(m, dict)
    )


def _matches_regime(row: dict[str, Any], regime: str) -> bool:
    """检查是否匹配市场环境"""
    # 检查因子的环境标签
    regime_tags = row.get("regimeTags") or []
    if regime == "trending":
        return "trend" in regime_tags or "trending" in regime_tags
    elif regime == "ranging":
        return "range" in regime_tags or "ranging" in regime_tags

    return True


def _get_win_rate(row: dict[str, Any]) -> float:
    """获取胜率"""
    return float(row.get("winRate") or 0)


def _get_ir(row: dict[str, Any]) -> float:
    """获取IR"""
    return float(row.get("ir") or 0)


def _get_sharpe(row: dict[str, Any]) -> float:
    """获取夏普比率"""
    return float(row.get("sharpe") or 0)


def _get_trades(row: dict[str, Any]) -> int:
    """获取交易次数"""
    return int(row.get("trades") or 0)


def _get_recent_win_rate(row: dict[str, Any]) -> float:
    """获取最近胜率"""
    return float(row.get("recentWinRate") or row.get("winRate") or 0)
