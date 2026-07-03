"""
模型搜索历史管理 - 保存和复用最优配置
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HISTORY_FILE = Path(__file__).resolve().parents[1] / "models" / "search_history.json"


def save_search_result(
    family: str,
    symbol: str,
    duration: str,
    config: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    """保存搜索结果到历史"""
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

    history = load_search_history()

    record = {
        "family": family,
        "symbol": symbol,
        "duration": duration,
        "config": config,
        "metrics": {
            "validation_win_rate": metrics.get("validation_win_rate", 0),
            "validation_score": metrics.get("validation_score", 0),
            "oos_win_rate": metrics.get("oos_win_rate"),
            "test_win_rate": metrics.get("test_win_rate"),
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # 移除旧记录（相同 family + symbol + duration）
    history = [
        h for h in history
        if not (h["family"] == family and h["symbol"] == symbol and h["duration"] == duration)
    ]

    history.append(record)

    # 保存
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def load_search_history() -> list[dict[str, Any]]:
    """加载搜索历史"""
    if not HISTORY_FILE.exists():
        return []

    try:
        with open(HISTORY_FILE) as f:
            return json.load(f)
    except Exception:
        return []


def get_best_historical_config(
    family: str,
    symbol: str,
    duration: str,
) -> dict[str, Any] | None:
    """获取历史最优配置"""
    history = load_search_history()

    # 1. 精确匹配（相同标的+周期）
    exact_matches = [
        h for h in history
        if h["family"] == family and h["symbol"] == symbol and h["duration"] == duration
    ]

    if exact_matches:
        # 按验证得分排序
        best = max(exact_matches, key=lambda h: h["metrics"].get("validation_score", 0))
        return best["config"]

    # 2. 相同标的但不同周期
    same_symbol = [
        h for h in history
        if h["family"] == family and h["symbol"] == symbol
    ]

    if same_symbol:
        best = max(same_symbol, key=lambda h: h["metrics"].get("validation_score", 0))
        return best["config"]

    # 3. 相同模型族但不同标的
    same_family = [
        h for h in history
        if h["family"] == family
    ]

    if same_family:
        best = max(same_family, key=lambda h: h["metrics"].get("validation_score", 0))
        return best["config"]

    return None


def get_historical_performance(
    family: str,
    symbol: str,
    duration: str,
) -> dict[str, float] | None:
    """获取历史表现数据"""
    history = load_search_history()

    matches = [
        h for h in history
        if h["family"] == family and h["symbol"] == symbol and h["duration"] == duration
    ]

    if not matches:
        return None

    best = max(matches, key=lambda h: h["metrics"].get("validation_score", 0))
    return best["metrics"]
