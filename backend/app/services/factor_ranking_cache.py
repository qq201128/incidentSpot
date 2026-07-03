"""
因子排名缓存服务 - 避免重复计算
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Any


class FactorRankingCache:
    """
    因子排名缓存

    策略：
    - 缓存时长：5分钟
    - 新K线到达时自动失效
    - 按标的和周期分别缓存
    """

    def __init__(self, ttl_seconds: int = 300):
        self.cache: dict[str, dict[str, Any]] = {}
        self.ttl_seconds = ttl_seconds

    def get(self, symbol: str, duration: str) -> dict | None:
        """获取缓存的排名数据"""
        key = self._make_key(symbol, duration)
        cached = self.cache.get(key)

        if not cached:
            return None

        # 检查是否过期
        age = time.time() - cached["timestamp"]
        if age > self.ttl_seconds:
            del self.cache[key]
            return None

        return cached["data"]

    def set(self, symbol: str, duration: str, data: dict) -> None:
        """缓存排名数据"""
        key = self._make_key(symbol, duration)
        self.cache[key] = {
            "data": data,
            "timestamp": time.time(),
        }

    def invalidate(self, symbol: str | None = None, duration: str | None = None) -> None:
        """
        失效缓存

        Args:
            symbol: 指定标的（None表示全部）
            duration: 指定周期（None表示全部）
        """
        if symbol is None and duration is None:
            # 清除全部
            self.cache.clear()
            return

        if symbol and duration:
            # 清除特定缓存
            key = self._make_key(symbol, duration)
            self.cache.pop(key, None)
        elif symbol:
            # 清除该标的的所有周期
            keys_to_remove = [k for k in self.cache.keys() if k.startswith(f"{symbol}_")]
            for key in keys_to_remove:
                del self.cache[key]
        elif duration:
            # 清除该周期的所有标的
            keys_to_remove = [k for k in self.cache.keys() if k.endswith(f"_{duration}")]
            for key in keys_to_remove:
                del self.cache[key]

    def get_stats(self) -> dict:
        """获取缓存统计"""
        now = time.time()
        valid_count = sum(1 for entry in self.cache.values() if now - entry["timestamp"] <= self.ttl_seconds)

        return {
            "total_entries": len(self.cache),
            "valid_entries": valid_count,
            "expired_entries": len(self.cache) - valid_count,
            "ttl_seconds": self.ttl_seconds,
        }

    def cleanup_expired(self) -> int:
        """清理过期缓存，返回清理数量"""
        now = time.time()
        expired_keys = [
            key for key, entry in self.cache.items()
            if now - entry["timestamp"] > self.ttl_seconds
        ]

        for key in expired_keys:
            del self.cache[key]

        return len(expired_keys)

    def _make_key(self, symbol: str, duration: str) -> str:
        """生成缓存键"""
        return f"{symbol.upper()}_{duration}"


# 全局实例
_ranking_cache = FactorRankingCache(ttl_seconds=300)


def get_ranking_cache() -> FactorRankingCache:
    """获取全局缓存实例"""
    return _ranking_cache


def invalidate_ranking_cache_on_new_kline(symbol: str, duration: str) -> None:
    """
    新K线到达时失效缓存

    应在K线接收/存储后调用
    """
    cache = get_ranking_cache()
    cache.invalidate(symbol, duration)
