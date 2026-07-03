"""
因子性能监控器 - 追踪性能衰减
"""
from __future__ import annotations

import json
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np


class FactorPerformanceMonitor:
    """
    因子性能监控器

    追踪：
    - 滚动窗口胜率
    - 近期vs历史对比
    - 衰减趋势检测
    """

    def __init__(self, window_days: int = 90, storage_path: Path | None = None):
        """
        Args:
            window_days: 滚动窗口天数
            storage_path: 持久化存储路径
        """
        self.window_days = window_days
        self.storage_path = storage_path or Path("backend/data/factor_performance_history.json")
        self.performance_history: dict[str, deque] = {}
        self._load_history()

    def update_performance(
        self,
        factor_id: str,
        date: datetime,
        win_rate: float,
        trades: int = 0,
        pnl: float = 0.0
    ) -> None:
        """
        更新性能记录

        Args:
            factor_id: 因子ID
            date: 日期
            win_rate: 胜率 [0, 1]
            trades: 交易次数
            pnl: 损益
        """
        if factor_id not in self.performance_history:
            self.performance_history[factor_id] = deque(maxlen=self.window_days)

        self.performance_history[factor_id].append({
            "date": date.isoformat(),
            "win_rate": float(win_rate),
            "trades": int(trades),
            "pnl": float(pnl)
        })

        # 定期持久化
        if len(self.performance_history[factor_id]) % 10 == 0:
            self._save_history()

    def check_degradation(self, factor_id: str, threshold: float = 0.05) -> dict[str, Any] | None:
        """
        检测性能衰减

        Args:
            factor_id: 因子ID
            threshold: 衰减阈值（默认5%）

        Returns:
            如果检测到衰减，返回警告信息
        """
        history = self.performance_history.get(factor_id)
        if not history or len(history) < 30:
            return None

        # 分割历史数据：最近30天 vs 更早的记录
        recent_days = 30
        recent = list(history)[-recent_days:]
        older = list(history)[:-recent_days]

        if not older:
            return None

        recent_wr = np.mean([r["win_rate"] for r in recent])
        older_wr = np.mean([r["win_rate"] for r in older])

        degradation = older_wr - recent_wr

        if degradation > threshold:
            severity = "critical" if degradation > 0.15 else "high" if degradation > 0.10 else "medium"

            return {
                "factor_id": factor_id,
                "severity": severity,
                "recent_win_rate": float(recent_wr),
                "historical_win_rate": float(older_wr),
                "degradation": float(degradation),
                "recent_trades": sum(r["trades"] for r in recent),
                "message": f"因子胜率下降 {degradation*100:.1f}%，建议复查",
                "recommendation": self._get_degradation_recommendation(severity, degradation)
            }

        return None

    def get_trending_factors(self, min_improvement: float = 0.03) -> dict[str, list]:
        """
        获取表现趋势（改进和衰减的因子）

        Args:
            min_improvement: 最小改进幅度

        Returns:
            {improving: [...], degrading: [...]}
        """
        improving = []
        degrading = []

        for factor_id in self.performance_history.keys():
            warning = self.check_degradation(factor_id)

            if warning:
                degrading.append(warning)
            else:
                # 检查是否在改进
                improvement = self._check_improvement(factor_id)
                if improvement and improvement > min_improvement:
                    history = self.performance_history[factor_id]
                    recent = list(history)[-30:]
                    older = list(history)[:-30]

                    improving.append({
                        "factor_id": factor_id,
                        "recent_win_rate": float(np.mean([r["win_rate"] for r in recent])),
                        "improvement": float(improvement),
                        "recent_trades": sum(r["trades"] for r in recent)
                    })

        return {
            "improving": sorted(improving, key=lambda x: x["improvement"], reverse=True),
            "degrading": sorted(degrading, key=lambda x: x["degradation"], reverse=True)
        }

    def get_factor_health_score(self, factor_id: str) -> float:
        """
        计算因子健康度评分

        Returns:
            健康度评分 [0, 100]
        """
        history = self.performance_history.get(factor_id)
        if not history or len(history) < 10:
            return 50.0  # 数据不足，返回中性评分

        recent = list(history)[-30:]

        # 1. 胜率得分（40%权重）
        avg_wr = np.mean([r["win_rate"] for r in recent])
        wr_score = min(100, avg_wr * 100 * 1.5)  # 60%胜率 = 90分

        # 2. 稳定性得分（30%权重）
        wr_std = np.std([r["win_rate"] for r in recent])
        stability_score = max(0, 100 - wr_std * 500)  # 波动越小越好

        # 3. 趋势得分（30%权重）
        degradation = self.check_degradation(factor_id)
        if degradation:
            trend_score = max(0, 100 - degradation["degradation"] * 500)
        else:
            improvement = self._check_improvement(factor_id) or 0
            trend_score = min(100, 50 + improvement * 500)

        # 综合评分
        health_score = 0.4 * wr_score + 0.3 * stability_score + 0.3 * trend_score

        return float(health_score)

    def _check_improvement(self, factor_id: str) -> float | None:
        """检查是否有改进"""
        history = self.performance_history.get(factor_id)
        if not history or len(history) < 30:
            return None

        recent = list(history)[-30:]
        older = list(history)[:-30]

        if not older:
            return None

        recent_wr = np.mean([r["win_rate"] for r in recent])
        older_wr = np.mean([r["win_rate"] for r in older])

        improvement = recent_wr - older_wr

        return improvement if improvement > 0 else None

    def _get_degradation_recommendation(self, severity: str, degradation: float) -> str:
        """根据严重程度给出建议"""
        if severity == "critical":
            return "立即停用，进行深度分析"
        elif severity == "high":
            return "暂停使用，检查市场环境变化"
        else:
            return "密切监控，考虑调整参数"

    def _load_history(self) -> None:
        """从文件加载历史数据"""
        if not self.storage_path.exists():
            return

        try:
            with open(self.storage_path) as f:
                data = json.load(f)

            for factor_id, records in data.items():
                self.performance_history[factor_id] = deque(records, maxlen=self.window_days)
        except Exception as e:
            print(f"Failed to load performance history: {e}")

    def _save_history(self) -> None:
        """保存历史数据到文件"""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            data = {
                factor_id: list(records)
                for factor_id, records in self.performance_history.items()
            }

            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Failed to save performance history: {e}")


# 全局实例
_monitor = FactorPerformanceMonitor()


def get_performance_monitor() -> FactorPerformanceMonitor:
    """获取全局监控器实例"""
    return _monitor
