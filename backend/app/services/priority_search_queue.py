"""
模型搜索优先级队列 - 智能调度不同速度的模型
"""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """任务优先级"""
    HIGH = "high"      # 历史最优配置
    MEDIUM = "medium"  # 粗筛配置
    LOW = "low"        # 精调配置


class ModelSpeed(Enum):
    """模型训练速度"""
    FAST = "fast"      # <10分钟：LightGBM, LogisticRegression
    MEDIUM = "medium"  # 10-30分钟：XGBoost, RandomForest
    SLOW = "slow"      # >30分钟：LSTM, GRU, Transformer


@dataclass
class SearchTask:
    """搜索任务"""
    family: str
    symbol: str
    duration: str
    config: dict[str, Any]
    priority: TaskPriority
    estimated_time: float  # 秒


class PriorityModelSearchQueue:
    """
    优先级模型搜索队列

    策略：
    1. 快速模型优先（高吞吐）
    2. 历史最优配置优先
    3. 不同速度的模型并行执行
    """

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.queues = {
            TaskPriority.HIGH: [],
            TaskPriority.MEDIUM: [],
            TaskPriority.LOW: [],
        }
        self.speed_queues = {
            ModelSpeed.FAST: [],
            ModelSpeed.MEDIUM: [],
            ModelSpeed.SLOW: [],
        }

    def enqueue(self, task: SearchTask) -> None:
        """入队"""
        self.queues[task.priority].append(task)

        # 同时按速度分类
        speed = self._classify_speed(task)
        self.speed_queues[speed].append(task)

        logger.info(
            f"Enqueued {task.family} task: "
            f"priority={task.priority.value}, speed={speed.value}, "
            f"estimated={task.estimated_time:.0f}s"
        )

    def _classify_speed(self, task: SearchTask) -> ModelSpeed:
        """分类模型速度"""
        # 基础时间
        base_times = {
            "lightgbm": 300,        # 5分钟
            "logistic_elasticnet": 60,
            "random_forest": 600,   # 10分钟
            "extra_trees": 600,
            "xgboost": 900,         # 15分钟
            "catboost": 1200,       # 20分钟
            "lstm": 1800,           # 30分钟
            "gru": 1500,
            "cnn": 1200,
            "transformer": 2400,    # 40分钟
        }

        base_time = base_times.get(task.family, 600)

        # 根据配置调整
        if task.family in ["lstm", "gru", "transformer"]:
            epochs = task.config.get("epochs", 10)
            hidden_size = task.config.get("hidden_size", 64)
            base_time = base_time * (epochs / 10) * (hidden_size / 64)

        if base_time < 600:
            return ModelSpeed.FAST
        elif base_time < 1800:
            return ModelSpeed.MEDIUM
        else:
            return ModelSpeed.SLOW

    async def process(self) -> list[dict[str, Any]]:
        """
        处理队列

        策略：
        1. 优先处理高优先级任务
        2. 快速任务高并发
        3. 慢速任务低并发
        """
        all_results = []

        # Phase 1: 高优先级（历史最优）
        high_tasks = self.queues[TaskPriority.HIGH]
        if high_tasks:
            logger.info(f"Processing {len(high_tasks)} HIGH priority tasks")
            high_results = await self._process_batch(
                high_tasks,
                max_workers=self.max_workers  # 全力处理
            )
            all_results.extend(high_results)

            # 如果历史配置已经很好，直接返回
            best_score = max((r.get("validation_score", 0) for r in high_results), default=0)
            if best_score > 0.6:
                logger.info(f"Historical config good enough: {best_score:.4f}, skipping other phases")
                return all_results

        # Phase 2: 按速度并行处理
        logger.info("Processing MEDIUM and LOW priority tasks by speed")

        # 快速任务：高并发
        fast_tasks = [t for t in self.speed_queues[ModelSpeed.FAST] if t.priority != TaskPriority.HIGH]
        if fast_tasks:
            logger.info(f"Processing {len(fast_tasks)} FAST tasks (parallel={self.max_workers})")
            fast_results = await self._process_batch(fast_tasks, max_workers=self.max_workers)
            all_results.extend(fast_results)

        # 中速任务：中等并发
        medium_tasks = [t for t in self.speed_queues[ModelSpeed.MEDIUM] if t.priority != TaskPriority.HIGH]
        if medium_tasks:
            logger.info(f"Processing {len(medium_tasks)} MEDIUM tasks (parallel={self.max_workers // 2})")
            medium_results = await self._process_batch(
                medium_tasks,
                max_workers=max(1, self.max_workers // 2)
            )
            all_results.extend(medium_results)

        # 慢速任务：低并发
        slow_tasks = [t for t in self.speed_queues[ModelSpeed.SLOW] if t.priority != TaskPriority.HIGH]
        if slow_tasks:
            logger.info(f"Processing {len(slow_tasks)} SLOW tasks (parallel=1)")
            slow_results = await self._process_batch(slow_tasks, max_workers=1)
            all_results.extend(slow_results)

        return all_results

    async def _process_batch(
        self,
        tasks: list[SearchTask],
        max_workers: int,
    ) -> list[dict[str, Any]]:
        """批量处理任务"""
        semaphore = asyncio.Semaphore(max_workers)

        async def process_one(task: SearchTask) -> dict[str, Any] | None:
            async with semaphore:
                try:
                    logger.info(f"Training {task.family} {task.symbol} {task.duration}")
                    result = await self._train_model(task)
                    logger.info(
                        f"Completed {task.family}: "
                        f"score={result.get('validation_score', 0):.4f}"
                    )
                    return result
                except Exception as exc:
                    logger.error(f"Training failed for {task.family}: {exc}")
                    return None

        results = await asyncio.gather(*[process_one(t) for t in tasks])
        return [r for r in results if r is not None]

    async def _train_model(self, task: SearchTask) -> dict[str, Any]:
        """训练模型"""
        from app.services.model_family_config import ModelFamilyTrainingConfig
        from app.services.model_family_training_impl import train_model_family

        config = ModelFamilyTrainingConfig(
            family=task.family,
            symbol=task.symbol,
            duration=task.duration,
            **task.config
        )

        result = await asyncio.to_thread(train_model_family, config)

        return {
            "task": task,
            "config": task.config,
            "validation_score": result.get("validation_score", 0),
            "validation_win_rate": result.get("validation_win_rate", 0),
            "result": result,
        }

    def get_queue_summary(self) -> dict[str, Any]:
        """获取队列摘要"""
        return {
            "by_priority": {
                priority.value: len(tasks)
                for priority, tasks in self.queues.items()
            },
            "by_speed": {
                speed.value: len(tasks)
                for speed, tasks in self.speed_queues.items()
            },
            "total": sum(len(tasks) for tasks in self.queues.values()),
        }


def estimate_task_time(family: str, config: dict) -> float:
    """估算任务耗时（秒）"""
    base_times = {
        "lightgbm": 300,
        "xgboost": 900,
        "lstm": 1800,
        "gru": 1500,
        "transformer": 2400,
        "random_forest": 600,
        "catboost": 1200,
    }

    base = base_times.get(family, 600)

    # 根据配置调整
    if family in ["lstm", "gru", "transformer"]:
        epochs = config.get("epochs", 10)
        base = base * (epochs / 10)

    return base
