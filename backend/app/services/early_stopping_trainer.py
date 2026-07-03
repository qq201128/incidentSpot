"""
提前停止训练器 - 避免浪费时间在差配置上
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class EarlyStoppingTrainer:
    """
    提前停止训练器

    策略：
    1. 前 20% epoch 快速淘汰明显差的配置
    2. 持续不改进则提前停止
    3. 保存最优模型状态
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        quick_reject_ratio: float = 0.2,
        quick_reject_threshold: float = 1.5,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.quick_reject_ratio = quick_reject_ratio
        self.quick_reject_threshold = quick_reject_threshold

        self.best_val_loss = float('inf')
        self.best_val_score = 0.0
        self.patience_counter = 0
        self.best_epoch = 0
        self.baseline_loss = None

    def should_stop(
        self,
        epoch: int,
        max_epochs: int,
        val_loss: float,
        val_score: float = 0.0,
    ) -> tuple[bool, str]:
        """
        判断是否应该停止训练

        Returns:
            (should_stop, reason)
        """
        # 1. 快速淘汰：前 20% 训练就很差
        if self.baseline_loss is not None:
            quick_reject_epoch = int(max_epochs * self.quick_reject_ratio)
            if epoch == quick_reject_epoch:
                if val_loss > self.baseline_loss * self.quick_reject_threshold:
                    logger.info(
                        f"Quick reject at epoch {epoch}: "
                        f"val_loss={val_loss:.4f} > baseline={self.baseline_loss:.4f} * {self.quick_reject_threshold}"
                    )
                    return True, "quick_reject"

        # 2. 检查改进
        improved = False
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.best_val_score = val_score
            self.best_epoch = epoch
            self.patience_counter = 0
            improved = True
        else:
            self.patience_counter += 1

        # 3. 耐心耗尽
        if self.patience_counter >= self.patience:
            logger.info(
                f"Early stop at epoch {epoch}: no improvement for {self.patience} epochs "
                f"(best at epoch {self.best_epoch})"
            )
            return True, "patience_exhausted"

        return False, ""

    def set_baseline_loss(self, loss: float) -> None:
        """设置基线损失（用于快速淘汰）"""
        self.baseline_loss = loss

    def get_best_metrics(self) -> dict[str, Any]:
        """获取最优指标"""
        return {
            "best_val_loss": self.best_val_loss,
            "best_val_score": self.best_val_score,
            "best_epoch": self.best_epoch,
            "stopped_early": self.patience_counter >= self.patience,
        }


def compute_baseline_loss(family: str, symbol: str, duration: str) -> float | None:
    """
    计算基线损失（用于快速淘汰）

    基线可以是：
    1. 历史最优模型的验证损失
    2. 简单基准模型的损失
    3. 随机预测的损失
    """
    from app.services.model_search_history import get_historical_performance

    performance = get_historical_performance(family, symbol, duration)

    if performance and performance.get("validation_score"):
        # 历史最优的损失作为基线
        # 假设 score 越高越好，loss 越低越好
        # 这里简化处理，实际可能需要更复杂的转换
        baseline_score = performance["validation_score"]
        return 1.0 - baseline_score  # 转换为损失

    # 默认基线：随机预测的交叉熵损失
    return 0.693  # -log(0.5) for binary classification
