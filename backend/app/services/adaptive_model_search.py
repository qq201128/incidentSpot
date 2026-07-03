"""
自适应模型搜索 - 分层搜索（粗筛 + 精调）
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.services.model_family_config import ModelFamilyTrainingConfig
from app.services.model_family_training_impl import train_model_family
from app.services.model_search_history import (
    get_best_historical_config,
    save_search_result,
)

logger = logging.getLogger(__name__)


class AdaptiveModelSearch:
    """
    自适应模型搜索器

    策略：
    1. 检查历史最优配置
    2. 粗筛阶段：大步长快速筛选
    3. 精调阶段：在最优区域密集搜索
    """

    def __init__(
        self,
        family: str,
        symbol: str,
        duration: str,
        mode: str = "balanced",  # fast / balanced / exhaustive
    ):
        self.family = family
        self.symbol = symbol
        self.duration = duration
        self.mode = mode

    async def search(self) -> dict[str, Any]:
        """执行自适应搜索"""
        logger.info(f"[{self.family}] Starting adaptive search: {self.symbol} {self.duration} (mode={self.mode})")

        # 1. 检查历史
        historical_config = get_best_historical_config(self.family, self.symbol, self.duration)

        if self.mode == "fast" and historical_config:
            logger.info(f"[{self.family}] Fast mode: using historical config + fine-tuning")
            return await self._fast_search(historical_config)

        elif self.mode == "balanced":
            logger.info(f"[{self.family}] Balanced mode: coarse + fine search")
            return await self._balanced_search(historical_config)

        else:  # exhaustive
            logger.info(f"[{self.family}] Exhaustive mode: full grid search")
            return await self._exhaustive_search()

    async def _fast_search(self, base_config: dict) -> dict[str, Any]:
        """快速搜索：历史配置 + 小范围微调"""
        # 生成微调网格（在历史最优附近小范围搜索）
        fine_configs = self._generate_fine_tuning_grid(base_config, radius=0.15)

        logger.info(f"[{self.family}] Fast search: {len(fine_configs)} configs around historical best")

        # 并行训练
        results = await self._parallel_train(fine_configs, max_workers=4)

        # 返回最优
        if not results:
            raise RuntimeError("Fast search produced no results")

        best = max(results, key=lambda r: r.get("validation_score", 0))
        self._save_best_result(best)
        return best

    async def _balanced_search(self, historical_config: dict | None) -> dict[str, Any]:
        """平衡搜索：粗筛 + 精调"""
        # Phase 1: 粗筛
        logger.info(f"[{self.family}] Phase 1: Coarse search")
        coarse_grid = self._generate_coarse_grid()

        # 如果有历史配置，加入粗筛
        if historical_config:
            coarse_grid.insert(0, historical_config)
            logger.info(f"[{self.family}] Added historical config to coarse grid")

        logger.info(f"[{self.family}] Coarse grid size: {len(coarse_grid)}")

        coarse_results = await self._parallel_train(coarse_grid, max_workers=4)

        if not coarse_results:
            raise RuntimeError("Coarse search produced no results")

        # 选择 top 3 配置
        top_k = min(3, len(coarse_results))
        top_configs = sorted(
            coarse_results,
            key=lambda r: r.get("validation_score", 0),
            reverse=True
        )[:top_k]

        logger.info(
            f"[{self.family}] Top {top_k} coarse configs: "
            f"scores={[r.get('validation_score', 0) for r in top_configs]}"
        )

        # Phase 2: 精调
        logger.info(f"[{self.family}] Phase 2: Fine-tuning top {top_k} configs")
        fine_grids = []
        for config_result in top_configs:
            fine_grid = self._generate_fine_grid(config_result["config"])
            fine_grids.extend(fine_grid)

        logger.info(f"[{self.family}] Fine grid size: {len(fine_grids)}")

        fine_results = await self._parallel_train(fine_grids, max_workers=2)

        # 合并粗筛和精调结果，选择最优
        all_results = coarse_results + fine_results
        best = max(all_results, key=lambda r: r.get("validation_score", 0))

        logger.info(
            f"[{self.family}] Best result: validation_score={best.get('validation_score', 0):.4f}, "
            f"win_rate={best.get('validation_win_rate', 0):.4f}"
        )

        self._save_best_result(best)
        return best

    async def _exhaustive_search(self) -> dict[str, Any]:
        """穷尽搜索：全量网格"""
        from app.services.model_family_search_rules import model_family_search_grid

        full_grid = model_family_search_grid(self.family)
        logger.info(f"[{self.family}] Exhaustive search: {len(full_grid)} configs")

        results = await self._parallel_train(full_grid, max_workers=4)

        if not results:
            raise RuntimeError("Exhaustive search produced no results")

        best = max(results, key=lambda r: r.get("validation_score", 0))
        self._save_best_result(best)
        return best

    def _generate_coarse_grid(self) -> list[dict]:
        """生成粗筛网格（大步长）"""
        if self.family == "lstm":
            return [
                {
                    "feature_window": w,
                    "hidden_size": h,
                    "num_layers": l,
                    "learning_rate": lr,
                    "dropout": d,
                    "epochs": e,
                }
                for w in [32, 64]              # 2 values
                for h in [32, 64, 128]         # 3 values
                for l in [1, 2]                # 2 values
                for lr in [0.001, 0.01]        # 2 values
                for d in [0.1, 0.2]            # 2 values
                for e in [8, 12]               # 2 values
            ]  # Total: 2×3×2×2×2×2 = 96 configs

        elif self.family == "lightgbm":
            return [
                {
                    "feature_window": w,
                    "num_leaves": n,
                    "learning_rate": lr,
                    "max_depth": d,
                    "min_data_in_leaf": m,
                }
                for w in [32, 64]              # 2 values
                for n in [31, 127]             # 2 values
                for lr in [0.01, 0.1]          # 2 values
                for d in [5, 10]               # 2 values
                for m in [20, 50]              # 2 values
            ]  # Total: 2×2×2×2×2 = 32 configs

        elif self.family == "xgboost":
            return [
                {
                    "feature_window": w,
                    "max_depth": d,
                    "learning_rate": lr,
                    "n_estimators": n,
                    "subsample": s,
                }
                for w in [32, 64]              # 2 values
                for d in [3, 6]                # 2 values
                for lr in [0.01, 0.1]          # 2 values
                for n in [50, 100]             # 2 values
                for s in [0.8, 1.0]            # 2 values
            ]  # Total: 2×2×2×2×2 = 32 configs

        else:
            # 默认：简化网格
            return [
                {"feature_window": w, "min_move_bps": m}
                for w in [32, 48, 64]
                for m in [8.0, 12.0, 20.0]
            ]

    def _generate_fine_grid(self, base_config: dict) -> list[dict]:
        """生成精调网格（小步长，局部密集）"""
        if self.family == "lstm":
            base_hidden = base_config.get("hidden_size", 64)
            base_layers = base_config.get("num_layers", 1)
            base_lr = base_config.get("learning_rate", 0.001)

            fine_grid = []
            for delta_h in [-16, 0, 16]:
                for delta_l in [0, 1]:
                    for lr_factor in [0.5, 1.0, 1.5]:
                        config = base_config.copy()
                        config["hidden_size"] = max(16, base_hidden + delta_h)
                        config["num_layers"] = max(1, base_layers + delta_l)
                        config["learning_rate"] = base_lr * lr_factor
                        fine_grid.append(config)

            return fine_grid[:15]  # 限制精调数量

        elif self.family in ["lightgbm", "xgboost"]:
            base_lr = base_config.get("learning_rate", 0.01)
            base_depth = base_config.get("max_depth", 5)

            fine_grid = []
            for lr_factor in [0.5, 1.0, 1.5]:
                for delta_depth in [-1, 0, 1]:
                    config = base_config.copy()
                    config["learning_rate"] = base_lr * lr_factor
                    config["max_depth"] = max(3, base_depth + delta_depth)
                    fine_grid.append(config)

            return fine_grid[:10]

        else:
            return [base_config]

    def _generate_fine_tuning_grid(self, base_config: dict, radius: float) -> list[dict]:
        """生成微调网格（极小范围）"""
        configs = [base_config]

        if self.family == "lstm":
            base_hidden = base_config.get("hidden_size", 64)
            base_lr = base_config.get("learning_rate", 0.001)

            for hidden_factor in [0.85, 1.0, 1.15]:
                for lr_factor in [0.5, 1.0, 2.0]:
                    config = base_config.copy()
                    config["hidden_size"] = int(base_hidden * hidden_factor)
                    config["learning_rate"] = base_lr * lr_factor
                    configs.append(config)

        return configs[:10]

    async def _parallel_train(
        self,
        configs: list[dict],
        max_workers: int = 4,
    ) -> list[dict]:
        """并行训练配置"""
        semaphore = asyncio.Semaphore(max_workers)

        async def train_one(config: dict) -> dict | None:
            async with semaphore:
                try:
                    training_config = self._build_training_config(config)
                    result = await asyncio.to_thread(train_model_family, training_config)
                    return {
                        "config": config,
                        "validation_score": result.get("validation_score", 0),
                        "validation_win_rate": result.get("validation_win_rate", 0),
                        "oos_win_rate": result.get("oos_win_rate"),
                        "result": result,
                    }
                except Exception as exc:
                    logger.error(f"[{self.family}] Training failed for config {config}: {exc}")
                    return None

        tasks = [train_one(cfg) for cfg in configs]
        results = await asyncio.gather(*tasks)

        return [r for r in results if r is not None]

    def _build_training_config(self, params: dict) -> ModelFamilyTrainingConfig:
        """构建训练配置"""
        return ModelFamilyTrainingConfig(
            family=self.family,
            symbol=self.symbol,
            duration=self.duration,
            **params
        )

    def _save_best_result(self, result: dict) -> None:
        """保存最优结果到历史"""
        save_search_result(
            family=self.family,
            symbol=self.symbol,
            duration=self.duration,
            config=result["config"],
            metrics={
                "validation_score": result.get("validation_score", 0),
                "validation_win_rate": result.get("validation_win_rate", 0),
                "oos_win_rate": result.get("oos_win_rate"),
            }
        )
