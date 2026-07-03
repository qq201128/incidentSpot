"""
智能模型搜索入口 - 整合所有优化策略
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.adaptive_model_search import AdaptiveModelSearch
from app.services.priority_search_queue import (
    PriorityModelSearchQueue,
    SearchTask,
    TaskPriority,
    estimate_task_time,
)
from app.services.model_search_history import get_best_historical_config

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


async def smart_model_search(
    family: str,
    symbol: str,
    duration: str,
    mode: str = "balanced",
) -> dict:
    """
    智能模型搜索入口

    Args:
        family: 模型族（lstm, lightgbm, xgboost等）
        symbol: 交易对
        duration: 周期
        mode: 搜索模式
            - fast: 历史配置 + 微调（10-30分钟）
            - balanced: 粗筛 + 精调（30-90分钟）
            - exhaustive: 全量网格（2-5小时）

    Returns:
        最优模型结果
    """
    logger.info("=" * 80)
    logger.info(f"Smart Model Search: {family} | {symbol} | {duration} | mode={mode}")
    logger.info("=" * 80)

    searcher = AdaptiveModelSearch(family, symbol, duration, mode=mode)
    result = await searcher.search()

    logger.info("=" * 80)
    logger.info(f"Search completed!")
    logger.info(f"  Validation Score: {result.get('validation_score', 0):.4f}")
    logger.info(f"  Validation Win Rate: {result.get('validation_win_rate', 0):.4f}")
    logger.info(f"  Best Config: {result.get('config')}")
    logger.info("=" * 80)

    return result


async def batch_model_search(
    families: list[str],
    symbols: list[str],
    durations: list[str],
    mode: str = "balanced",
    max_workers: int = 4,
) -> list[dict]:
    """
    批量模型搜索（优先级队列调度）

    Args:
        families: 模型族列表
        symbols: 交易对列表
        durations: 周期列表
        mode: 搜索模式
        max_workers: 最大并行数

    Returns:
        所有搜索结果
    """
    logger.info("=" * 80)
    logger.info(f"Batch Model Search: {len(families)} families × {len(symbols)} symbols × {len(durations)} durations")
    logger.info(f"Mode: {mode} | Max Workers: {max_workers}")
    logger.info("=" * 80)

    queue = PriorityModelSearchQueue(max_workers=max_workers)

    # 1. 构建任务队列
    for family in families:
        for symbol in symbols:
            for duration in durations:
                # 检查历史
                historical_config = get_best_historical_config(family, symbol, duration)

                if mode == "fast" and historical_config:
                    # 快速模式：只用历史配置
                    task = SearchTask(
                        family=family,
                        symbol=symbol,
                        duration=duration,
                        config=historical_config,
                        priority=TaskPriority.HIGH,
                        estimated_time=estimate_task_time(family, historical_config) * 0.3
                    )
                    queue.enqueue(task)

                elif mode == "balanced":
                    # 平衡模式：历史 + 粗筛 + 精调
                    if historical_config:
                        # 历史配置高优先级
                        task = SearchTask(
                            family=family,
                            symbol=symbol,
                            duration=duration,
                            config=historical_config,
                            priority=TaskPriority.HIGH,
                            estimated_time=estimate_task_time(family, historical_config)
                        )
                        queue.enqueue(task)

                    # 粗筛配置中优先级
                    # （这里简化，实际应该通过 AdaptiveModelSearch 生成）
                    logger.info(f"Will run balanced search for {family} {symbol} {duration}")

                else:  # exhaustive
                    # 全量模式：低优先级
                    logger.info(f"Will run exhaustive search for {family} {symbol} {duration}")

    # 2. 处理队列
    logger.info(f"Queue summary: {queue.get_queue_summary()}")
    results = await queue.process()

    logger.info("=" * 80)
    logger.info(f"Batch search completed! Total results: {len(results)}")
    logger.info("=" * 80)

    return results


def main():
    parser = argparse.ArgumentParser(description="Smart Model Search with Adaptive Strategies")

    parser.add_argument("--family", type=str, help="Model family (lstm, lightgbm, etc.)")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--duration", type=str, default="10m", help="Duration")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["fast", "balanced", "exhaustive"],
        default="balanced",
        help="Search mode"
    )

    # 批量搜索
    parser.add_argument("--families", type=str, help="Comma-separated families (e.g., lstm,lightgbm)")
    parser.add_argument("--symbols", type=str, help="Comma-separated symbols (e.g., BTCUSDT,ETHUSDT)")
    parser.add_argument("--durations", type=str, help="Comma-separated durations (e.g., 10m,30m)")
    parser.add_argument("--max-workers", type=int, default=4, help="Max parallel workers")

    args = parser.parse_args()

    # 批量模式
    if args.families:
        families = args.families.split(",")
        symbols = args.symbols.split(",") if args.symbols else ["BTCUSDT"]
        durations = args.durations.split(",") if args.durations else ["10m"]

        results = asyncio.run(
            batch_model_search(
                families=families,
                symbols=symbols,
                durations=durations,
                mode=args.mode,
                max_workers=args.max_workers,
            )
        )

        print(f"\n{'='*80}")
        print(f"Batch Search Summary:")
        print(f"{'='*80}")
        for result in results:
            task = result.get("task")
            print(
                f"{task.family:15} {task.symbol:10} {task.duration:5} | "
                f"Score: {result.get('validation_score', 0):.4f} | "
                f"Win Rate: {result.get('validation_win_rate', 0):.4f}"
            )
        return 0

    # 单个模式
    if not args.family:
        parser.print_help()
        return 1

    result = asyncio.run(
        smart_model_search(
            family=args.family,
            symbol=args.symbol,
            duration=args.duration,
            mode=args.mode,
        )
    )

    print(f"\n{'='*80}")
    print(f"Final Result:")
    print(f"{'='*80}")
    print(f"  Validation Score: {result.get('validation_score', 0):.4f}")
    print(f"  Validation Win Rate: {result.get('validation_win_rate', 0):.4f}")
    print(f"  Config: {result.get('config')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
