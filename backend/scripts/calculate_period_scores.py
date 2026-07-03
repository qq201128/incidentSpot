"""
周期评分计算脚本
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

from app.services.factor_period_scoring import (
    batch_calculate_period_scores,
    calculate_period_scores,
)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="计算因子周期评分")
    parser.add_argument('--factor-id', type=int, help="计算单个因子")
    parser.add_argument('--symbol', default='BTCUSDT', help="交易对")
    parser.add_argument('--all', action='store_true', help="计算所有因子")

    args = parser.parse_args()

    if args.factor_id:
        # 计算单个因子
        logger.info(f"计算因子 {args.factor_id} 的周期评分...")
        scores = calculate_period_scores(args.factor_id, args.symbol)

        logger.info(f"\n周期评分结果:")
        for period, data in scores.items():
            if data:
                logger.info(
                    f"  {period}: 评分={data['score']:.1f}, "
                    f"胜率={data['win_rate']:.2%}, "
                    f"ICIR={data['icir']:.3f}, "
                    f"交易={data['trades']}"
                )
            else:
                logger.info(f"  {period}: 无数据")

    elif args.all:
        # 批量计算
        logger.info("批量计算所有因子的周期评分...")
        results = batch_calculate_period_scores(symbol=args.symbol)

        success_count = sum(1 for v in results.values() if v is not None)
        logger.info(f"\n完成: 成功 {success_count}/{len(results)}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
