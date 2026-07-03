"""
因子库健康检查定时任务
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.factor_performance_monitor import get_performance_monitor

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


async def check_factor_health(notify: bool = False):
    """
    检查因子健康状态

    Args:
        notify: 是否发送通知
    """
    logger.info("=" * 80)
    logger.info("Factor Health Check - %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 80)

    monitor = get_performance_monitor()

    # 获取趋势
    trends = monitor.get_trending_factors()

    # 改进的因子
    improving = trends["improving"]
    if improving:
        logger.info(f"\n✅ 表现改进的因子 ({len(improving)}个):")
        for factor in improving[:5]:  # 显示前5个
            logger.info(
                f"  - {factor['factor_id']}: "
                f"胜率 {factor['recent_win_rate']:.2%} "
                f"(改进 {factor['improvement']*100:+.1f}%)"
            )

    # 衰减的因子
    degrading = trends["degrading"]
    if degrading:
        logger.warning(f"\n⚠️ 性能衰减的因子 ({len(degrading)}个):")
        for factor in degrading:
            logger.warning(
                f"  - {factor['factor_id']}: "
                f"胜率从 {factor['historical_win_rate']:.2%} 降至 {factor['recent_win_rate']:.2%} "
                f"(下降 {factor['degradation']*100:.1f}%)"
            )
            logger.warning(f"    建议: {factor['recommendation']}")

        # 发送通知
        if notify and degrading:
            send_degradation_notification(degrading)

    # 清理过期缓存
    expired_count = monitor.cleanup_expired()
    if expired_count > 0:
        logger.info(f"\n🗑️ 清理了 {expired_count} 条过期性能记录")

    logger.info("\n" + "=" * 80)
    logger.info("Health Check Complete")
    logger.info("=" * 80)


def send_degradation_notification(degrading_factors: list[dict]):
    """
    发送衰减通知

    TODO: 集成实际的通知系统（邮件、钉钉、Slack等）
    """
    critical = [f for f in degrading_factors if f["severity"] == "critical"]
    high = [f for f in degrading_factors if f["severity"] == "high"]

    message = f"⚠️ 因子性能衰减警告\n\n"
    message += f"严重衰减: {len(critical)}个\n"
    message += f"高度衰减: {len(high)}个\n"
    message += f"总计: {len(degrading_factors)}个\n"

    logger.info(f"\n📧 通知内容:\n{message}")

    # TODO: 实际发送
    # send_email(subject="因子衰减警告", body=message)
    # send_dingtalk(message)


def main():
    parser = argparse.ArgumentParser(description="Factor Health Check")
    parser.add_argument("--notify", action="store_true", help="Send notification if degradation detected")
    parser.add_argument("--loop", action="store_true", help="Run continuously (every hour)")
    parser.add_argument("--interval", type=int, default=3600, help="Loop interval in seconds")

    args = parser.parse_args()

    if args.loop:
        logger.info(f"Running in loop mode (interval={args.interval}s)")
        while True:
            asyncio.run(check_factor_health(notify=args.notify))
            logger.info(f"Sleeping for {args.interval}s...")
            import time
            time.sleep(args.interval)
    else:
        asyncio.run(check_factor_health(notify=args.notify))


if __name__ == "__main__":
    main()
