"""
批量结算服务 - 减少API调用，提升结算效率
按标的分组结算事件，复用价格查询结果
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.db.session import get_conn
from app.services.binance_service import fetch_premium_index
from app.services.event_search_index import refresh_event_search_row
from app.services.settlement_service import (
    LIVE_SETTLEMENT_SOURCE,
    SettlementQuote,
    _event_contract_pnl,
    evaluate_event_result,
    parse_event_end_time_ms,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BatchSettlementResult:
    """批量结算结果"""
    total_events: int
    settled_count: int
    failed_events: list[dict[str, Any]]
    price_fetch_count: int  # API调用次数


def batch_settle_due_events(due_event_ids: list[int]) -> BatchSettlementResult:
    """
    批量结算到期事件

    优化策略：
    1. 按标的分组事件，每个标的只调用一次价格API
    2. 分批处理事件，减少单次事务时间
    3. 批量刷新事件搜索索引

    Args:
        due_event_ids: 到期事件ID列表

    Returns:
        BatchSettlementResult: 包含结算统计和失败详情
    """
    if not due_event_ids:
        return BatchSettlementResult(
            total_events=0,
            settled_count=0,
            failed_events=[],
            price_fetch_count=0,
        )

    conn = get_conn()
    try:
        events = _load_events_batch(conn, due_event_ids)
        events_by_symbol = _group_events_by_symbol(events)

        settled_count = 0
        failed_events = []
        price_fetch_count = 0

        for symbol, symbol_events in events_by_symbol.items():
            try:
                quote = _fetch_settlement_quote_for_symbol(symbol)
                price_fetch_count += 1

                # 分批处理，每批最多 10 个事件
                batch_size = 10
                for i in range(0, len(symbol_events), batch_size):
                    batch = symbol_events[i:i + batch_size]
                    for event in batch:
                        if _settle_single_event_with_quote(conn, event, quote):
                            settled_count += 1
                        else:
                            failed_events.append({
                                "eventId": event["id"],
                                "symbol": event["symbol"],
                                "reason": "settlement_execution_failed"
                            })
                    # 每批提交一次
                    conn.commit()

            except Exception as exc:
                logger.error(f"Failed to fetch price for symbol={symbol}: {exc}")
                for event in symbol_events:
                    failed_events.append({
                        "eventId": event["id"],
                        "symbol": event["symbol"],
                        "reason": f"price_fetch_failed: {type(exc).__name__}",
                        "error": str(exc),
                    })

        return BatchSettlementResult(
            total_events=len(events),
            settled_count=settled_count,
            failed_events=failed_events,
            price_fetch_count=price_fetch_count,
        )

    except Exception as exc:
        conn.rollback()
        logger.exception("Batch settlement failed")
        raise
    finally:
        conn.close()


def _load_events_batch(conn, event_ids: list[int]) -> list[dict]:
    """批量加载事件"""
    placeholders = ",".join("?" * len(event_ids))
    query = f"""
        SELECT * FROM events
        WHERE id IN ({placeholders})
        AND status = 'OPEN'
    """
    rows = conn.execute(query, event_ids).fetchall()
    return [dict(row) for row in rows]


def _group_events_by_symbol(events: list[dict]) -> dict[str, list[dict]]:
    """按标的分组事件"""
    grouped = defaultdict(list)
    for event in events:
        grouped[event["symbol"]].append(event)
    return dict(grouped)


def _fetch_settlement_quote_for_symbol(symbol: str) -> SettlementQuote:
    """获取标的的结算价格"""
    premium_index = fetch_premium_index(symbol)
    price = float(premium_index.get("indexPrice") or 0)
    quote_time = int(premium_index.get("time") or 0)

    if price <= 0 or quote_time <= 0:
        raise ValueError(f"Invalid premium index response for {symbol}")

    return SettlementQuote(
        price=price,
        quote_time_ms=quote_time,
        source=LIVE_SETTLEMENT_SOURCE,
    )


def _settle_single_event_with_quote(conn, event: dict, quote: SettlementQuote) -> bool:
    """
    使用给定价格结算单个事件

    Returns:
        bool: 是否成功结算
    """
    try:
        event_id = event["id"]

        # 检查事件是否已到期
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        end_time_ms = parse_event_end_time_ms(event["end_time"])

        if now_ms < end_time_ms:
            logger.warning(f"Event {event_id} not yet due, skipping")
            return False

        # 计算价格漂移
        drift_ms = abs(quote.quote_time_ms - end_time_ms)
        source_with_drift = f"{quote.source};driftMs={drift_ms}"

        # 评估事件结果
        result = evaluate_event_result(event, quote.price)

        # 结算订单
        _settle_orders_for_event(conn, event_id, result)

        # 判断预测是否正确
        ai_correct = _prediction_correct(event, result)

        # 更新事件状态
        conn.execute(
            """
            UPDATE events
            SET status = 'SETTLED',
                result = ?,
                settlement_price = ?,
                settlement_quote_time = ?,
                settlement_source = ?,
                ai_prediction_correct = ?
            WHERE id = ?
            """,
            (result, quote.price, quote.quote_time_ms, source_with_drift, ai_correct, event_id),
        )

        # 刷新搜索索引
        refresh_event_search_row(conn, event_id)

        logger.info(
            f"Settled event {event_id} with result={result}, "
            f"price={quote.price}, drift={drift_ms}ms"
        )

        return True

    except Exception as exc:
        logger.exception(f"Failed to settle event {event.get('id')}")
        return False


def _settle_orders_for_event(conn, event_id: int, result: str) -> None:
    """结算事件的所有订单"""
    orders = conn.execute("SELECT * FROM orders WHERE event_id = ?", (event_id,)).fetchall()
    settled_at = datetime.now(timezone.utc).isoformat()

    for order in orders:
        pnl = _event_contract_pnl(order, result)
        conn.execute(
            "INSERT INTO settlements(event_id, order_id, pnl, settled_at) VALUES(?, ?, ?, ?)",
            (event_id, order["id"], pnl, settled_at),
        )


def _prediction_correct(event: dict, result: str) -> int | None:
    """判断AI预测是否正确"""
    pred_dir = (event.get("ai_predicted_direction") or "").lower()
    if pred_dir not in {"up", "down"}:
        return None

    hit = (pred_dir == "up" and result == "YES") or (pred_dir == "down" and result == "NO")
    return 1 if hit else 0
