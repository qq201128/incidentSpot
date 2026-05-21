from __future__ import annotations


def event_response(conn, row) -> dict:
    latest_order = _latest_order(conn, row["id"])
    settlement = _event_settlement(conn, row["id"])
    return {
        **_event_base(row),
        **_order_fields(latest_order),
        "totalPnl": settlement["total_pnl"] if settlement else 0,
        **_event_ai_fields(row),
    }


def _latest_order(conn, event_id: int):
    return conn.execute(
        """
        SELECT id, side, qty, price, status, created_at, external_order_id, external_status, external_response
        FROM orders
        WHERE event_id = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (event_id,),
    ).fetchone()


def _event_settlement(conn, event_id: int):
    return conn.execute(
        "SELECT COALESCE(SUM(pnl), 0) AS total_pnl FROM settlements WHERE event_id = ?",
        (event_id,),
    ).fetchone()


def _event_base(row) -> dict:
    return {
        "id": row["id"],
        "strategyKey": row["strategy_key"],
        "symbol": row["symbol"],
        "title": row["title"],
        "eventInterval": row["event_interval"],
        "ruleType": row["rule_type"],
        "strikeValue": row["strike_value"],
        "upperBound": row["upper_bound"],
        "startTime": row["start_time"],
        "endTime": row["end_time"],
        "status": row["status"],
        "result": row["result"],
        "settlementPrice": row["settlement_price"],
        "settlementQuoteTime": row["settlement_quote_time"],
        "settlementSource": row["settlement_source"],
    }


def _order_fields(order) -> dict:
    return {
        "orderSide": order["side"] if order else None,
        "orderQty": order["qty"] if order else None,
        "orderPrice": order["price"] if order else None,
        "orderCreatedAt": order["created_at"] if order else None,
        "orderStatus": order["status"] if order else None,
        "externalOrderId": order["external_order_id"] if order else None,
        "externalStatus": order["external_status"] if order else None,
        "externalResponse": order["external_response"] if order else None,
    }


def _event_ai_fields(row) -> dict:
    return {
        "aiProbabilityUp": row["ai_probability_up"],
        "aiPredictedDirection": row["ai_predicted_direction"],
        "aiPredictionCorrect": row["ai_prediction_correct"],
        "aiQualityScore": row["ai_quality_score"],
        "aiQualityPassed": row["ai_quality_passed"],
        "aiHighWinrateGate": row["ai_high_winrate_gate"],
        "aiHighWinrateRule": row["ai_high_winrate_rule"],
        "aiHighWinratePassed": row["ai_high_winrate_passed"],
        "aiHighWinrateValue": row["ai_high_winrate_value"],
    }
