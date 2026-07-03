"""
测试批量结算服务
"""
import pytest
from datetime import datetime, timezone, timedelta

from app.services.batch_settlement_service import (
    batch_settle_due_events,
    BatchSettlementResult,
)


def test_batch_settle_empty_list():
    """测试空事件列表"""
    result = batch_settle_due_events([])

    assert result.total_events == 0
    assert result.settled_count == 0
    assert result.failed_events == []
    assert result.price_fetch_count == 0


def test_batch_settlement_reduces_api_calls(monkeypatch, setup_test_db):
    """测试批量结算减少API调用"""
    from app.db.session import get_conn

    # 创建测试事件（同一标的）
    conn = get_conn()
    try:
        end_time = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()

        event_ids = []
        for i in range(5):
            cursor = conn.execute(
                """
                INSERT INTO events(
                    strategy_key, symbol, title, event_interval,
                    rule_type, strike_value, start_time, end_time, status
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "test",
                    "BTCUSDT",
                    f"Test Event {i}",
                    "10m",
                    "ABOVE",
                    50000.0,
                    datetime.now(timezone.utc).isoformat(),
                    end_time,
                    "OPEN",
                ),
            )
            event_ids.append(cursor.lastrowid)

        conn.commit()

        # Mock Binance API
        fetch_count = {"count": 0}

        def mock_fetch_premium_index(symbol: str):
            fetch_count["count"] += 1
            return {
                "indexPrice": "51000.0",
                "time": int(datetime.now(timezone.utc).timestamp() * 1000),
            }

        monkeypatch.setattr(
            "app.services.batch_settlement_service.fetch_premium_index",
            mock_fetch_premium_index,
        )

        # 执行批量结算
        result = batch_settle_due_events(event_ids)

        # 验证结果
        assert result.total_events == 5
        assert result.settled_count == 5
        assert result.price_fetch_count == 1  # 只调用1次API
        assert fetch_count["count"] == 1

        # 验证事件已结算
        for event_id in event_ids:
            event = conn.execute(
                "SELECT status, result, settlement_price FROM events WHERE id = ?",
                (event_id,)
            ).fetchone()
            assert event["status"] == "SETTLED"
            assert event["result"] == "YES"  # 51000 > 50000
            assert event["settlement_price"] == 51000.0

    finally:
        conn.close()


def test_batch_settlement_handles_multiple_symbols(monkeypatch, setup_test_db):
    """测试多标的批量结算"""
    from app.db.session import get_conn

    conn = get_conn()
    try:
        end_time = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()

        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        event_ids = []

        for symbol in symbols:
            for i in range(2):
                cursor = conn.execute(
                    """
                    INSERT INTO events(
                        strategy_key, symbol, title, event_interval,
                        rule_type, strike_value, start_time, end_time, status
                    )
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "test",
                        symbol,
                        f"Test {symbol} {i}",
                        "10m",
                        "ABOVE",
                        1000.0,
                        datetime.now(timezone.utc).isoformat(),
                        end_time,
                        "OPEN",
                    ),
                )
                event_ids.append(cursor.lastrowid)

        conn.commit()

        # Mock Binance API
        fetch_count = {"count": 0}

        def mock_fetch_premium_index(symbol: str):
            fetch_count["count"] += 1
            prices = {"BTCUSDT": "51000.0", "ETHUSDT": "3500.0", "BNBUSDT": "600.0"}
            return {
                "indexPrice": prices.get(symbol, "1000.0"),
                "time": int(datetime.now(timezone.utc).timestamp() * 1000),
            }

        monkeypatch.setattr(
            "app.services.batch_settlement_service.fetch_premium_index",
            mock_fetch_premium_index,
        )

        # 执行批量结算
        result = batch_settle_due_events(event_ids)

        # 验证结果
        assert result.total_events == 6
        assert result.settled_count == 6
        assert result.price_fetch_count == 3  # 3个标的，3次API调用
        assert fetch_count["count"] == 3

    finally:
        conn.close()


@pytest.fixture
def setup_test_db():
    """设置测试数据库"""
    from app.db.session import init_db
    init_db()
    yield
    # 清理由测试创建的数据
