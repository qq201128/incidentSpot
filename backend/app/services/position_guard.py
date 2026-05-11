from __future__ import annotations


def has_open_position(
    conn,
    symbol: str,
    strategy_key: str | None = None,
    *,
    event_interval: str | None = None,
) -> bool:
    if strategy_key is not None:
        return _has_open_strategy_position(conn, symbol, strategy_key, event_interval=event_interval)
    row = conn.execute(
        """
        SELECT e.id
        FROM events e
        JOIN orders o ON o.event_id = e.id
        WHERE e.symbol = ? AND e.status = 'OPEN'
        LIMIT 1
        """,
        (symbol.upper(),),
    ).fetchone()
    return row is not None


def _has_open_strategy_position(
    conn,
    symbol: str,
    strategy_key: str,
    *,
    event_interval: str | None = None,
) -> bool:
    # 仅用「未结算事件」判断持仓：避免仅有 events 无 orders 时误判无仓而重复开仓。
    # event_interval 与自动下单 settings.duration 一致时，同一策略可多周期并行持仓。
    if event_interval is not None:
        row = conn.execute(
            """
            SELECT e.id
            FROM events e
            WHERE e.symbol = ? AND e.strategy_key = ? AND e.event_interval = ?
              AND e.status = 'OPEN'
            LIMIT 1
            """,
            (symbol.upper(), strategy_key, event_interval),
        ).fetchone()
        return row is not None
    row = conn.execute(
        """
        SELECT e.id
        FROM events e
        WHERE e.symbol = ? AND e.strategy_key = ? AND e.status = 'OPEN'
        LIMIT 1
        """,
        (symbol.upper(), strategy_key),
    ).fetchone()
    return row is not None
