from __future__ import annotations


def has_open_position(conn, symbol: str, strategy_key: str | None = None) -> bool:
    if strategy_key is not None:
        return _has_open_strategy_position(conn, symbol, strategy_key)
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


def _has_open_strategy_position(conn, symbol: str, strategy_key: str) -> bool:
    row = conn.execute(
        """
        SELECT e.id
        FROM events e
        JOIN orders o ON o.event_id = e.id
        WHERE e.symbol = ? AND e.strategy_key = ? AND e.status = 'OPEN'
        LIMIT 1
        """,
        (symbol.upper(), strategy_key),
    ).fetchone()
    return row is not None
