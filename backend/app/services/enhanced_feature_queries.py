from __future__ import annotations

KLINES_SELECT_SQL = """
SELECT open_time, open, high, low, close, volume
FROM klines
WHERE symbol = ? AND interval = ?
ORDER BY open_time ASC
"""

KLINES_LOOKBACK_SELECT_SQL = """
SELECT open_time, open, high, low, close, volume
FROM klines
WHERE symbol = ? AND interval = ? AND open_time >= ?
ORDER BY open_time ASC
"""

ORDERBOOK_SELECT_SQL = """
SELECT open_time, imbalance, spread_bps, bid_qty_sum, ask_qty_sum, microprice_bps, ofi_ratio
FROM orderbook_features
WHERE symbol = ?
ORDER BY open_time ASC
"""

ORDERBOOK_LOOKBACK_SELECT_SQL = """
SELECT open_time, imbalance, spread_bps, bid_qty_sum, ask_qty_sum, microprice_bps, ofi_ratio
FROM orderbook_features
WHERE symbol = ? AND open_time >= ?
ORDER BY open_time ASC
"""

FUNDING_SELECT_SQL = """
SELECT open_time, funding_rate
FROM funding_features
WHERE symbol = ?
ORDER BY open_time ASC
"""

FUNDING_LOOKBACK_SELECT_SQL = """
SELECT open_time, funding_rate
FROM funding_features
WHERE symbol = ? AND open_time >= ?
ORDER BY open_time ASC
"""
