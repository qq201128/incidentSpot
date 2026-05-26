from __future__ import annotations

import pandas as pd

from app.db.session import get_conn
from app.services.binance_service import fetch_funding_rate, fetch_klines, fetch_orderbook


def upsert_orderbook_rows(symbol: str, rows: list[dict]) -> None:
    conn = get_conn()
    for row in rows:
        conn.execute(
            """
            INSERT INTO orderbook_features(symbol, open_time, imbalance, spread_bps, bid_qty_sum, ask_qty_sum)
            VALUES(?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, open_time) DO UPDATE SET
              imbalance=excluded.imbalance,
              spread_bps=excluded.spread_bps,
              bid_qty_sum=excluded.bid_qty_sum,
              ask_qty_sum=excluded.ask_qty_sum
            """,
            (
                symbol.upper(),
                int(row["open_time"]),
                float(row.get("imbalance", 0.0)),
                float(row.get("spread_bps", 0.0)),
                float(row.get("bid_qty_sum", 0.0)),
                float(row.get("ask_qty_sum", 0.0)),
            ),
        )
    conn.commit()
    conn.close()


def upsert_funding_rows(symbol: str, rows: list[dict]) -> None:
    conn = get_conn()
    for row in rows:
        conn.execute(
            """
            INSERT INTO funding_features(symbol, open_time, funding_rate)
            VALUES(?, ?, ?)
            ON CONFLICT(symbol, open_time) DO UPDATE SET
              funding_rate=excluded.funding_rate
            """,
            (symbol.upper(), int(row["open_time"]), float(row.get("funding_rate", 0.0))),
        )
    conn.commit()
    conn.close()


def ingest_enhanced_data(symbol: str, target_klines: int = 20_000, intervals: tuple[str, ...] = ("1m", "5m", "15m", "1h")) -> None:
    sym = symbol.upper()
    for interval in intervals:
        rows = fetch_klines(sym, interval, limit=min(1000, target_klines))
        if rows:
            upsert_klines(sym, interval, rows)
    ingest_orderbook_snapshot(sym)
    ingest_funding_snapshot(sym)


def ingest_orderbook_snapshot(symbol: str) -> None:
    ob = fetch_orderbook(symbol, limit=500)
    df_1m = pd.read_sql_query(
        "SELECT open_time, close_time FROM klines WHERE symbol=? AND interval='1m' ORDER BY open_time DESC LIMIT 1",
        get_conn(),
        params=(symbol,),
    )
    if not df_1m.empty:
        upsert_orderbook_rows(symbol, [{"open_time": int(df_1m.iloc[0]["open_time"]), **ob}])


def ingest_funding_snapshot(symbol: str) -> None:
    rate = fetch_funding_rate(symbol)
    if rate is None:
        return
    df_1m_ot = pd.read_sql_query(
        "SELECT open_time FROM klines WHERE symbol=? AND interval='1m' ORDER BY open_time DESC LIMIT 1",
        get_conn(),
        params=(symbol,),
    )
    if not df_1m_ot.empty:
        upsert_funding_rows(symbol, [{"open_time": int(df_1m_ot.iloc[0]["open_time"]), "funding_rate": rate}])


def upsert_klines(symbol: str, interval: str, rows: list[dict]) -> None:
    conn = get_conn()
    for item in rows:
        conn.execute(
            """
            INSERT INTO klines(symbol, interval, open_time, open, high, low, close, volume, close_time)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, interval, open_time) DO UPDATE SET
              open=excluded.open,
              high=excluded.high,
              low=excluded.low,
              close=excluded.close,
              volume=excluded.volume,
              close_time=excluded.close_time
            """,
            (
                symbol.upper(),
                interval,
                item["openTime"],
                item["open"],
                item["high"],
                item["low"],
                item["close"],
                item["volume"],
                item["closeTime"],
            ),
        )
    conn.commit()
    conn.close()
