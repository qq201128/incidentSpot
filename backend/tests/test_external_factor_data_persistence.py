from __future__ import annotations

import sqlite3

import pytest

from app.services import external_factor_data


def test_positioning_upsert_persists_taker_share_and_derivatives(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "context.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE futures_positioning_features (
          symbol TEXT NOT NULL,
          open_time INTEGER NOT NULL,
          open_interest REAL,
          open_interest_value REAL,
          long_short_ratio REAL,
          long_account REAL,
          short_account REAL,
          taker_buy_sell_ratio REAL,
          taker_buy_vol REAL,
          taker_sell_vol REAL,
          open_interest_chg_1 REAL,
          open_interest_value_chg_1 REAL,
          open_interest_z_20 REAL,
          long_short_ratio_chg_1 REAL,
          taker_buy_share REAL,
          PRIMARY KEY (symbol, open_time)
        )
        """
    )
    conn.close()
    monkeypatch.setattr(external_factor_data, "get_conn", lambda: sqlite3.connect(db_path))

    external_factor_data.upsert_positioning_rows("btcusdt", _positioning_rows())

    conn = sqlite3.connect(db_path)
    row = conn.execute(
        """
        SELECT taker_buy_share, open_interest_chg_1, long_short_ratio_chg_1
        FROM futures_positioning_features
        WHERE symbol = 'BTCUSDT' AND open_time = 2
        """
    ).fetchone()
    conn.close()
    assert row[0] == pytest.approx(0.6)
    assert row[1] == pytest.approx(0.1)
    assert row[2] == pytest.approx(0.1)


def _positioning_rows() -> list[dict]:
    return [
        _positioning_row(open_time=1, open_interest=100.0, long_short_ratio=1.0, buy_vol=60.0, sell_vol=40.0),
        _positioning_row(open_time=2, open_interest=110.0, long_short_ratio=1.1, buy_vol=90.0, sell_vol=60.0),
        _positioning_row(open_time=3, open_interest=121.0, long_short_ratio=1.21, buy_vol=100.0, sell_vol=100.0),
    ]


def _positioning_row(
    *,
    open_time: int,
    open_interest: float,
    long_short_ratio: float,
    buy_vol: float,
    sell_vol: float,
) -> dict:
    return {
        "open_time": open_time,
        "open_interest": open_interest,
        "open_interest_value": open_interest * 2.0,
        "long_short_ratio": long_short_ratio,
        "long_account": long_short_ratio,
        "short_account": 1.0,
        "taker_buy_sell_ratio": buy_vol / sell_vol,
        "taker_buy_vol": buy_vol,
        "taker_sell_vol": sell_vol,
    }
