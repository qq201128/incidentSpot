from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from app.services import external_factor_data


def test_onchain_upsert_persists_derivatives(monkeypatch) -> None:
    db_path = Path(__file__).resolve().parent / "_onchain_test.db"
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE onchain_features (
          symbol TEXT NOT NULL,
          open_time INTEGER NOT NULL,
          exchange_netflow REAL,
          stablecoin_supply_ratio REAL,
          active_addresses REAL,
          transaction_count REAL,
          exchange_netflow_z_20 REAL,
          active_addresses_chg_1 REAL,
          transaction_count_chg_1 REAL,
          PRIMARY KEY (symbol, open_time)
        )
        """
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(external_factor_data, "get_conn", lambda: sqlite3.connect(db_path))

    external_factor_data.upsert_onchain_rows(
        "btcusdt",
        [
            {"open_time": 1, "exchange_netflow": 1.0, "active_addresses": 100.0, "transaction_count": 10.0},
            {"open_time": 2, "exchange_netflow": 3.0, "active_addresses": 110.0, "transaction_count": 11.0},
            {"open_time": 3, "exchange_netflow": 5.0, "active_addresses": 121.0, "transaction_count": 12.0},
        ],
    )

    verify = sqlite3.connect(db_path)
    row = verify.execute(
        """
        SELECT exchange_netflow_z_20, active_addresses_chg_1, transaction_count_chg_1
        FROM onchain_features
        WHERE symbol = 'BTCUSDT' AND open_time = 2
        """
    ).fetchone()
    verify.close()
    assert row[0] is not None
    assert row[1] == pytest.approx(0.1)
    assert row[2] == pytest.approx(0.1)
