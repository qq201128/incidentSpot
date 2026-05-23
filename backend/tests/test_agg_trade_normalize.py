from __future__ import annotations

from app.services.agg_trade_normalize import normalize_agg_trade_row


def test_normalize_agg_trade_row_maps_buy_side() -> None:
    row = normalize_agg_trade_row({"p": "100.5", "q": "2", "T": 1_700_000_000_000, "m": False})
    assert row == {
        "price": 100.5,
        "qty": 2.0,
        "quoteQty": 201.0,
        "time": 1_700_000_000_000,
        "side": "buy",
    }


def test_normalize_agg_trade_row_maps_sell_side() -> None:
    row = normalize_agg_trade_row({"p": "1", "q": "3", "T": 99, "m": True})
    assert row is not None
    assert row["side"] == "sell"


def test_normalize_agg_trade_row_rejects_invalid() -> None:
    assert normalize_agg_trade_row({"p": "0", "q": "1", "T": 1}) is None
    assert normalize_agg_trade_row({"p": "x", "q": "1", "T": 1}) is None
