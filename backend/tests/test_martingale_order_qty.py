from __future__ import annotations

from app.services.auto_trade_execution import MARTINGALE_MAX_USDT, martingale_order_qty_usdt
from app.services.auto_trade_types import AutoTradeSettings
from app.services.strategy_registry import (
    BLIND_REVERSE_MARTINGALE_STRATEGY_KEY,
    FIVE_BAR_10M_RM_STRATEGY_KEY,
    FOUR_BAR_10M_RM_STRATEGY_KEY,
    THREE_BAR_10M_RM_STRATEGY_KEY,
    ORDERBOOK_NOTIONAL_MG_5102045_STRATEGY_KEY,
    ORDERBOOK_NOTIONAL_MG_STRATEGY_KEY,
    ORDERBOOK_TRADE_FLOW_INVERT_MG_STRATEGY_KEY,
    ORDERBOOK_TRADE_FLOW_STRATEGY_KEY,
)


def _settings(strategy_key: str, qty: float = 5.0) -> AutoTradeSettings:
    return AutoTradeSettings(
        strategy_key=strategy_key,
        enabled=True,
        symbol="BTCUSDT",
        duration="10m",
        duration_minutes=10,
        qty=qty,
        live_trading_enabled=False,
    )


class _FakeCursor:
    def __init__(self, row: dict | None) -> None:
        self._row = row

    def fetchone(self):
        return self._row


class _FakeConn:
    def __init__(self, row: dict | None) -> None:
        self.row = row

    def execute(self, *_args, **_kwargs):
        return _FakeCursor(self.row)

    def close(self) -> None:
        pass


class _FakeMultiCursor:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def fetchall(self) -> list[dict]:
        return self._rows

    def fetchone(self) -> dict | None:
        return self._rows[0] if self._rows else None


class _FakeMultiConn:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def execute(self, *_args, **_kwargs):
        return _FakeMultiCursor(self._rows)

    def close(self) -> None:
        pass


def test_martingale_non_invert_ignores_history(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.auto_trade_execution.get_conn",
        lambda: _FakeConn({"qty": 100.0, "correct": 0}),
    )
    assert martingale_order_qty_usdt(_settings(ORDERBOOK_TRADE_FLOW_STRATEGY_KEY, qty=5)) == 5.0


def test_martingale_base_when_no_prior_settlement(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.auto_trade_execution.get_conn",
        lambda: _FakeConn(None),
    )
    assert martingale_order_qty_usdt(_settings(ORDERBOOK_TRADE_FLOW_INVERT_MG_STRATEGY_KEY)) == 5.0


def test_martingale_resets_after_win(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.auto_trade_execution.get_conn",
        lambda: _FakeConn({"qty": 20.0, "correct": 1}),
    )
    assert martingale_order_qty_usdt(_settings(ORDERBOOK_TRADE_FLOW_INVERT_MG_STRATEGY_KEY, qty=5)) == 5.0


def test_martingale_doubles_after_loss(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.auto_trade_execution.get_conn",
        lambda: _FakeConn({"qty": 5.0, "correct": 0}),
    )
    assert martingale_order_qty_usdt(_settings(ORDERBOOK_TRADE_FLOW_INVERT_MG_STRATEGY_KEY, qty=5)) == 10.0


def test_martingale_caps_at_max(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.auto_trade_execution.get_conn",
        lambda: _FakeConn({"qty": 15.0, "correct": 0}),
    )
    assert martingale_order_qty_usdt(_settings(ORDERBOOK_TRADE_FLOW_INVERT_MG_STRATEGY_KEY, qty=5)) == MARTINGALE_MAX_USDT


def test_notional_mg_base_after_win(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.auto_trade_execution.get_conn",
        lambda: _FakeMultiConn([{"qty": 20.0, "correct": 1}]),
    )
    assert martingale_order_qty_usdt(_settings(ORDERBOOK_NOTIONAL_MG_STRATEGY_KEY)) == 5.0


def test_notional_mg_doubles_after_one_loss(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.auto_trade_execution.get_conn",
        lambda: _FakeMultiConn([{"qty": 5.0, "correct": 0}]),
    )
    assert martingale_order_qty_usdt(_settings(ORDERBOOK_NOTIONAL_MG_STRATEGY_KEY)) == 10.0


def test_notional_mg_doubles_twice(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.auto_trade_execution.get_conn",
        lambda: _FakeMultiConn([{"qty": 10.0, "correct": 0}, {"qty": 5.0, "correct": 0}]),
    )
    assert martingale_order_qty_usdt(_settings(ORDERBOOK_NOTIONAL_MG_STRATEGY_KEY)) == 20.0


def test_notional_mg_resets_after_three_losses(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.auto_trade_execution.get_conn",
        lambda: _FakeMultiConn(
            [
                {"qty": 20.0, "correct": 0},
                {"qty": 10.0, "correct": 0},
                {"qty": 5.0, "correct": 0},
            ]
        ),
    )
    assert martingale_order_qty_usdt(_settings(ORDERBOOK_NOTIONAL_MG_STRATEGY_KEY)) == 5.0


def test_notional_mg_caps_at_max(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.auto_trade_execution.get_conn",
        lambda: _FakeMultiConn([{"qty": 15.0, "correct": 0}]),
    )
    assert martingale_order_qty_usdt(_settings(ORDERBOOK_NOTIONAL_MG_STRATEGY_KEY)) == MARTINGALE_MAX_USDT


def test_notional_mg_5102045_base_after_win(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.auto_trade_execution.get_conn",
        lambda: _FakeMultiConn([{"qty": 45.0, "correct": 1}]),
    )
    assert martingale_order_qty_usdt(_settings(ORDERBOOK_NOTIONAL_MG_5102045_STRATEGY_KEY)) == 5.0


def test_notional_mg_5102045_after_one_loss(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.auto_trade_execution.get_conn",
        lambda: _FakeMultiConn([{"qty": 5.0, "correct": 0}]),
    )
    assert martingale_order_qty_usdt(_settings(ORDERBOOK_NOTIONAL_MG_5102045_STRATEGY_KEY)) == 10.0


def test_notional_mg_5102045_after_three_losses_next_is_45(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.auto_trade_execution.get_conn",
        lambda: _FakeMultiConn(
            [
                {"qty": 20.0, "correct": 0},
                {"qty": 10.0, "correct": 0},
                {"qty": 5.0, "correct": 0},
            ]
        ),
    )
    assert martingale_order_qty_usdt(_settings(ORDERBOOK_NOTIONAL_MG_5102045_STRATEGY_KEY)) == 45.0


def test_notional_mg_5102045_resets_after_four_losses(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.auto_trade_execution.get_conn",
        lambda: _FakeMultiConn(
            [
                {"qty": 45.0, "correct": 0},
                {"qty": 20.0, "correct": 0},
                {"qty": 10.0, "correct": 0},
                {"qty": 5.0, "correct": 0},
            ]
        ),
    )
    assert martingale_order_qty_usdt(_settings(ORDERBOOK_NOTIONAL_MG_5102045_STRATEGY_KEY)) == 5.0


def _patch_blind_conn(rows: list[dict]):
    def _conn():
        return _FakeMultiConn(rows)

    return _conn


def test_blind_rm_qty_base_no_history(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.blind_reverse_martingale_strategy.get_conn",
        _patch_blind_conn([]),
    )
    assert martingale_order_qty_usdt(_settings(BLIND_REVERSE_MARTINGALE_STRATEGY_KEY, qty=7)) == 7.0


def test_blind_rm_qty_after_one_loss(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.blind_reverse_martingale_strategy.get_conn",
        _patch_blind_conn([{"pred": "up", "correct": 0}]),
    )
    assert martingale_order_qty_usdt(_settings(BLIND_REVERSE_MARTINGALE_STRATEGY_KEY, qty=7)) == 10.0


def test_blind_rm_qty_after_one_loss_escalates_when_base_matches_first_rung(monkeypatch) -> None:
    """Base 10 + first ladder 10 looked like no martingale; next tier must be higher."""
    monkeypatch.setattr(
        "app.services.blind_reverse_martingale_strategy.get_conn",
        _patch_blind_conn([{"pred": "up", "correct": 0}]),
    )
    assert martingale_order_qty_usdt(_settings(BLIND_REVERSE_MARTINGALE_STRATEGY_KEY, qty=10)) == 20.0


def test_blind_rm_qty_after_two_losses(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.blind_reverse_martingale_strategy.get_conn",
        _patch_blind_conn(
            [
                {"pred": "down", "correct": 0},
                {"pred": "up", "correct": 0},
            ]
        ),
    )
    assert martingale_order_qty_usdt(_settings(BLIND_REVERSE_MARTINGALE_STRATEGY_KEY, qty=7)) == 20.0


def test_blind_rm_qty_after_three_losses(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.blind_reverse_martingale_strategy.get_conn",
        _patch_blind_conn(
            [
                {"pred": "down", "correct": 0},
                {"pred": "down", "correct": 0},
                {"pred": "up", "correct": 0},
            ]
        ),
    )
    assert martingale_order_qty_usdt(_settings(BLIND_REVERSE_MARTINGALE_STRATEGY_KEY, qty=7)) == 45.0


def test_blind_rm_qty_resets_after_four_losses(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.blind_reverse_martingale_strategy.get_conn",
        _patch_blind_conn(
            [
                {"pred": "down", "correct": 0},
                {"pred": "down", "correct": 0},
                {"pred": "down", "correct": 0},
                {"pred": "up", "correct": 0},
            ]
        ),
    )
    assert martingale_order_qty_usdt(_settings(BLIND_REVERSE_MARTINGALE_STRATEGY_KEY, qty=7)) == 7.0


def test_blind_rm_qty_resets_after_win(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.blind_reverse_martingale_strategy.get_conn",
        _patch_blind_conn(
            [
                {"pred": "up", "correct": 1},
                {"pred": "down", "correct": 0},
            ]
        ),
    )
    assert martingale_order_qty_usdt(_settings(BLIND_REVERSE_MARTINGALE_STRATEGY_KEY, qty=7)) == 7.0


def test_three_bar_rm_qty_shares_blind_martingale(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.blind_reverse_martingale_strategy.get_conn",
        _patch_blind_conn([{"pred": "up", "correct": 0}]),
    )
    assert martingale_order_qty_usdt(_settings(THREE_BAR_10M_RM_STRATEGY_KEY, qty=7)) == 10.0


def test_four_bar_rm_qty_shares_blind_martingale(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.blind_reverse_martingale_strategy.get_conn",
        _patch_blind_conn([{"pred": "up", "correct": 0}]),
    )
    assert martingale_order_qty_usdt(_settings(FOUR_BAR_10M_RM_STRATEGY_KEY, qty=7)) == 10.0


def test_five_bar_rm_qty_shares_blind_martingale(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.blind_reverse_martingale_strategy.get_conn",
        _patch_blind_conn([{"pred": "up", "correct": 0}]),
    )
    assert martingale_order_qty_usdt(_settings(FIVE_BAR_10M_RM_STRATEGY_KEY, qty=7)) == 10.0
