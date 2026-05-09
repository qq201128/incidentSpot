from __future__ import annotations

from app.services.blind_reverse_martingale_strategy import (
    BlindRmSettlementState,
    predict_blind_reverse_martingale_direction,
)
from app.services.strategy_registry import BLIND_REVERSE_MARTINGALE_STRATEGY_KEY


def test_cycle_anchor_opposite_chain() -> None:
    st = BlindRmSettlementState(
        consecutive_losses=2,
        rows_considered=[
            {"pred": "down", "correct": 0},
            {"pred": "up", "correct": 0},
        ],
    )
    assert st.cycle_anchor_direction() == "up"


def test_predict_inverts_when_mid_cycle(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.blind_reverse_martingale_strategy.load_blind_rm_settlement_state",
        lambda *_args, **_kwargs: BlindRmSettlementState(
            consecutive_losses=1,
            rows_considered=[{"pred": "up", "correct": 0}],
        ),
    )
    monkeypatch.setattr(
        "app.services.blind_reverse_martingale_strategy.fetch_premium_index",
        lambda _sym: {"indexPrice": 100.0},
    )
    monkeypatch.setattr(
        "app.services.blind_reverse_martingale_strategy.is_within_entry_grace",
        lambda *_a, **_k: True,
    )
    out = predict_blind_reverse_martingale_direction("BTCUSDT", entry_open_time=1_700_000_000_000)
    assert out["direction"] == "down"
    assert out["strategy_key"] == BLIND_REVERSE_MARTINGALE_STRATEGY_KEY
