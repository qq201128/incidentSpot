from __future__ import annotations

from unittest.mock import patch

from app.services.n_bar_rm_htf_vol_gate import (
    _atr_wilder_last,
    evaluate_htf_counter_trend_suppress,
    evaluate_volatility_spike_suppress,
)


def test_atr_wilder_flat_series():
    tr = [2.0] * 20
    atr = _atr_wilder_last(tr, 14)
    assert atr is not None
    assert abs(atr - 2.0) < 1e-9


def test_vol_spike_suppress_high_tr():
    entry_ms = 1_700_000_000_000
    bars = []
    base = entry_ms - 60 * 10 * 1000 * 30
    for i in range(30):
        ot = base + i * 600_000
        o, h, l, c = 100.0, 100.5, 99.5, 100.0
        if i == 29:
            h, l = 110.0, 95.0
        bars.append(
            {
                "openTime": ot,
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "closeTime": ot + 600_000 - 1,
            }
        )

    with patch(
        "app.services.n_bar_rm_htf_vol_gate.fetch_index_price_klines",
        return_value=bars,
    ):
        sup, meta = evaluate_volatility_spike_suppress(
            "BTCUSDT", entry_ms, atr_period=14, tr_atr_mult=0.01
        )
    assert sup is True
    assert meta.get("volSuppress") is True


def test_htf_counter_trend_long_when_1h_bear():
    entry_ms = 1_700_000_000_000
    bars = []
    base = entry_ms - 60 * 60 * 1000 * 40
    for i in range(40):
        ot = base + i * 3600_000
        # downtrend closes
        c = 100.0 - i * 0.5
        bars.append(
            {
                "openTime": ot,
                "open": c + 1,
                "high": c + 2,
                "low": c - 1,
                "close": c,
                "closeTime": ot + 3600_000 - 1,
            }
        )

    with patch(
        "app.services.n_bar_rm_htf_vol_gate.fetch_index_price_klines",
        return_value=bars,
    ):
        sup, meta = evaluate_htf_counter_trend_suppress(
            "BTCUSDT", entry_ms, "up", sma_period=20
        )
    assert sup is True
    assert meta.get("htfTrend") == "bear"
