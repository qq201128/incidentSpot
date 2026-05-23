from __future__ import annotations

from app.services import coinmetrics_onchain_data as onchain


def test_fetch_onchain_feature_rows_maps_flow_and_stablecoin_ratio(monkeypatch) -> None:
    flow_payload = {
        "data": [
            {
                "asset": "btc",
                "time": "2026-05-20T00:00:00.000000000Z",
                "AdrActCnt": "100",
                "TxCnt": "200",
                "FlowInExNtv": "10",
                "FlowOutExNtv": "4",
            }
        ]
    }
    stable_payload = {
        "data": [
            {
                "asset": "btc",
                "time": "2026-05-20T00:00:00.000000000Z",
                "CapMrktCurUSD": "1000",
            },
            {
                "asset": "usdt",
                "time": "2026-05-20T00:00:00.000000000Z",
                "SplyCur": "300",
            },
            {
                "asset": "usdc",
                "time": "2026-05-20T00:00:00.000000000Z",
                "SplyCur": "200",
            },
        ]
    }
    calls: list[dict] = []

    def fake_retry_get(_url: str, params: dict, **_kwargs):
        calls.append(dict(params))
        if "FlowInExNtv" in params["metrics"]:
            return flow_payload
        return stable_payload

    monkeypatch.setattr(onchain, "retry_get", fake_retry_get)
    monkeypatch.setattr(onchain, "_lookback_window", lambda _days: ("2026-05-01", "2026-05-21"))

    rows = onchain.fetch_onchain_feature_rows("BTCUSDT")

    assert len(rows) == 1
    row = rows[0]
    assert row["exchange_netflow"] == 6.0
    assert row["active_addresses"] == 100.0
    assert row["transaction_count"] == 200.0
    assert row["stablecoin_supply_ratio"] == 0.5
    assert len(calls) == 2


def test_fetch_onchain_feature_rows_returns_empty_for_unsupported_symbol() -> None:
    assert onchain.fetch_onchain_feature_rows("SOLUSDT") == []
