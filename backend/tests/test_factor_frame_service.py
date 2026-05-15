from __future__ import annotations

import pandas as pd

from app.services import factor_frame_service


def test_load_factor_frame_uses_strategy_duration_klines(monkeypatch) -> None:
    captured = {}

    def load_klines(symbol: str, interval: str) -> pd.DataFrame:
        captured["klines"] = (symbol, interval)
        return pd.DataFrame({"open_time": [0], "open": [1], "high": [1], "low": [1], "close": [1], "volume": [1]})

    def build_frame(df: pd.DataFrame, **_kwargs) -> tuple[pd.DataFrame, list[str]]:
        return df.assign(feature_a=1.0), ["feature_a"]

    monkeypatch.setattr(factor_frame_service, "load_klines", load_klines)
    monkeypatch.setattr(factor_frame_service, "load_orderbook_features", lambda _symbol: pd.DataFrame())
    monkeypatch.setattr(factor_frame_service, "load_funding_features", lambda _symbol: pd.DataFrame())
    monkeypatch.setattr(factor_frame_service, "load_external_feature_frames", lambda _symbol: None)
    monkeypatch.setattr(factor_frame_service, "build_enhanced_feature_frame", build_frame)

    frame = factor_frame_service.load_factor_frame("btcusdt", "10m")

    assert captured["klines"] == ("btcusdt", "10m")
    assert frame["feature_a"].iloc[0] == 1.0
