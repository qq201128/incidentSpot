from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from app.services import factor_combo_strategy


def test_prediction_uses_usable_combo_cache(monkeypatch) -> None:
    cache = {
        "ranking": [_ranking_row("cached_combo")],
        "cacheStatus": {"usable": True, "reason": "usable"},
    }

    monkeypatch.setattr(factor_combo_strategy, "get_cached_combination_ranking", lambda *_args: cache)
    monkeypatch.setattr(factor_combo_strategy, "load_factor_frame", lambda _symbol, _duration: object())
    monkeypatch.setattr(
        factor_combo_strategy,
        "materialize_mined_factor_frame",
        lambda frame, **_kwargs: SimpleNamespace(frame=frame),
    )
    monkeypatch.setattr(factor_combo_strategy, "build_live_signal_from_ranking", _signal_from_row)

    result = factor_combo_strategy.predict_factor_combo_rank_direction(
        "btcusdt",
        "10m",
        combo_rank=1,
        result_strategy_key="factor_combo_ranker_v1",
        entry_open_time=123,
    )

    assert result["high_winrate_rule"] == "cached_combo"
    assert result["open_time"] == 123


def test_prediction_rejects_stale_combo_cache(monkeypatch) -> None:
    cache = {
        "ranking": [_ranking_row("legacy_combo")],
        "cacheStatus": {"usable": False, "reason": "legacy_without_fingerprint"},
    }

    monkeypatch.setattr(factor_combo_strategy, "get_cached_combination_ranking", lambda *_args: cache)

    try:
        factor_combo_strategy.predict_factor_combo_rank_direction(
            "btcusdt",
            "10m",
            combo_rank=1,
            result_strategy_key="factor_combo_ranker_v1",
        )
    except ValueError as exc:
        assert "legacy_without_fingerprint" in str(exc)
    else:
        raise AssertionError("stale factor combo cache should fail prediction")


def test_prediction_allows_append_only_combo_cache(monkeypatch) -> None:
    cache = {
        "ranking": [_ranking_row("append_combo")],
        "cacheStatus": {
            "usable": False,
            "reason": "market_data_changed",
            "cachedMarketData": {"rowCount": 1000, "maxOpenTime": 100},
            "currentMarketData": {"rowCount": 1001, "maxOpenTime": 200},
        },
    }

    monkeypatch.setattr(factor_combo_strategy, "get_cached_combination_ranking", lambda *_args: cache)
    monkeypatch.setattr(factor_combo_strategy, "load_factor_frame", lambda _symbol, _duration: object())
    monkeypatch.setattr(
        factor_combo_strategy,
        "materialize_mined_factor_frame",
        lambda frame, **_kwargs: SimpleNamespace(frame=frame),
    )
    monkeypatch.setattr(factor_combo_strategy, "build_live_signal_from_ranking", _signal_from_row)

    result = factor_combo_strategy.predict_factor_combo_rank_direction(
        "btcusdt",
        "10m",
        combo_rank=1,
        result_strategy_key="factor_combo_ranker_v1",
    )

    assert result["high_winrate_rule"] == "append_combo"
    assert "combo_cache=market_data_appended" in result["rule_reasons"]


def _ranking_row(name: str) -> dict[str, Any]:
    return {
        "factorName": name,
        "method": "test_method",
        "members": [{"name": "factor_a"}],
        "winRate": 0.7,
        "profitFactor": 1.3,
    }


def _signal_from_row(_frame: object, row: dict[str, Any], **kwargs) -> dict[str, Any]:
    return {
        "symbol": kwargs["symbol"],
        "duration": kwargs["duration"],
        "sourceOpenTime": 100,
        "entryPrice": 1.0,
        "direction": "up",
        "probabilityUp": 0.7,
        "confidence": 0.7,
        "qualityPassed": True,
        "factorName": row["factorName"],
        "members": row["members"],
        "method": row["method"],
        "historicalWinRate": row["winRate"],
        "historicalProfitFactor": row["profitFactor"],
        "qualityMinWinRate": 0.55,
        "qualityMinProfitFactor": 1.05,
        "qualityGateReason": "passed",
        "source": "factor_combination_ranking",
        "score": 0.1,
        "factorTimingReason": "passed",
        "factorTimingPassed": True,
        "factorTimingBlockedMembers": [],
    }
