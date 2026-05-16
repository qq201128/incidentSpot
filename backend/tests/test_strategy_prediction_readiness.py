from __future__ import annotations

from app.services import strategy_prediction_readiness as readiness
from app.services.strategy_registry import FACTOR_COMBO_STRATEGY_KEY, HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY


def test_factor_combo_readiness_blocks_empty_ranking(monkeypatch) -> None:
    monkeypatch.setattr(
        readiness,
        "get_cached_combination_ranking",
        lambda *_args: _cache([]),
    )

    result = readiness.strategy_prediction_readiness(FACTOR_COMBO_STRATEGY_KEY, "BTCUSDT", "10m")

    assert result.ready is False
    assert result.reason == "ranking_cache_empty"


def test_high_winrate_readiness_allows_nonempty_ranking(monkeypatch) -> None:
    monkeypatch.setattr(
        readiness,
        "get_cached_high_winrate_combo_ranking",
        lambda *_args: _cache([{"factorName": "goal_combo__a__b"}]),
    )

    result = readiness.strategy_prediction_readiness(HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY, "BTCUSDT", "10m")

    assert result.ready is True
    assert result.reason == "ready"


def test_non_combo_readiness_is_ready() -> None:
    result = readiness.strategy_prediction_readiness("factor_lstm_shadow_10m", "BTCUSDT", "10m")

    assert result.ready is True


def _cache(ranking: list[dict]) -> dict:
    return {
        "ranking": ranking,
        "cacheStatus": {
            "usable": True,
            "reason": "usable",
        },
    }
