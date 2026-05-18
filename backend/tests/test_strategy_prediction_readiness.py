from __future__ import annotations

import pandas as pd

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
    assert result.recoverable is True


def test_high_winrate_readiness_allows_nonempty_ranking(monkeypatch) -> None:
    monkeypatch.setattr(
        readiness,
        "get_cached_high_winrate_combo_ranking",
        lambda *_args: _cache([{"factorName": "goal_combo__a__b"}]),
    )

    result = readiness.strategy_prediction_readiness(HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY, "BTCUSDT", "10m")

    assert result.ready is True
    assert result.reason == "ready"


def test_factor_combo_readiness_recovers_empty_cache(monkeypatch) -> None:
    caches = [_cache([]), _cache([{"factorName": "combo__a__b"}])]
    recovered = []

    monkeypatch.setattr(readiness, "get_cached_combination_ranking", lambda *_args: caches.pop(0))
    monkeypatch.setattr(
        readiness,
        "_recover_factor_combo_ranking",
        lambda symbol, duration: recovered.append((symbol, duration)) or {"rankingTotal": 1},
    )

    result = readiness.strategy_prediction_readiness(
        FACTOR_COMBO_STRATEGY_KEY,
        "BTCUSDT",
        "10m",
        attempt_recovery=True,
    )

    assert result.ready is True
    assert result.recovery_attempted is True
    assert result.recovery_status == "recovered"
    assert recovered == [("BTCUSDT", "10m")]


def test_factor_combo_recovery_uses_fast_profile_when_default_is_empty(monkeypatch) -> None:
    from app.services import factor_combination_background as combo_background
    from app.services import factor_combination_cache_service as combo_cache
    from app.services import factor_combination_service as combo_service
    from app.services import factor_mined_library as mined_library

    reports = [
        {"symbol": "BTCUSDT", "duration": "10m", "ranking": [], "searchConfig": {"baseFactorLimit": 16}},
        {"symbol": "BTCUSDT", "duration": "10m", "ranking": [{"factorName": "combo__a__b"}], "searchConfig": {"baseFactorLimit": 8}},
    ]
    saved = []

    monkeypatch.setattr(combo_background, "_refresh_duration_klines", lambda *_args: None)
    monkeypatch.setattr(combo_service, "run_factor_combination_ranking", lambda *_args, **_kwargs: reports.pop(0))
    monkeypatch.setattr(combo_cache, "save_cached_combination_ranking", lambda report: saved.append(report))
    monkeypatch.setattr(mined_library, "regular_library_combination_rows_for_duration", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(mined_library, "mined_factor_rows_for_duration", lambda *_args: [])
    monkeypatch.setattr(mined_library, "upsert_good_combinations", lambda report: {"promoted": len(report["ranking"])})

    diagnostics = readiness._recover_factor_combo_ranking("BTCUSDT", "10m")

    assert diagnostics["recoveryProfile"] == "fast"
    assert diagnostics["rankingTotal"] == 1
    assert saved[0]["searchConfig"]["baseFactorLimit"] == 8


def test_factor_combo_recovery_prefers_library_rows(monkeypatch) -> None:
    from app.services import factor_combination_background as combo_background
    from app.services import factor_combination_cache_service as combo_cache
    from app.services import factor_frame_service
    from app.services import factor_learning_controls
    from app.services import factor_mined_library as mined_library
    from app.services import factor_mined_candidates

    saved = []

    monkeypatch.setattr(combo_background, "_refresh_duration_klines", lambda *_args: None)
    monkeypatch.setattr(
        mined_library,
        "regular_library_combination_rows_for_duration",
        lambda *_args, **_kwargs: [{"factorName": "combo__library__row", "members": [{"name": "factor_a"}], "winRate": 0.6, "profitFactor": 1.2, "totalPeriods": 120}],
    )
    monkeypatch.setattr(
        mined_library,
        "mined_factor_rows_for_duration",
        lambda *_args: [{"factorName": "combo__library__row"}],
    )
    monkeypatch.setattr(factor_frame_service, "load_factor_frame", lambda *_args: pd.DataFrame({"close": []}))
    monkeypatch.setattr(factor_learning_controls, "load_factor_learning_memory_for", lambda *_args: None)
    monkeypatch.setattr(factor_learning_controls, "learning_risk_blocked_factor_names", lambda *_args: set())
    monkeypatch.setattr(
        factor_mined_candidates,
        "materialize_mined_factor_frame_for_rows",
        lambda *_args, **_kwargs: type(
            "Result",
            (),
            {"frame": pd.DataFrame({"combo__library__row": []}), "failures": ()},
        )(),
    )
    monkeypatch.setattr(combo_cache, "save_cached_combination_ranking", lambda report: saved.append(report))

    diagnostics = readiness._recover_factor_combo_ranking("BTCUSDT", "10m")

    assert diagnostics["recoveryProfile"] == "library"
    assert diagnostics["rankingTotal"] == 1
    assert saved[0]["searchConfig"]["source"] == "mined_factor_library"


def test_high_winrate_readiness_recovers_missing_cache(monkeypatch) -> None:
    caches = [None, _cache([{"factorName": "goal_combo__a__b"}])]
    recovered = []

    monkeypatch.setattr(readiness, "get_cached_high_winrate_combo_ranking", lambda *_args: caches.pop(0))
    monkeypatch.setattr(
        readiness,
        "_recover_high_winrate_ranking",
        lambda symbol, duration: recovered.append((symbol, duration)) or {"rankingTotal": 1},
    )

    result = readiness.strategy_prediction_readiness(
        HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY,
        "BTCUSDT",
        "10m",
        attempt_recovery=True,
    )

    assert result.ready is True
    assert result.recovery_status == "recovered"
    assert recovered == [("BTCUSDT", "10m")]


def test_high_winrate_recovery_refreshes_duration_klines_first(monkeypatch) -> None:
    from app.services import factor_combination_background as combo_background
    from app.services import high_winrate_strategy_rotation as rotation

    calls = []

    def refresh(symbol: str, duration: str) -> None:
        calls.append(("refresh", symbol, duration))

    def run_goal(symbol: str, duration: str) -> dict:
        calls.append(("goal", symbol, duration))
        return _cache([{"factorName": "goal_combo__a__b"}])

    monkeypatch.setattr(combo_background, "_refresh_duration_klines", refresh)
    monkeypatch.setattr(rotation, "refresh_high_winrate_goal", run_goal)

    diagnostics = readiness._recover_high_winrate_ranking(" ethusdt ", "10m")

    assert calls == [
        ("refresh", "ETHUSDT", "10m"),
        ("goal", "ETHUSDT", "10m"),
    ]
    assert diagnostics["rankingTotal"] == 1


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
