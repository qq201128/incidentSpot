from __future__ import annotations

from fastapi import BackgroundTasks
import pandas as pd

from app.api import factor_combinations
from app.api.factor_combinations import _stale_combination_ranking
from app.services.combination_ranking_page import build_combination_ranking_page
from app.services.factor_combination_refresh_api import combination_config
from app.services import factor_combination_incremental_refresh as incremental_refresh


def test_combination_config_maps_query_values_to_dataclass_fields() -> None:
    config = combination_config(
        profile="full",
        base_factor_limit=25,
        combo_sizes="2,3",
        result_limit=400,
    )

    assert config.base_factor_limit == 25
    assert config.combo_sizes == (2, 3)
    assert config.result_limit == 400
    assert config.native_factor_limit == 120
    assert config.mined_factor_limit == 12
    assert config.agent_factor_limit == 12
    assert config.prefilter_limit == 500
    assert config.beam_width == 500
    assert config.lookback_days is None
    assert config.lookback_bars == 720


def test_stale_combination_ranking_filters_nested_combo_rows() -> None:
    cached = {
        "ranking": [
            {
                "factorName": "combo__factor_a__factor_b",
                "factorScore": 88.0,
                "members": [{"name": "factor_a"}, {"name": "factor_b"}],
            },
            {
                "factorName": "combo__combo__factor_a__factor_b__factor_c",
                "factorScore": 99.0,
                "members": [{"name": "combo__factor_a__factor_b"}, {"name": "factor_c"}],
            },
        ],
        "updatedAt": "2026-05-14T00:00:00+00:00",
        "cacheStatus": {"usable": False, "reason": "market_data_changed"},
    }

    payload = _stale_combination_ranking("BTCUSDT", "10m", cached)

    assert payload["source"] == "stale_cache"
    assert payload["total"] == 1
    assert payload["rawTotal"] == 2
    assert payload["nestedComboFilteredCount"] == 1
    assert "ranking" not in payload
    assert "regularRanking" not in payload
    assert payload["cacheStatus"]["reason"] == "market_data_changed"


def test_combination_ranking_pagination_filters_by_member_name() -> None:
    rows = [
        {
            "factorName": "combo__trend__volume",
            "members": [{"name": "trend"}, {"name": "volume"}],
        },
        {
            "factorName": "combo__carry__basis",
            "members": [{"name": "funding_carry"}, {"name": "basis"}],
        },
        {
            "factorName": "combo__momentum__volatility",
            "members": [{"name": "momentum"}, {"name": "volatility"}],
        },
    ]

    payload = build_combination_ranking_page(rows, "carry", page=1, page_size=1)

    assert payload["total"] == 1
    assert payload["unfilteredTotal"] == 3
    assert payload["pageCount"] == 1
    assert payload["ranking"][0]["factorName"] == "combo__carry__basis"


def test_combination_ranking_search_clamps_page_after_filter() -> None:
    rows = [
        {
            "factorName": "combo__trend__volume",
            "members": [{"name": "trend"}, {"name": "volume"}],
        },
        {
            "factorName": "combo__carry__basis",
            "members": [{"name": "funding_carry"}, {"name": "basis"}],
        },
    ]

    payload = build_combination_ranking_page(rows, "carry", page=3, page_size=1)

    assert payload["page"] == 1
    assert payload["pageCount"] == 1
    assert payload["total"] == 1
    assert payload["ranking"][0]["factorName"] == "combo__carry__basis"


def test_combination_ranking_empty_search_is_explicit() -> None:
    rows = [{"factorName": "combo__trend__volume", "members": [{"name": "trend"}]}]

    payload = build_combination_ranking_page(rows, "missing", page=8, page_size=1)

    assert payload["ranking"] == []
    assert payload["total"] == 0
    assert payload["unfilteredTotal"] == 1
    assert payload["page"] == 1
    assert payload["pageCount"] == 1
    assert payload["query"] == "missing"


def test_combination_ranking_api_does_not_return_full_regular_ranking(monkeypatch) -> None:
    rows = [
        {"factorName": "combo__trend__volume", "members": [{"name": "trend"}, {"name": "volume"}]},
        {"factorName": "combo__carry__basis", "members": [{"name": "carry"}, {"name": "basis"}]},
        {"factorName": "combo__momentum__volatility", "members": [{"name": "momentum"}, {"name": "volatility"}]},
    ]
    monkeypatch.setattr(
        factor_combinations,
        "get_cached_combination_ranking",
        lambda _symbol, _duration: {
            "symbol": "BTCUSDT",
            "duration": "10m",
            "ranking": rows,
            "updatedAt": "2026-05-31T00:00:00+00:00",
        },
    )
    monkeypatch.setattr(factor_combinations, "cache_is_usable", lambda _cached: True)
    monkeypatch.setattr(factor_combinations, "_high_winrate_view", lambda _symbol, _duration: {})

    payload = factor_combinations.factor_combination_ranking(
        symbol="BTCUSDT",
        duration="10m",
        page=1,
        page_size=1,
    )

    assert len(payload["ranking"]) == 1
    assert payload["total"] == 3
    assert payload["unfilteredTotal"] == 3
    assert payload["regularTotal"] == 3
    assert "regularRanking" not in payload


def test_combination_ranking_api_displays_evaluated_rows_without_promoting_to_trading(monkeypatch) -> None:
    rows = [
        {
            "factorName": "combo__passed__volume",
            "members": [{"name": "passed"}, {"name": "volume"}],
            "walkForwardPassed": True,
        }
    ]
    evaluated = [
        *rows,
        {
            "factorName": "combo__failed__carry",
            "members": [{"name": "failed"}, {"name": "carry"}],
            "walkForwardPassed": False,
            "walkForwardFailureReason": "validation_profit_factor_below_min",
        },
    ]
    monkeypatch.setattr(
        factor_combinations,
        "get_cached_combination_ranking",
        lambda _symbol, _duration: {
            "symbol": "BTCUSDT",
            "duration": "10m",
            "ranking": rows,
            "evaluatedRanking": evaluated,
            "updatedAt": "2026-05-31T00:00:00+00:00",
        },
    )
    monkeypatch.setattr(factor_combinations, "cache_is_usable", lambda _cached: True)
    monkeypatch.setattr(factor_combinations, "_high_winrate_view", lambda _symbol, _duration: {})

    payload = factor_combinations.factor_combination_ranking(symbol="BTCUSDT", duration="10m", page_size=10)

    assert [row["factorName"] for row in payload["ranking"]] == [
        "combo__passed__volume",
        "combo__failed__carry",
    ]
    assert payload["passedRankingTotal"] == 1
    assert payload["evaluatedRankingTotal"] == 2
    assert payload["regularTotal"] == 2


def test_combination_ranking_route_normalizes_optional_query_defaults(monkeypatch) -> None:
    rows = [
        {"factorName": "combo__trend__volume", "members": [{"name": "trend"}, {"name": "volume"}]},
        {"factorName": "combo__carry__basis", "members": [{"name": "carry"}, {"name": "basis"}]},
    ]
    monkeypatch.setattr(
        factor_combinations,
        "get_cached_combination_ranking",
        lambda _symbol, _duration: {
            "symbol": "BTCUSDT",
            "duration": "10m",
            "ranking": rows,
            "updatedAt": "2026-05-31T00:00:00+00:00",
        },
    )
    monkeypatch.setattr(factor_combinations, "cache_is_usable", lambda _cached: True)
    monkeypatch.setattr(factor_combinations, "_high_winrate_view", lambda _symbol, _duration: {})

    payload = factor_combinations.factor_combination_ranking(symbol="BTCUSDT")

    assert payload["duration"] == "10m"
    assert payload["query"] == ""
    assert payload["page"] == 1
    assert payload["pageSize"] == factor_combinations.DEFAULT_RANKING_PAGE_SIZE
    assert len(payload["ranking"]) == 2


def test_combination_refresh_route_normalizes_optional_query_defaults(monkeypatch) -> None:
    tasks = BackgroundTasks()
    monkeypatch.setattr(
        factor_combinations,
        "background_refresh_combo_rankings",
        lambda *_args: None,
    )

    payload = factor_combinations.factor_combination_refresh(background_tasks=tasks, symbol="btcusdt")

    assert payload["symbol"] == "BTCUSDT"
    assert payload["duration"] is None
    assert payload["profile"] == "full"
    assert tasks.tasks[0].args[1] is None


def test_incremental_refresh_filters_non_computable_source_rows(monkeypatch) -> None:
    frame = pd.DataFrame({"open": [1, 2], "close": [2, 3], "factor_a": [1, 2], "factor_b": [2, 1]})
    monkeypatch.setattr(incremental_refresh, "load_factor_frame", lambda *_args, **_kwargs: frame)
    monkeypatch.setattr(incremental_refresh, "get_cached_combination_ranking", lambda *_args: None)
    monkeypatch.setattr(incremental_refresh, "save_cached_combination_ranking", lambda _report: None)
    monkeypatch.setattr(incremental_refresh, "assert_cache_usable_for_live_signal", lambda *_args: None)
    monkeypatch.setattr(
        incremental_refresh,
        "get_cached_ranking",
        lambda *_args: {
            "ranking": [
                {"factorName": "missing_agent"},
                {"factorName": "factor_a"},
                {"factorName": "factor_b"},
            ]
        },
    )
    monkeypatch.setattr(
        incremental_refresh,
        "combination_result_for_member_rows",
        lambda *_args, **_kwargs: {"factorName": "combo__factor_a__factor_b", "walkForwardPassed": False},
    )

    report = incremental_refresh.refresh_incremental_combination_cache("BTCUSDT", "30m", batch_size=10)

    assert report["evaluatedTotal"] == 1
    assert report["searchDiagnostics"]["missingSourceFactorCount"] == 1
    assert report["searchDiagnostics"]["lookbackDays"] == incremental_refresh.DEFAULT_LOOKBACK_DAYS
