from __future__ import annotations

from fastapi import BackgroundTasks

from app.api import factor_combinations
from app.api.factor_combinations import _combination_config
from app.api.factor_combinations import _paginated_ranking_payload
from app.api.factor_combinations import _stale_combination_ranking


def test_combination_config_maps_query_values_to_dataclass_fields() -> None:
    config = _combination_config(
        profile="full",
        base_factor_limit=25,
        combo_sizes="2,3",
        result_limit=400,
    )

    assert config.base_factor_limit == 25
    assert config.combo_sizes == (2, 3)
    assert config.result_limit == 400
    assert config.native_factor_limit == 32
    assert config.prefilter_limit == 800


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

    payload = _paginated_ranking_payload(rows, "carry", page=1, page_size=1)

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

    payload = _paginated_ranking_payload(rows, "carry", page=3, page_size=1)

    assert payload["page"] == 1
    assert payload["pageCount"] == 1
    assert payload["total"] == 1
    assert payload["ranking"][0]["factorName"] == "combo__carry__basis"


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
        "_background_refresh_combo_rankings",
        lambda *_args: None,
    )

    payload = factor_combinations.factor_combination_refresh(background_tasks=tasks, symbol="btcusdt")

    assert payload["symbol"] == "BTCUSDT"
    assert payload["duration"] is None
    assert payload["profile"] == "full"
    assert tasks.tasks[0].args[1] is None
