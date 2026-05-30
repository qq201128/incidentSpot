from __future__ import annotations

from app.api.factor_combinations import _combination_config
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
    assert payload["ranking"] == [cached["ranking"][0] | {"strategyBucket": "regular_combo"}]
    assert payload["cacheStatus"]["reason"] == "market_data_changed"
