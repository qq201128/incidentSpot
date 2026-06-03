from __future__ import annotations

from app.services import factor_combination_base_metric_cache as metric_cache


def test_cached_factor_metrics_by_name_returns_usable_cache(monkeypatch) -> None:
    monkeypatch.setattr(metric_cache, "cache_is_usable", lambda _cached: True)
    monkeypatch.setattr(metric_cache, "bar_aligned_features_match", lambda *_args: True)
    monkeypatch.setattr(
        metric_cache,
        "get_cached_ranking",
        lambda *_args: {"cacheMeta": {}, "ranking": [{"factorName": "factor_a", "winRate": 0.6}]},
    )

    result = metric_cache.cached_factor_metrics_by_name("BTCUSDT", "10m")
    result["factor_a"]["winRate"] = 0.1

    fresh = metric_cache.cached_factor_metrics_by_name("BTCUSDT", "10m")
    assert fresh == {"factor_a": {"factorName": "factor_a", "winRate": 0.6}}


def test_cached_factor_metrics_by_name_rejects_stale_cache(monkeypatch) -> None:
    monkeypatch.setattr(metric_cache, "cache_is_usable", lambda _cached: False)
    monkeypatch.setattr(metric_cache, "bar_aligned_features_match", lambda *_args: True)
    monkeypatch.setattr(
        metric_cache,
        "get_cached_ranking",
        lambda *_args: {"ranking": [{"factorName": "factor_a", "winRate": 0.6}]},
    )

    assert metric_cache.cached_factor_metrics_by_name("BTCUSDT", "10m") == {}


def test_cached_factor_metrics_by_name_rejects_feature_dependency_mismatch(monkeypatch) -> None:
    monkeypatch.setattr(metric_cache, "cache_is_usable", lambda _cached: True)
    monkeypatch.setattr(metric_cache, "bar_aligned_features_match", lambda *_args: False)
    monkeypatch.setattr(
        metric_cache,
        "get_cached_ranking",
        lambda *_args: {"cacheMeta": {}, "ranking": [{"factorName": "factor_a", "winRate": 0.6}]},
    )

    assert metric_cache.cached_factor_metrics_by_name("BTCUSDT", "10m") == {}
