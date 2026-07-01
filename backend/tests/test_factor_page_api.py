from __future__ import annotations

import pytest

from app.api import factors as factors_api
from app.services import factor_page_service
from app.services import factor_ranking_api_payloads
from app.services.factor_ranking_page import build_ranking_page
from app.services.factor_metric_enrichment import factor_score


def test_classify_factor_sources() -> None:
    assert factor_page_service.classify_factor_source("kline_features.py", "ret_1") == "local_definition"
    assert (
        factor_page_service.classify_factor_source("agent_mined_factor_library.json", "agent__x")
        == "agent_candidate"
    )
    assert factor_page_service.classify_factor_source("mined_factor_library.json", "combo__a__b") == "composite_cache"
    assert factor_page_service.classify_factor_source("lstm_features.py", "lstm_score") == "lstm_shadow"


def test_build_factor_overview_counts(monkeypatch) -> None:
    monkeypatch.setattr(
        factor_page_service,
        "list_single_factor_summaries",
        lambda *_args, **_kwargs: [
            {"name": "ret_1", "sourceFile": "kline_features.py"},
            {"name": "agent__a", "sourceFile": "agent_mined_factor_library.json"},
        ],
    )
    monkeypatch.setattr(
        factor_page_service,
        "list_combo_factor_summaries",
        lambda: [{"name": "combo__a__b", "sourceFile": "mined_factor_library.json"}],
    )

    overview = factor_page_service.build_factor_overview()

    assert overview["singleTotal"] == 2
    assert overview["comboTotal"] == 1
    assert overview["sourceSummary"]["local_definition"] == 1
    assert overview["sourceSummary"]["agent_candidate"] == 1
    assert overview["sourceSummary"]["composite_cache"] == 1


def test_build_factor_list_page_pagination(monkeypatch) -> None:
    rows = [{"name": f"f{i}", "sourceFile": "kline_features.py"} for i in range(5)]
    monkeypatch.setattr(factor_page_service, "list_single_factor_summaries", lambda *_a, **_k: rows)
    monkeypatch.setattr(factor_page_service, "list_combo_factor_summaries", lambda: [])
    monkeypatch.setattr(factor_page_service, "list_single_factor_categories", lambda: [])

    page1 = factor_page_service.build_factor_list_page(
        category=None, kind="single", query=None, page=1, page_size=2
    )
    assert len(page1["factors"]) == 2
    assert page1["total"] == 5
    assert page1["unfilteredTotal"] == 5
    assert page1["pageCount"] == 3
    assert page1["query"] == ""


def test_single_factor_list_page_does_not_expand_combo_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [{"name": "ret_1", "sourceFile": "kline_features.py"}]
    monkeypatch.setattr(factor_page_service, "list_single_factor_summaries", lambda *_a, **_k: rows)
    monkeypatch.setattr(factor_page_service, "list_combo_factor_summaries", lambda: [])
    monkeypatch.setattr(factor_page_service, "list_single_factor_categories", lambda: [])

    def fail_combo_rows(*_args):
        raise AssertionError("single factor list should not expand combo rows")

    monkeypatch.setattr(factor_page_service, "combo_list_rows", fail_combo_rows)

    page = factor_page_service.build_factor_list_page(
        category=None,
        kind="single",
        symbol="BTCUSDT",
        duration="10m",
        query=None,
        page=1,
        page_size=20,
    )

    assert page["factors"][0]["name"] == "ret_1"
    assert page["comboFactors"] == []


def test_build_factor_list_page_search_and_clamps_last_page(monkeypatch) -> None:
    rows = [
        {"name": "ret_1", "description": "return", "sourceFile": "kline_features.py"},
        {"name": "risk_1", "description": "drawdown risk", "sourceFile": "kline_features.py"},
    ]
    monkeypatch.setattr(factor_page_service, "list_single_factor_summaries", lambda *_a, **_k: rows)
    monkeypatch.setattr(factor_page_service, "list_combo_factor_summaries", lambda: [])
    monkeypatch.setattr(factor_page_service, "list_single_factor_categories", lambda: [])

    page = factor_page_service.build_factor_list_page(
        category=None, kind="single", query="risk", page=99, page_size=1
    )

    assert page["factors"][0]["name"] == "risk_1"
    assert page["total"] == 1
    assert page["unfilteredTotal"] == 2
    assert page["page"] == 1
    assert page["query"] == "risk"


def test_combo_factor_list_page_sorts_by_score(monkeypatch) -> None:
    rows = [
        {"name": "combo__low", "sourceFile": "mined_factor_library.json", "factorScore": 10.0},
        {"name": "combo__high", "sourceFile": "mined_factor_library.json", "factorScore": 90.0},
        {"name": "combo__mid", "sourceFile": "mined_factor_library.json", "factorScore": 50.0},
    ]
    monkeypatch.setattr(factor_page_service, "list_single_factor_summaries", lambda *_a, **_k: [])
    monkeypatch.setattr(factor_page_service, "list_combo_factor_summaries", lambda: rows)
    monkeypatch.setattr(factor_page_service, "list_single_factor_categories", lambda: [])

    page = factor_page_service.build_factor_list_page(
        category=None, kind="combo", query=None, page=1, page_size=20
    )

    assert [row["name"] for row in page["factors"]] == ["combo__high", "combo__mid", "combo__low"]


def test_combo_factor_list_page_uses_current_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(factor_page_service, "list_single_factor_summaries", lambda *_a, **_k: [])
    monkeypatch.setattr(factor_page_service, "list_combo_factor_summaries", lambda: [])
    monkeypatch.setattr(factor_page_service, "list_single_factor_categories", lambda: [])
    monkeypatch.setattr(
        "app.services.factor_page_combo_rows.get_cached_combination_ranking",
        lambda *_a: {
            "ranking": [],
            "evaluatedRanking": [
                {
                    "factorName": "combo__a__b",
                    "factorScore": 11.0,
                    "sharpe": 0.8,
                    "trades": 48,
                    "avgTradesPerDay": 24.0,
                    "walkForwardPassed": False,
                    "walkForwardFailureReason": "validation_win_rate_below_min",
                    "members": [{"name": "a"}, {"name": "b"}],
                },
                {
                    "factorName": "combo__c__d",
                    "factorScore": 22.0,
                    "members": [{"name": "c"}, {"name": "d"}],
                },
            ],
        },
    )

    page = factor_page_service.build_factor_list_page(
        category=None,
        kind="combo",
        symbol="BTCUSDT",
        duration="30m",
        query=None,
        page=1,
        page_size=20,
    )

    assert page["comboTotal"] == 2
    assert page["total"] == 2
    assert [row["name"] for row in page["factors"]] == ["combo__c__d", "combo__a__b"]
    failed = next(row for row in page["factors"] if row["name"] == "combo__a__b")
    assert failed["paperLiveStatus"] == "observe_only"
    assert failed["walkForwardFailureReason"] == "validation_win_rate_below_min"
    assert failed["sharpe"] == 0.8
    assert failed["trades"] == 48
    assert failed["avgTradesPerDay"] == 24.0


def test_build_factor_page_bundle_omits_full_ranking(monkeypatch: pytest.MonkeyPatch) -> None:
    cached = {
        "ranking": [
            {"factorName": "ret_high", "category": "trend", "factorScore": 90.0, "ir": 1.0},
            {"factorName": "ret_low", "category": "trend", "factorScore": 10.0, "ir": 0.5},
            {"factorName": "risk_high", "category": "risk", "factorScore": 80.0, "ir": 1.5},
        ],
        "updatedAt": "2026-05-21T00:00:00+00:00",
    }
    monkeypatch.setattr(factor_page_service, "list_single_factor_summaries", lambda *_a, **_k: [])
    monkeypatch.setattr(factor_page_service, "list_combo_factor_summaries", lambda: [])
    monkeypatch.setattr(factor_page_service, "list_single_factor_categories", lambda: [])
    monkeypatch.setattr(factor_page_service, "agent_factor_rows_for_duration", lambda *_a: [])
    monkeypatch.setattr(factor_page_service, "_high_winrate_card", lambda *_a: None)
    calls = []

    def fake_cached_ranking(*_args):
        calls.append(_args)
        return cached

    monkeypatch.setattr(factor_page_service, "get_cached_ranking", fake_cached_ranking)

    payload = factor_page_service.build_factor_page_bundle(
        "btcusdt",
        "10m",
        category="trend",
        page=1,
        page_size=1,
    )

    assert "ranking" not in payload
    assert payload["rankingPageTotal"] == 2
    assert payload["rankingTotal"] == 2
    assert len(calls) == 1


def test_build_factor_page_bundle_does_not_build_alerts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(factor_page_service, "list_single_factor_summaries", lambda *_a, **_k: [])
    monkeypatch.setattr(factor_page_service, "list_combo_factor_summaries", lambda: [])
    monkeypatch.setattr(factor_page_service, "list_single_factor_categories", lambda: [])
    monkeypatch.setattr(factor_page_service, "agent_factor_rows_for_duration", lambda *_a: [])
    monkeypatch.setattr(factor_page_service, "_high_winrate_card", lambda *_a: None)
    monkeypatch.setattr(factor_page_service, "get_cached_ranking", lambda *_a: None)

    def fail_alerts(*_args, **_kwargs):
        raise AssertionError("page bundle should not block on alerts")

    monkeypatch.setattr(factor_page_service, "build_factor_alerts", fail_alerts)

    payload = factor_page_service.build_factor_page_bundle("BTCUSDT", "10m")

    assert payload["alerts"] == []


def test_factor_page_context_counts_evaluated_combo_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(factor_page_service, "list_single_factor_summaries", lambda *_a, **_k: [])
    monkeypatch.setattr(factor_page_service, "list_combo_factor_summaries", lambda: [])
    monkeypatch.setattr(factor_page_service, "agent_factor_rows_for_duration", lambda *_a: [])
    monkeypatch.setattr(factor_page_service, "_high_winrate_card", lambda *_a: None)
    monkeypatch.setattr(
        "app.services.factor_page_combo_rows.get_cached_combination_ranking",
        lambda *_a: {
            "ranking": [],
            "evaluatedRanking": [
                {"factorName": "combo__a__b", "members": [{"name": "a"}, {"name": "b"}]},
                {"factorName": "combo__c__d", "members": [{"name": "c"}, {"name": "d"}]},
            ],
        },
    )
    monkeypatch.setattr(factor_page_service, "get_cached_ranking", lambda *_a: None)

    payload = factor_page_service.build_factor_page_context("BTCUSDT", "30m")

    assert payload["comboTotal"] == 2


def test_build_factor_period_scores_from_cache(monkeypatch) -> None:
    def fake_cached(symbol: str, duration: str):
        if duration != "10m":
            return None
        return {
            "ranking": [
                {"factorName": "ret_1", "factorScore": 82.4, "totalPeriods": 1284},
            ],
            "updatedAt": "2026-05-21T00:00:00+00:00",
        }

    monkeypatch.setattr(factor_page_service, "get_cached_ranking", fake_cached)

    payload = factor_page_service.build_factor_period_scores("btcusdt", "ret_1")

    assert payload["symbol"] == "BTCUSDT"
    ten = next(item for item in payload["scores"] if item["duration"] == "10m")
    assert ten["factorScore"] == 82.4
    assert ten["available"] is True
    thirty = next(item for item in payload["scores"] if item["duration"] == "30m")
    assert thirty["available"] is False


def test_combo_detail_uses_combination_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        factors_api,
        "get_factor_payload_by_name",
        lambda _name: {
            "name": "combo__a__b",
            "displayName": "组合：A + B",
            "description": "组合：A + B",
            "sourceFile": "mined_factor_library.json",
        },
    )
    monkeypatch.setattr(
        factors_api,
        "combo_metrics_for_factor",
        lambda *_args: {"factorName": "combo__a__b", "factorScore": 91.2, "winRate": 0.72},
    )

    detail = factors_api.get_factor_detail("combo__a__b", symbol="BTCUSDT", duration="10m")

    assert detail["factorScore"] == 91.2
    assert detail["winRate"] == 0.72


def test_combo_detail_uses_evaluated_cache_without_library_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(factors_api, "get_factor_payload_by_name", lambda _name: None)
    monkeypatch.setattr(
        "app.services.factor_combo_metrics.get_cached_combination_ranking",
        lambda *_args: {
            "cacheStatus": {"usable": False, "reason": "market_data_changed"},
            "ranking": [],
            "evaluatedRanking": [
                {
                    "factorName": "combo__a__b",
                    "factorDisplayName": "组合：A + B",
                    "factorScore": 42.0,
                    "winRate": 0.51,
                    "members": [{"name": "a"}, {"name": "b"}],
                }
            ],
        },
    )

    detail = factors_api.get_factor_detail("combo__a__b", symbol="BTCUSDT", duration="30m")

    assert detail["name"] == "combo__a__b"
    assert detail["factorScore"] == 42.0
    assert detail["winRate"] == 0.51


def test_combo_period_scores_use_combination_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        factors_api,
        "combo_period_scores",
        lambda symbol, factor_name: {
            "symbol": symbol,
            "factorName": factor_name,
            "scores": [{"duration": "10m", "factorScore": 88.0, "available": True}],
        },
    )

    payload = factors_api.factor_period_scores("combo__a__b", symbol="BTCUSDT")

    assert payload["scores"][0]["factorScore"] == 88.0


def test_factor_detail_ignores_query_defaults_without_symbol_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        factors_api,
        "get_factor_payload_by_name",
        lambda _name: {"name": "ret_1", "sourceFile": "kline_features.py"},
    )

    def fail_metrics(*_args):
        raise AssertionError("metrics should not be loaded without explicit symbol and duration")

    monkeypatch.setattr(factors_api, "_metrics_for_factor", fail_metrics)

    detail = factors_api.get_factor_detail("ret_1")

    assert detail["name"] == "ret_1"


def test_library_combo_detail_merges_full_backtest_without_overwriting_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = {
        "factorName": "combo__a__b",
        "factorScore": 65.8,
        "winRate": 0.714,
        "ir": 0.98,
        "profitFactor": 2.64,
    }
    monkeypatch.setattr(factors_api, "get_factor_payload_by_name", lambda _name: {"name": "combo__a__b"})
    monkeypatch.setattr(
        factors_api,
        "combo_metrics_for_factor",
        lambda *_args: {
            **row,
            "winRate": 0.5035,
            "icMean": 0.12,
            "longShortReturn": 0.034,
            "maxDrawdown": -0.08,
            "tStat": 2.4,
            "pValue": 0.02,
        },
    )

    detail = factors_api.get_factor_detail("combo__a__b", symbol="BTCUSDT", duration="10m")

    assert detail["factorScore"] == 65.8
    assert detail["winRate"] == 0.5035
    assert detail["icMean"] == 0.12
    assert detail["longShortReturn"] == 0.034
    assert detail["maxDrawdown"] == -0.08
    assert detail["tStat"] == 2.4
    assert detail["pValue"] == 0.02


def test_library_combo_detail_uses_backtest_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    row = {
        "factorName": "combo__a__b",
        "factorDisplayName": "组合：A + B",
        "metrics": {"winRate": 0.714, "profitFactor": 2.64, "totalPeriods": 114},
        "score": 65.8,
    }
    monkeypatch.setattr(
        "app.services.factor_combo_metrics.mined_factor_rows_for_duration",
        lambda *_args: [row],
    )
    monkeypatch.setattr(
        "app.services.factor_combo_metrics.get_cached_combination_ranking",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        "app.services.factor_combo_metrics.get_usable_combo_backtest",
        lambda *_args: {"factorName": "combo__a__b", "icMean": 0.12, "factorScore": 99.0},
    )

    def fail_backtest(*_args):
        raise AssertionError("backtest should not run on cache hit")

    monkeypatch.setattr("app.services.factor_backtest_service.run_factor_backtest", fail_backtest)

    detail = factors_api.combo_metrics_for_factor("BTCUSDT", "10m", "combo__a__b")

    assert detail["factorScore"] == factor_score(row["metrics"])
    assert detail["icMean"] == 0.12


def test_factor_ranking_page_filters_and_paginates() -> None:
    rows = [
        {"factorName": "ret_1", "category": "momentum", "factorScore": 90.0},
        {"factorName": "volatility_1", "description": "rolling risk", "factorScore": 80.0},
        {"factorName": "basis_gap", "sourceLabel": "local_definition", "factorScore": 70.0},
    ]

    page = build_ranking_page(rows, "risk", page=1, page_size=1)

    assert page["ranking"][0]["factorName"] == "volatility_1"
    assert page["total"] == 1
    assert page["unfilteredTotal"] == 3
    assert page["page"] == 1
    assert page["pageCount"] == 1
    assert page["query"] == "risk"


def test_factor_ranking_page_empty_search_clamps_page() -> None:
    rows = [{"factorName": "ret_1", "category": "momentum"}]

    page = build_ranking_page(rows, "missing", page=5, page_size=1)

    assert page["ranking"] == []
    assert page["total"] == 0
    assert page["unfilteredTotal"] == 1
    assert page["page"] == 1
    assert page["pageCount"] == 1


def test_factor_ranking_api_returns_backend_page(monkeypatch: pytest.MonkeyPatch) -> None:
    cached = {
        "ranking": [
            {"factorName": "ret_low", "category": "trend", "factorScore": 10.0, "ir": 1.0},
            {"factorName": "ret_high", "category": "trend", "factorScore": 30.0, "ir": 1.0},
            {"factorName": "risk_high", "category": "risk", "factorScore": 50.0, "ir": 1.0},
        ],
        "updatedAt": "2026-05-21T00:00:00+00:00",
        "rankingDiagnostics": {},
        "rankingFailures": [],
    }
    monkeypatch.setattr(factor_ranking_api_payloads, "factor_ranking_precomputed_symbols", lambda: ["BTCUSDT"])
    monkeypatch.setattr(factor_ranking_api_payloads, "get_cached_ranking", lambda *_args: cached)
    monkeypatch.setattr(factor_ranking_api_payloads, "cache_is_usable", lambda _cached: True)

    payload = factors_api.factor_ranking(
        symbol="btcusdt",
        duration="10m",
        category="trend",
        q=None,
        page=1,
        page_size=1,
    )

    assert payload["ranking"][0]["factorName"] == "ret_high"
    assert payload["total"] == 2
    assert payload["unfilteredTotal"] == 2
    assert payload["pageCount"] == 2
    assert payload["pageSize"] == 1
    assert "regularRanking" not in payload


def test_factor_ranking_api_normalizes_query_default(monkeypatch: pytest.MonkeyPatch) -> None:
    cached = {
        "ranking": [
            {"factorName": "ret_low", "category": "trend", "factorScore": 10.0, "ir": 1.0},
            {"factorName": "ret_high", "category": "trend", "factorScore": 30.0, "ir": 1.0},
        ],
        "updatedAt": "2026-05-21T00:00:00+00:00",
    }
    monkeypatch.setattr(factor_ranking_api_payloads, "factor_ranking_precomputed_symbols", lambda: ["BTCUSDT"])
    monkeypatch.setattr(factor_ranking_api_payloads, "get_cached_ranking", lambda *_args: cached)
    monkeypatch.setattr(factor_ranking_api_payloads, "cache_is_usable", lambda _cached: True)

    payload = factors_api.factor_ranking(symbol="btcusdt", duration="10m", page=1, page_size=1)

    assert payload["query"] == ""
    assert payload["ranking"][0]["factorName"] == "ret_high"
    assert "regularRanking" not in payload


def test_factor_page_route_normalizes_optional_query_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_bundle(symbol, duration, **kwargs):
        captured.update({"symbol": symbol, "duration": duration, **kwargs})
        return {"ok": True}

    monkeypatch.setattr(factors_api, "build_factor_page_bundle", fake_bundle)

    payload = factors_api.factor_page(symbol="BTCUSDT")

    assert payload == {"ok": True}
    assert captured["duration"] == "10m"
    assert captured["category"] is None
    assert captured["query"] is None
    assert captured["kind"] == "single"
    assert captured["page"] == 1
    assert captured["page_size"] == 20


def test_factor_list_route_passes_symbol_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_page(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(factors_api, "build_factor_list_page", fake_page)

    payload = factors_api.list_factors(kind="combo", symbol="BTCUSDT", duration="30m")

    assert payload == {"ok": True}
    assert captured["symbol"] == "BTCUSDT"
    assert captured["duration"] == "30m"
    assert captured["kind"] == "combo"
