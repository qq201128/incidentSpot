from __future__ import annotations

from app.services import factor_page_service


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
    assert page1["listTotal"] == 5
    assert page1["pageCount"] == 3


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
