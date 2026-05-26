from __future__ import annotations

from types import SimpleNamespace

from app.services import factor_combo_batch_predictions as service


def test_offline_screening_reports_focus_pool_and_rejected_reasons(monkeypatch) -> None:
    rows = [_ranking_row(index) for index in range(11)]
    monkeypatch.setattr(service, "_usable_caches", lambda *_args: [{"ranking": rows}])
    monkeypatch.setattr(service, "load_factor_frame", lambda *_args: object())
    monkeypatch.setattr(service, "materialize_mined_factor_frame", lambda *_args, **_kwargs: SimpleNamespace(frame=object()))
    monkeypatch.setattr(service, "build_live_signal_from_ranking", _signal)

    report = service.offline_candidate_screening_report("btcusdt", "10m")

    assert report["focusedCount"] == service.OBSERVATION_POOL_LIMIT
    assert report["candidateCount"] == 11
    assert report["rejectedCount"] == 1
    assert report["rejectedReasons"][0]["reason"] == "outside_observation_pool_limit"
    assert report["focusedCandidates"][0]["factorName"] == "combo_10"


def test_backtest_qualified_combo_rows_use_shared_backtest_gate(monkeypatch) -> None:
    rows = [
        {**_ranking_row(1), "winRate": 0.63, "profitFactor": 1.1, "totalPeriods": 120},
        {**_ranking_row(2), "winRate": 0.50, "profitFactor": 1.1, "totalPeriods": 120},
    ]
    monkeypatch.setattr(service, "_usable_caches", lambda *_args: [{"ranking": rows}])

    qualified = service.backtest_qualified_factor_combo_rows("BTCUSDT", "10m")

    assert [row["factorName"] for row in qualified] == ["combo_1"]


def test_offline_screening_reports_no_usable_cache_reason(monkeypatch) -> None:
    monkeypatch.setattr(service, "_usable_caches", lambda *_args: [])

    report = service.offline_candidate_screening_report("BTCUSDT", "10m")

    assert report["focusedCandidates"] == []
    assert report["rejectedReasons"][0]["reason"] == "no_usable_offline_candidate_cache"


def _ranking_row(index: int) -> dict:
    return {
        "factorName": f"combo_{index}",
        "winRate": 0.63 + index / 1000,
        "profitFactor": 1.2 + index / 100,
        "walkForward": {"stabilityScore": index, "oosWinRate": 0.6},
        "walkForwardPassed": True,
    }


def _signal(_frame, _row, **_kwargs) -> dict:
    return {"qualityPassed": True}
