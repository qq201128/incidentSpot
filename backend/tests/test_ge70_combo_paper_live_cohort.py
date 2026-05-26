from __future__ import annotations

import pytest

from app.services import ge70_combo_paper_live_cohort_service as cohort
from app.services import ge70_combo_screening
from app.services.factor_combo_simulation_keys import simulation_strategy_key_for_factor_name


def test_load_ge70_mined_combo_rows_filters_by_win_rate() -> None:
    rows = cohort.load_ge70_mined_combo_rows()
    assert rows
    assert all(float((row.get("metrics") or {})["winRate"]) >= 0.62 for row in rows)
    assert all(row.get("members") for row in rows)


def test_ge70_screening_report_exposes_rejected_reasons(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ge70_combo_screening,
        "load_mined_factor_library",
        lambda _path: {
            "factors": [
                _library_row("accepted", 0.71, members=True),
                _library_row("weak", 0.61, members=True),
                _library_row("empty", 0.75, members=False),
            ]
        },
    )

    report = ge70_combo_screening.ge70_mined_combo_screening_report()

    assert [row["factorName"] for row in report["selected"]] == ["accepted"]
    assert report["rejectedCount"] == 2
    assert report["reasonCounts"]["backtest_win_rate_below_paper_live_prefilter"] == 1
    assert report["reasonCounts"]["members_missing"] == 1


def test_bootstrap_writes_high_winrate_cache_per_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    saved_reports = []
    monkeypatch.setattr(
        cohort,
        "load_ge70_mined_combo_rows",
        lambda **_kwargs: [_library_row("goal_combo__alpha__beta", 0.71, members=True)],
    )
    monkeypatch.setattr(cohort, "save_cached_high_winrate_combo_ranking", lambda report: saved_reports.append(report))
    monkeypatch.setattr(
        cohort,
        "ge70_mined_combo_screening_report",
        lambda: {
            "policy": "offline_prefilter_only_requires_paper_live_settlement",
            "rejectedReasons": [{"duration": "10m", "factorName": "weak", "reason": "below_threshold"}],
        },
    )
    monkeypatch.setattr(cohort, "promote_high_winrate_strategy", lambda *_args, **_kwargs: {"status": "backtest_candidate"})
    monkeypatch.setattr(
        cohort,
        "_ensure_batch_simulation_slots",
        lambda _symbol, duration, ranking: {"duration": duration, "enabledSlots": len(ranking)},
    )
    monkeypatch.setattr(
        cohort,
        "rebuild_combination_signal_watchlist",
        lambda *_args, **_kwargs: {"eligibleTotal": 1, "total": 1, "signalFailures": [], "cacheIssues": []},
    )
    monkeypatch.setattr(
        cohort,
        "_seed_batch_paper_live_predictions",
        lambda *_args, **_kwargs: {"savedPredictions": 0, "simulationTrades": 0, "skipped": 0, "failures": [], "failureCount": 0},
    )

    report = cohort.bootstrap_ge70_paper_live_cohort("BTCUSDT", duration="10m", seed_predictions=False)

    assert report["totalCombos"] == 1
    assert report["durations"][0]["comboCount"] == 1
    assert report["batchStrategySlots"][0]["enabledSlots"] == 1
    assert report["offlineScreening"]["rejectedCount"] == 1
    assert saved_reports[0]["ranking"][0]["backtestWinRate"] == 0.71
    assert saved_reports[0]["ranking"][0]["paperLiveWinRate"] is None
    assert saved_reports[0]["ranking"][0]["paperLiveStatus"] == "paper_collecting"
    assert "_combo_" in simulation_strategy_key_for_factor_name("goal_combo__alpha__beta")


def _library_row(name: str, win_rate: float, *, members: bool) -> dict:
    return {
        "symbol": "BTCUSDT",
        "duration": "10m",
        "factorName": name,
        "formula": f"oriented_zscore_pair_threshold_v1({name})",
        "method": "oriented_expanding_zscore_pair_threshold_v1",
        "members": _members() if members else [],
        "metrics": {"winRate": win_rate, "profitFactor": 1.2, "totalPeriods": 120},
    }


def _members() -> list[dict]:
    return [
        {"name": "alpha", "displayName": "alpha", "category": "unknown", "orientation": 1},
        {"name": "beta", "displayName": "beta", "category": "unknown", "orientation": -1},
    ]
