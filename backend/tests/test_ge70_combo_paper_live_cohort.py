from __future__ import annotations

import pytest

from app.services import ge70_combo_paper_live_cohort_service as cohort
from app.services.factor_combo_simulation_keys import simulation_strategy_key_for_factor_name
def test_load_ge70_mined_combo_rows_filters_by_win_rate() -> None:
    rows = cohort.load_ge70_mined_combo_rows()
    assert rows
    assert all(float((row.get("metrics") or {})["winRate"]) >= 0.62 for row in rows)
    assert all(row.get("members") for row in rows)


def test_bootstrap_writes_high_winrate_cache_per_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cohort,
        "load_ge70_mined_combo_rows",
        lambda **_kwargs: [
            {
                "symbol": "BTCUSDT",
                "duration": "10m",
                "factorName": "goal_combo__alpha__beta",
                "formula": "oriented_zscore_pair_threshold_v1(alpha, beta)",
                "method": "oriented_expanding_zscore_pair_threshold_v1",
                "members": [
                    {"name": "alpha", "displayName": "alpha", "category": "unknown", "orientation": 1},
                    {"name": "beta", "displayName": "beta", "category": "unknown", "orientation": -1},
                ],
                "metrics": {"winRate": 0.71, "profitFactor": 1.2, "totalPeriods": 120},
            }
        ],
    )
    monkeypatch.setattr(cohort, "save_cached_high_winrate_combo_ranking", lambda *_args, **_kwargs: None)
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
    assert "_combo_" in simulation_strategy_key_for_factor_name("goal_combo__alpha__beta")
