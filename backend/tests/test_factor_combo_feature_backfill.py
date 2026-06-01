from __future__ import annotations

import pandas as pd
import pytest

from app.services import factor_combo_feature_backfill as backfill


def test_backfill_generates_snapshots_from_prior_history(monkeypatch: pytest.MonkeyPatch) -> None:
    saved = {}
    frame = pd.DataFrame({
        "open_time": [index * 600_000 for index in range(12)],
        "close": [100.0 + index for index in range(12)],
        "factor_a": [float(index) for index in range(12)],
        "factor_b": [float(12 - index) for index in range(12)],
    })

    monkeypatch.setattr(backfill, "load_factor_frame", lambda *_args: frame)
    monkeypatch.setattr(backfill, "oriented_score_search", lambda history: _score_search(history))
    monkeypatch.setattr(backfill, "ranked_hit_search", lambda history, _scores, _config: _ranked_search(history))
    monkeypatch.setattr(
        backfill,
        "save_factor_combo_feature_snapshots",
        lambda symbol, duration, snapshots: saved.update(
            {"symbol": symbol, "duration": duration, "snapshots": snapshots}
        ),
    )

    report = backfill.backfill_factor_combo_feature_snapshots(
        "btcusdt",
        "10m",
        backfill.FactorComboSnapshotBackfillConfig(
            lookback_rows=6,
            step_rows=3,
            min_history_rows=4,
            ranking_limit=2,
            candidate_limit=2,
            min_trades=2,
        ),
    )

    assert report["saved"] == 4
    assert saved["symbol"] == "BTCUSDT"
    assert saved["duration"] == "10m"
    assert [row["entryOpenTime"] for row in saved["snapshots"]] == [
        3_000_000,
        4_800_000,
        6_600_000,
        7_200_000,
    ]
    assert saved["snapshots"][0]["ranking"][0]["totalPeriods"] == 4
    assert saved["snapshots"][1]["ranking"][0]["totalPeriods"] == 6


def test_backfill_rejects_invalid_window_config() -> None:
    with pytest.raises(ValueError, match="min_history_rows"):
        backfill.backfill_factor_combo_feature_snapshots(
            "BTCUSDT",
            "10m",
            backfill.FactorComboSnapshotBackfillConfig(lookback_rows=10, min_history_rows=11),
        )


class _ScoreSearch:
    def __init__(self) -> None:
        self.scores = {"factor_a": object(), "factor_b": object()}


class _RankedSearch:
    def __init__(self, history: pd.DataFrame) -> None:
        self.hits = [_Hit(len(history), history["factor_a"])]


class _Hit:
    def __init__(self, trades: int, score: pd.Series) -> None:
        self.members = ("factor_a", "factor_b")
        self.orientations = (1, -1)
        self.threshold = 1.0
        self.win_rate = 0.6
        self.profit_factor = 1.2
        self.trades = trades
        self.avg_return = 0.001
        self.score = score


def _score_search(_history: pd.DataFrame) -> _ScoreSearch:
    return _ScoreSearch()


def _ranked_search(history: pd.DataFrame) -> _RankedSearch:
    return _RankedSearch(history)
