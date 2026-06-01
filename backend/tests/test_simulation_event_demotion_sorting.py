from __future__ import annotations

from pathlib import Path

from app.services.simulation_event_demotion import evaluate_simulation_event_demotion

HIGHER_WIN_KEY = "strategy_higher_win"
LOWER_WIN_KEY = "strategy_lower_win"


def test_simulation_observation_rows_are_sorted_by_win_rate(monkeypatch, tmp_path: Path) -> None:
    samples = {
        LOWER_WIN_KEY: _losing_streak_rows(wins=15, losses=15),
        HIGHER_WIN_KEY: _losing_streak_rows(wins=20, losses=10),
    }
    monkeypatch.setattr("app.db.session.get_conn", lambda: _Conn(tmp_path))
    monkeypatch.setattr("app.services.event_pnl_rows.settled_event_metric_rows", _settled_rows(samples))

    result = evaluate_simulation_event_demotion(
        "BTCUSDT",
        "10m",
        source="test",
        list_strategy_keys=lambda *_args: [LOWER_WIN_KEY, HIGHER_WIN_KEY],
        validate_strategy_key=lambda _key: True,
    )

    assert [row["strategyKey"] for row in result["evaluations"]] == [HIGHER_WIN_KEY, LOWER_WIN_KEY]
    assert [row["strategyKey"] for row in result["watchlist"]] == [HIGHER_WIN_KEY, LOWER_WIN_KEY]
    assert result["evaluations"][0]["winRate"] == 0.6667
    assert result["evaluations"][0]["maxConsecutiveWins"] == 20
    assert result["evaluations"][0]["maxConsecutiveLosses"] == 10
    assert result["evaluations"][0]["currentConsecutiveWins"] == 0
    assert result["evaluations"][0]["currentConsecutiveLosses"] == 10


def test_simulation_observation_uses_full_settled_event_history(monkeypatch, tmp_path: Path) -> None:
    recorder = _SettledRowsRecorder({HIGHER_WIN_KEY: _event_rows(wins=35, losses=10)})
    monkeypatch.setattr("app.db.session.get_conn", lambda: _Conn(tmp_path))
    monkeypatch.setattr("app.services.event_pnl_rows.settled_event_metric_rows", recorder.settled_rows)

    result = evaluate_simulation_event_demotion(
        "BTCUSDT",
        "10m",
        source="test",
        list_strategy_keys=lambda *_args: [HIGHER_WIN_KEY],
        validate_strategy_key=lambda _key: True,
    )

    assert recorder.calls == [{"strategy_key": HIGHER_WIN_KEY, "limit": None}]
    assert result["evaluations"][0]["sampleCount"] == 45
    assert result["evaluations"][0]["winRate"] == 0.7778


class _Conn:
    def __init__(self, _path: Path) -> None:
        pass

    def close(self) -> None:
        pass


def _settled_rows(samples: dict[str, list[dict]]):
    def settled_rows(_conn, _symbol, _duration, *, strategy_key=None, limit=30):
        return samples[strategy_key]

    return settled_rows


class _SettledRowsRecorder:
    def __init__(self, samples: dict[str, list[dict]]) -> None:
        self.samples = samples
        self.calls: list[dict] = []

    def settled_rows(self, _conn, _symbol, _duration, *, strategy_key=None, limit=30):
        self.calls.append({"strategy_key": strategy_key, "limit": limit})
        return self.samples[strategy_key]


def _event_rows(*, wins: int, losses: int) -> list[dict]:
    return [
        *({"event_pnl": 1.0} for _index in range(wins)),
        *({"event_pnl": -1.0} for _index in range(losses)),
    ]


def _losing_streak_rows(*, wins: int, losses: int) -> list[dict]:
    return [
        *({"event_pnl": -1.0} for _index in range(losses)),
        *({"event_pnl": 1.0} for _index in range(wins)),
    ]
