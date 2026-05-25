from __future__ import annotations

from pathlib import Path

from app.services.simulation_event_demotion import evaluate_simulation_event_demotion

HIGHER_WIN_KEY = "strategy_higher_win"
LOWER_WIN_KEY = "strategy_lower_win"


def test_simulation_observation_rows_are_sorted_by_win_rate(monkeypatch, tmp_path: Path) -> None:
    samples = {
        LOWER_WIN_KEY: _event_rows(wins=10, losses=10),
        HIGHER_WIN_KEY: _event_rows(wins=12, losses=8),
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
    assert result["evaluations"][0]["maxConsecutiveWins"] == 12
    assert result["evaluations"][0]["maxConsecutiveLosses"] == 8
    assert result["evaluations"][0]["currentConsecutiveWins"] == 12
    assert result["evaluations"][0]["currentConsecutiveLosses"] == 0


class _Conn:
    def __init__(self, _path: Path) -> None:
        pass

    def close(self) -> None:
        pass


def _settled_rows(samples: dict[str, list[dict]]):
    def settled_rows(_conn, _symbol, _duration, *, strategy_key=None):
        return samples[strategy_key]

    return settled_rows


def _event_rows(*, wins: int, losses: int) -> list[dict]:
    return [
        *({"event_pnl": 1.0} for _index in range(wins)),
        *({"event_pnl": -1.0} for _index in range(losses)),
    ]
