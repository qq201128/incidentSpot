from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from app.services import factor_combination_background as combo_background
from app.services import factor_combination_service as combo_service
from app.services import rule_signal_service
from app.services.factor_combination_service import CombinationSearchConfig
from app.services.factor_combination_signal_service import build_live_signal_from_ranking
from app.services.factor_registry import FactorCategory, FactorDefinition, FactorDirection
from app.services.strategy_registry import FACTOR_COMBO_STRATEGY_KEY

ROWS = 260
HORIZON = 10
SECONDS_PER_MINUTE = 60
SECONDS_PER_DAY = 86_400


@pytest.fixture
def synthetic_frame() -> pd.DataFrame:
    idx = np.arange(ROWS, dtype=float)
    returns = 0.001 * np.sin(idx / 7.0) + 0.0005 * np.cos(idx / 13.0)
    close = 100.0 * np.cumprod(1.0 + returns)
    future = pd.Series(close).pct_change(HORIZON).shift(-HORIZON)
    return pd.DataFrame(
        {
            "open_time": np.arange(ROWS) * 60_000,
            "close": close,
            "factor_a": future.fillna(0.0),
            "factor_b": future.rolling(3, min_periods=1).mean().fillna(0.0),
            "factor_c": (-future).fillna(0.0),
        }
    )


@pytest.fixture
def synthetic_factors() -> list[FactorDefinition]:
    return [
        _factor("factor_a", "未来收益动量", FactorDirection.HIGHER_BETTER),
        _factor("factor_b", "平滑收益动量", FactorDirection.HIGHER_BETTER),
        _factor("factor_c", "反向收益动量", FactorDirection.LOWER_BETTER),
    ]


def test_combination_ranking_returns_win_rate_sorted_rows(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_frame: pd.DataFrame,
    synthetic_factors: list[FactorDefinition],
) -> None:
    monkeypatch.setattr(combo_service, "list_factors", lambda: synthetic_factors)
    report = combo_service.run_factor_combination_ranking_on_frame(
        synthetic_frame,
        symbol="btcusdt",
        duration="10m",
        config=CombinationSearchConfig(base_factor_limit=3, combo_sizes=(2,), result_limit=5),
    )
    ranking = report["ranking"]
    assert report["symbol"] == "BTCUSDT"
    assert report["testedCombinationCount"] == 3
    assert len(ranking) == 3
    assert ranking == sorted(ranking, key=lambda row: row["winRate"], reverse=True)
    assert ranking[0]["comboSize"] == 2
    assert len(ranking[0]["members"]) == 2
    assert ranking[0]["winRate"] is not None


def test_live_signal_uses_cached_combo_members(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_frame: pd.DataFrame,
    synthetic_factors: list[FactorDefinition],
) -> None:
    monkeypatch.setattr(combo_service, "list_factors", lambda: synthetic_factors)
    report = combo_service.run_factor_combination_ranking_on_frame(
        synthetic_frame,
        symbol="BTCUSDT",
        duration="10m",
        config=CombinationSearchConfig(base_factor_limit=3, combo_sizes=(2,), result_limit=1),
    )
    signal = build_live_signal_from_ranking(
        synthetic_frame,
        report["ranking"][0],
        symbol="BTCUSDT",
        duration="10m",
    )
    assert signal["direction"] in {"up", "down"}
    assert 0.0 <= signal["probabilityUp"] <= 1.0
    assert signal["entryPrice"] == pytest.approx(float(synthetic_frame["close"].iloc[-1]))
    assert signal["source"] == "factor_combination_ranking"


def test_rule_signal_routes_factor_combo_strategy(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_factor_combo_direction(symbol: str, duration: str, **kwargs) -> dict:
        captured.update({"symbol": symbol, "duration": duration, **kwargs})
        return {"strategy_key": FACTOR_COMBO_STRATEGY_KEY}

    monkeypatch.setattr(rule_signal_service, "predict_factor_combo_direction", fake_factor_combo_direction)
    result = rule_signal_service.predict_rule_direction(
        "btcusdt",
        "10m",
        entry_open_time=123,
        strategy_key=FACTOR_COMBO_STRATEGY_KEY,
    )
    assert result["strategy_key"] == FACTOR_COMBO_STRATEGY_KEY
    assert captured["symbol"] == "BTCUSDT"
    assert captured["entry_open_time"] == 123
    assert captured["entry_grace_ms"] > 0


def test_daily_refresh_seconds_targets_next_midnight() -> None:
    tz = combo_background.DAILY_REFRESH_TZ
    before_midnight = datetime(2026, 5, 13, 23, 59, tzinfo=tz)
    exactly_midnight = datetime(2026, 5, 13, 0, 0, tzinfo=tz)
    assert combo_background.seconds_until_next_daily_refresh(before_midnight) == SECONDS_PER_MINUTE
    assert combo_background.seconds_until_next_daily_refresh(exactly_midnight) == SECONDS_PER_DAY


def test_daily_refresh_updates_all_combo_durations(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_run(symbol: str, duration: str, config: object) -> dict:
        calls.append(("run", symbol, duration, config))
        return {"symbol": symbol, "duration": duration, "ranking": []}

    def fake_save(report: dict) -> None:
        calls.append(("save", report["symbol"], report["duration"], None))

    def fake_learning(
        symbol: str,
        duration: str,
        ranking_report: dict,
        *,
        run_llm_agent: bool,
    ) -> None:
        assert run_llm_agent is True
        calls.append(("learn", symbol, duration, ranking_report["duration"]))

    monkeypatch.setattr(combo_background, "run_factor_combination_ranking", fake_run)
    monkeypatch.setattr(combo_background, "save_cached_combination_ranking", fake_save)
    monkeypatch.setattr(combo_background, "refresh_factor_learning_memory", fake_learning)
    combo_background.refresh_symbol_combination_rankings("btcusdt")
    run_durations = [item[2] for item in calls if item[0] == "run"]
    save_durations = [item[2] for item in calls if item[0] == "save"]
    learn_durations = [item[2] for item in calls if item[0] == "learn"]
    assert run_durations == ["10m", "30m", "60m", "1d"]
    assert save_durations == run_durations
    assert learn_durations == run_durations


def _factor(name: str, description: str, direction: FactorDirection) -> FactorDefinition:
    return FactorDefinition(
        name=name,
        category=FactorCategory.RETURN,
        description=description,
        formula=name,
        direction=direction,
    )
