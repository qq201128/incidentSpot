from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services import factor_combination_live_service as combo_live
from app.services import factor_combination_service as combo_service
from app.services import factor_learning_signal_filter
from app.services import rule_signal_service
from app.services.factor_combination_service import CombinationSearchConfig
from app.services.factor_combination_signal_service import (
    LIVE_MIN_PROFIT_FACTOR,
    LIVE_MIN_WIN_RATE,
    build_live_signal_from_ranking,
)
from app.services.factor_combo_simulation_keys import factor_combo_shadow_strategy_key
from app.services.factor_mined_candidates import MinedCandidateResult, MinedFrameResult
from app.services.factor_registry import FactorCategory, FactorDefinition, FactorDirection
from app.services.strategy_registry import FACTOR_COMBO_STRATEGY_KEY

ROWS = 1300
HORIZON = 10


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


@pytest.fixture(autouse=True)
def empty_mined_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    def empty(frame: pd.DataFrame, *, symbol: str, duration: str) -> MinedCandidateResult:
        return MinedCandidateResult(frame, (), 0, ())

    monkeypatch.setattr(combo_service, "build_mined_candidates", empty)


def test_combination_ranking_returns_score_sorted_rows(
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
    assert ranking == sorted(ranking, key=lambda row: row["factorScore"], reverse=True)
    assert ranking[0]["comboSize"] == 2
    assert len(ranking[0]["members"]) == 2
    assert ranking[0]["winRate"] is not None
    assert ranking[0]["factorScore"] > 0
    assert ranking[0]["avgAbsCorrelation"] is not None
    assert report["baseFactors"][0]["factorScore"] > 0


def test_live_signal_uses_cached_combo_members(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_frame: pd.DataFrame,
    synthetic_factors: list[FactorDefinition],
) -> None:
    monkeypatch.setattr(combo_service, "list_factors", lambda: synthetic_factors)
    monkeypatch.setattr(factor_learning_signal_filter, "load_factor_learning_memory", lambda *_args: None)
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


def test_live_signal_requires_profitable_combo_for_sim_candidate(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_frame: pd.DataFrame,
    synthetic_factors: list[FactorDefinition],
) -> None:
    monkeypatch.setattr(combo_service, "list_factors", lambda: synthetic_factors)
    monkeypatch.setattr(factor_learning_signal_filter, "load_factor_learning_memory", lambda *_args: None)
    report = combo_service.run_factor_combination_ranking_on_frame(
        synthetic_frame,
        symbol="BTCUSDT",
        duration="10m",
        config=CombinationSearchConfig(base_factor_limit=3, combo_sizes=(2,), result_limit=1),
    )
    row = dict(report["ranking"][0], winRate=0.70, profitFactor=1.20, totalPeriods=ROWS)
    passed = build_live_signal_from_ranking(synthetic_frame, row, symbol="BTCUSDT", duration="10m")
    blocked = build_live_signal_from_ranking(
        synthetic_frame,
        {**row, "profitFactor": 1.0},
        symbol="BTCUSDT",
        duration="10m",
    )

    assert passed["qualityPassed"] is True
    assert passed["qualityGateReason"] == "passed"
    assert blocked["qualityPassed"] is False
    assert blocked["qualityGateReason"] == "profit_factor_below_min"
    assert blocked["qualityMinWinRate"] == LIVE_MIN_WIN_RATE
    assert blocked["qualityMinProfitFactor"] == LIVE_MIN_PROFIT_FACTOR


def test_live_signal_uses_completed_duration_entry_row(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_frame: pd.DataFrame,
    synthetic_factors: list[FactorDefinition],
) -> None:
    monkeypatch.setattr(combo_service, "list_factors", lambda: synthetic_factors)
    monkeypatch.setattr(factor_learning_signal_filter, "load_factor_learning_memory", lambda *_args: None)
    report = combo_service.run_factor_combination_ranking_on_frame(
        synthetic_frame,
        symbol="BTCUSDT",
        duration="10m",
        config=CombinationSearchConfig(base_factor_limit=3, combo_sizes=(2,), result_limit=1),
    )
    entry_open_time = 180 * 60_000
    source_open_time = entry_open_time - 10 * 60_000
    signal = build_live_signal_from_ranking(
        synthetic_frame,
        report["ranking"][0],
        symbol="BTCUSDT",
        duration="10m",
        entry_open_time=entry_open_time,
    )
    source_index = synthetic_frame.index[synthetic_frame["open_time"] == source_open_time][-1]

    assert signal["sourceOpenTime"] == source_open_time
    assert signal["entryPrice"] == pytest.approx(float(synthetic_frame.at[source_index, "close"]))
    assert signal["frameIndex"] == str(source_index)


def test_live_signal_blocks_non_kline_close_members(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_frame: pd.DataFrame,
) -> None:
    monkeypatch.setattr(factor_learning_signal_filter, "load_factor_learning_memory", lambda *_args: None)
    frame = synthetic_frame.assign(orderbook_imbalance=np.linspace(-1.0, 1.0, ROWS))
    row = {
        "factorName": "combo__orderbook_imbalance",
        "factorDisplayName": "组合：订单簿不平衡",
        "members": [{"name": "orderbook_imbalance", "category": "orderbook", "orientation": 1}],
        "method": "test",
        "winRate": 0.70,
        "profitFactor": 1.20,
        "totalPeriods": ROWS,
    }

    signal = build_live_signal_from_ranking(frame, row, symbol="BTCUSDT", duration="10m")

    assert signal["qualityPassed"] is False
    assert signal["qualityGateReason"] == "factor_timing_not_kline_close"
    assert signal["factorTimingPassed"] is False
    assert signal["factorTimingBlockedMembers"] == ["orderbook_imbalance"]


def test_signal_watchlist_returns_top_three_per_duration(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_frame: pd.DataFrame,
) -> None:
    rows = [{"factorName": f"combo_{idx}"} for idx in range(4)]
    monkeypatch.setattr(combo_live, "get_cached_combination_signals", lambda _symbol: None)
    monkeypatch.setattr(combo_live, "save_cached_combination_signals", lambda _payload: None)
    monkeypatch.setattr(combo_live, "load_factor_frame", lambda _symbol, _duration: synthetic_frame)
    monkeypatch.setattr(
        combo_live,
        "get_cached_combination_ranking",
        lambda _symbol, dur: _usable_cache(rows) if dur == "10m" else None,
    )
    monkeypatch.setattr(
        combo_live,
        "materialize_mined_factor_frame",
        lambda frame, **_kwargs: MinedFrameResult(frame, 0, ()),
    )
    monkeypatch.setattr(combo_live, "build_live_signal_from_ranking", _fake_live_signal)

    payload = combo_live.build_combination_signal_watchlist("BTCUSDT", limit=12, top_per_duration=3)

    assert [item["comboRank"] for item in payload["signals"]] == [1, 2, 3]
    assert [item["factorName"] for item in payload["signals"]] == ["combo_0", "combo_1", "combo_2"]
    assert payload["signals"][1]["simulationStrategyKey"] == factor_combo_shadow_strategy_key(2)
    assert payload["signals"][2]["simulationMode"] == "paper_live"
    assert payload["topPerDuration"] == 3


def test_signal_watchlist_uses_matching_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    cached = {
        "symbol": "BTCUSDT",
        "source": "signal_cache",
        "signals": [{"factorName": "cached"}],
        "total": 1,
        "limit": 12,
        "topPerDuration": 3,
        "durationCacheReasons": {
            "10m": "usable:u1:100:200",
            "30m": "stale::",
            "60m": "stale::",
            "1d": "stale::",
        },
    }

    monkeypatch.setattr(combo_live, "get_cached_combination_signals", lambda _symbol: cached)
    monkeypatch.setattr(
        combo_live,
        "get_cached_combination_ranking",
        lambda _symbol, dur: _usable_cache([], updated_at="u1") if dur == "10m" else None,
    )
    monkeypatch.setattr(combo_live, "rebuild_combination_signal_watchlist", _fail_rebuild_watchlist)

    payload = combo_live.build_combination_signal_watchlist("BTCUSDT", limit=12, top_per_duration=3)

    assert payload["signals"] == [{"factorName": "cached"}]
    assert payload["source"] == "signal_cache"


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


def _fake_live_signal(_frame: pd.DataFrame, row: dict, *, symbol: str, duration: str) -> dict:
    return {
        "symbol": symbol,
        "duration": duration,
        "factorName": row["factorName"],
        "factorDisplayName": row["factorName"],
        "members": [],
        "direction": "up",
        "qualityPassed": True,
    }


def _usable_cache(rows: list[dict], *, updated_at: str = "") -> dict:
    return {
        "ranking": rows,
        "updatedAt": updated_at,
        "cacheStatus": {
            "usable": True,
            "reason": "usable",
            "currentMarketData": {"rowCount": 100, "maxOpenTime": 200},
        },
    }


def _fail_rebuild_watchlist(*_args, **_kwargs) -> dict:
    raise AssertionError("matching signal cache should skip rebuild")


def _factor(name: str, description: str, direction: FactorDirection) -> FactorDefinition:
    return FactorDefinition(
        name=name,
        category=FactorCategory.RETURN,
        description=description,
        formula=name,
        direction=direction,
    )
