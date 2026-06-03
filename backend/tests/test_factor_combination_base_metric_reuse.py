from __future__ import annotations

import pandas as pd

from app.services import factor_combination_service as combo_service
from app.services.factor_combination_service import CombinationSearchConfig
from app.services.factor_registry import FactorCategory, FactorDefinition, FactorDirection


def test_base_candidates_reuse_valid_metric_cache(monkeypatch) -> None:
    frame = _frame()
    cached = {"factor_a": _metrics(win_rate=0.61)}
    factor = _factor("factor_a")

    monkeypatch.setattr(combo_service, "list_factors", lambda: [factor])
    monkeypatch.setattr(combo_service, "run_factor_backtest_on_frame", _fail_backtest)

    candidates, failures = combo_service._base_candidates(
        combo_service._CombinationContext(frame, "BTCUSDT", "10m"),
        metric_cache=cached,
    )

    assert failures == []
    assert candidates[0].factor.name == "factor_a"
    assert candidates[0].metrics["winRate"] == 0.61


def test_base_candidates_backtest_missing_or_invalid_cache(monkeypatch) -> None:
    frame = _frame()
    factors = [_factor("factor_a"), _factor("factor_b")]
    calls = []

    def backtest(factor, *_args, **_kwargs) -> dict:
        calls.append(factor.name)
        return _metrics(win_rate=0.62)

    monkeypatch.setattr(combo_service, "list_factors", lambda: factors)
    monkeypatch.setattr(combo_service, "run_factor_backtest_on_frame", backtest)

    candidates, failures = combo_service._base_candidates(
        combo_service._CombinationContext(frame, "BTCUSDT", "10m"),
        metric_cache={"factor_a": _metrics(win_rate=0.50, total_periods=10)},
    )

    assert failures == []
    assert calls == ["factor_a", "factor_b"]
    assert [item.factor.name for item in candidates] == ["factor_a", "factor_b"]


def test_ranking_entry_passes_base_metric_cache(monkeypatch) -> None:
    frame = _frame()
    captured = {}

    def rank_on_frame(frame_arg, **kwargs) -> dict:
        captured["frame"] = frame_arg
        captured["cache"] = kwargs["base_metric_cache"]
        return {"ok": True}

    monkeypatch.setattr(combo_service, "load_factor_frame", lambda *_args: frame)
    monkeypatch.setattr(combo_service, "cached_factor_metrics_by_name", lambda *_args: {"factor_a": _metrics()})
    monkeypatch.setattr(combo_service, "run_factor_combination_ranking_on_frame", rank_on_frame)

    result = combo_service.run_factor_combination_ranking(
        "btcusdt",
        "10m",
        CombinationSearchConfig(base_factor_limit=2, combo_sizes=(2,)),
    )

    assert result == {"ok": True}
    assert captured == {"frame": frame, "cache": {"factor_a": _metrics()}}


def _frame() -> pd.DataFrame:
    return pd.DataFrame({
        "open_time": [0, 1, 2],
        "close": [100.0, 101.0, 102.0],
        "factor_a": [0.1, 0.2, 0.3],
        "factor_b": [0.3, 0.2, 0.1],
    })


def _factor(name: str) -> FactorDefinition:
    return FactorDefinition(
        name=name,
        category=FactorCategory.RETURN,
        description=name,
        formula=name,
        direction=FactorDirection.NEUTRAL,
    )


def _metrics(*, win_rate: float = 0.60, total_periods: int = 100) -> dict:
    return {"totalPeriods": total_periods, "winRate": win_rate, "ir": 0.1}


def _fail_backtest(*_args, **_kwargs) -> dict:
    raise AssertionError("cached factor metrics should avoid backtest")
