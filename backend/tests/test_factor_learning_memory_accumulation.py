from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services import factor_learning_core
from app.services.factor_learning_core import build_factor_learning_memory

ROWS = 130
PREDICTION_START = 100
PREDICTION_COUNT = 16


def test_factor_learning_memory_accumulates_success_patterns_and_forbidden_regions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(factor_learning_core, "utc_now", lambda: "2026-05-18T00:00:00+00:00")
    memory = build_factor_learning_memory(
        _learning_frame(),
        _ranking_report(),
        _settled_predictions(),
        symbol="BTCUSDT",
        duration="10m",
        previous_memory=_previous_memory(),
    )

    patterns = {item["pattern"]: item for item in memory["factorMining"]["successPatterns"]}
    regions = {item["region"]: item for item in memory["factorMining"]["forbiddenRegions"]}
    assert patterns["category:return"]["support"] > 10
    assert "legacy_factor" in patterns["category:return"]["factors"]
    assert patterns["category:return"]["firstSeenAt"] == "2026-05-17T00:00:00+00:00"
    assert patterns["category:return"]["lastSeenAt"] == "2026-05-18T00:00:00+00:00"
    assert patterns["operator:legacy"]["support"] == 2
    assert regions["correlation_cluster:factor_a"]["support"] > 2
    assert "legacy_factor" in regions["correlation_cluster:factor_a"]["members"]
    assert regions["correlation_cluster:legacy"]["support"] == 2
    assert "correlation_cluster:combo__factor_a__factor_b" not in regions
    assert _forbidden_regions_have_no_combo_members(memory)


def test_factor_learning_forbidden_regions_skip_combo_factor_columns() -> None:
    frame = _learning_frame()
    frame["combo__factor_a__factor_b"] = frame["factor_a"] + frame["factor_b"]
    report = _ranking_report()
    report["baseFactors"].append(
        _base_factor("combo__factor_a__factor_b", "performance", win_rate=0.64, ir=0.90, sharpe=1.30)
    )

    memory = build_factor_learning_memory(
        frame,
        report,
        _settled_predictions(),
        symbol="BTCUSDT",
        duration="10m",
    )

    assert _forbidden_regions_have_no_combo_members(memory)


def _previous_memory() -> dict:
    return {
        "factorMining": {
            "successPatterns": [
                _success_pattern("category:return", "category=return", 10),
                _success_pattern("operator:legacy", "operator=legacy", 2),
            ],
            "forbiddenRegions": [
                _forbidden_region("correlation_cluster:factor_a", 2, ["legacy_factor"]),
                _forbidden_region("correlation_cluster:legacy", 2, ["legacy_a", "legacy_b"]),
                _forbidden_region(
                    "correlation_cluster:combo__factor_a__factor_b",
                    2,
                    ["combo__factor_a__factor_b", "factor_a"],
                ),
            ],
        }
    }


def _success_pattern(pattern: str, label: str, support: int) -> dict:
    payload = {
        "pattern": pattern,
        "label": label,
        "support": support,
        "score": 1.0,
        "factors": ["legacy_factor"],
    }
    if pattern == "category:return":
        payload["firstSeenAt"] = "2026-05-17T00:00:00+00:00"
    return payload


def _forbidden_region(region: str, support: int, members: list[str]) -> dict:
    return {
        "region": region,
        "reason": "redundant_factor_neighborhood",
        "support": support,
        "avgAbsCorrelation": 0.5,
        "members": members,
    }


def _forbidden_regions_have_no_combo_members(memory: dict) -> bool:
    for region in memory["factorMining"]["forbiddenRegions"]:
        seed = str(region["region"]).removeprefix("correlation_cluster:")
        names = [seed, *region.get("members", [])]
        if any(str(name).startswith(("combo__", "goal_combo__")) for name in names):
            return False
    return True


def _learning_frame() -> pd.DataFrame:
    base = np.linspace(-0.2, 0.2, ROWS)
    factor_a = base.copy()
    factor_a[PREDICTION_START:PREDICTION_START + 8] = -2.0
    factor_a[PREDICTION_START + 8:PREDICTION_START + PREDICTION_COUNT] = 4.0
    factor_a[-1] = 4.0
    return pd.DataFrame({
        "open_time": np.arange(ROWS) * 60_000,
        "close": 100 + np.arange(ROWS) * 0.1,
        "factor_a": factor_a,
        "factor_b": factor_a * 0.9 + 0.01,
        "factor_c": -base,
    })


def _settled_predictions() -> list[dict]:
    return [
        {
            "open_time": (PREDICTION_START + offset) * 60_000,
            "direction": "up",
            "confidence": 0.62,
            "actual_return": -0.01 if offset >= 8 else 0.01,
            "prediction_correct": 0 if offset >= 8 else 1,
            "high_winrate_rule": "combo__factor_a__factor_b",
        }
        for offset in range(PREDICTION_COUNT)
    ]


def _ranking_report() -> dict:
    return {
        "symbol": "BTCUSDT",
        "duration": "10m",
        "total": 1,
        "baseFactorCount": 3,
        "baseFactors": [
            _base_factor("factor_a", "return", win_rate=0.62, ir=0.80, sharpe=1.20),
            _base_factor("factor_b", "return", win_rate=0.58, ir=0.40, sharpe=0.80),
            _base_factor("factor_c", "momentum", win_rate=0.49, ir=-0.10, sharpe=0.90),
        ],
        "ranking": [{
            "factorName": "combo__factor_a__factor_b",
            "factorDisplayName": "组合：A + B",
            "winRate": 0.63,
            "profitFactor": 1.12,
            "totalPeriods": ROWS,
            "members": [
                {"name": "factor_a", "category": "return", "orientation": 1, "singleWinRate": 0.62},
                {"name": "factor_b", "category": "return", "orientation": 1, "singleWinRate": 0.58},
            ],
        }],
    }


def _base_factor(name: str, category: str, *, win_rate: float, ir: float, sharpe: float) -> dict:
    return {
        "name": name,
        "category": category,
        "formula": f"{name}.rolling(10).mean()",
        "sourceFile": "test.py",
        "winRate": win_rate,
        "ir": ir,
        "sharpe": sharpe,
        "profitFactor": 1.10,
    }
