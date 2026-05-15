from __future__ import annotations

import pandas as pd
import pytest

from app.services import factor_backtest_service, factor_catalog, factor_mined_library, factor_registry


def test_catalog_lists_static_and_mined_factors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(factor_mined_library, "load_mined_factor_library", lambda *_args: _library())

    payloads = factor_catalog.list_single_factor_payloads()
    combo_payloads = factor_catalog.list_combo_factor_payloads()
    names = {row["name"] for row in payloads}
    combo_names = {row["name"] for row in combo_payloads}
    categories = factor_catalog.list_single_factor_categories()

    assert "ret_1" in names
    assert "goal_combo__factor_a__factor_b" not in names
    assert "goal_combo__factor_a__factor_b" in combo_names
    assert factor_catalog.get_factor_payload_by_name("goal_combo__factor_a__factor_b")["sourceFile"] == (
        "mined_factor_library.json"
    )
    performance = next(item for item in categories if item["key"] == "performance")
    static_performance = next(item for item in factor_registry.list_factor_categories() if item["key"] == "performance")
    assert performance["count"] == static_performance["count"]


def test_catalog_uses_symbol_duration_for_mined_backtest_definition(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(factor_mined_library, "load_mined_factor_library", lambda *_args: _library())

    factor = factor_catalog.factor_definition_for_backtest(
        "goal_combo__factor_a__factor_b",
        "btcusdt",
        "10m",
    )

    assert factor.source_file == "mined_factor_library.json"
    assert factor.parameters["members"] == ["factor_a", "factor_b"]


def test_mined_factor_backtest_materializes_combo(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pd.DataFrame(
        {
            "open_time": [idx * 60_000 for idx in range(130)],
            "close": [100 + idx for idx in range(130)],
            "factor_a": [idx / 100 for idx in range(130)],
            "factor_b": [idx / 200 for idx in range(130)],
        }
    )
    monkeypatch.setattr(factor_mined_library, "load_mined_factor_library", lambda *_args: _library())
    monkeypatch.setattr(factor_backtest_service, "load_factor_frame", lambda *_args: frame)

    result = factor_backtest_service.run_factor_backtest(
        "goal_combo__factor_a__factor_b",
        "BTCUSDT",
        "10m",
    )

    assert result["factorName"] == "goal_combo__factor_a__factor_b"
    assert result["category"] == "performance"
    assert result["totalPeriods"] > 0


def _library() -> dict:
    return {
        "version": "mined_factor_library_v1",
        "factors": [
            {
                "symbol": "BTCUSDT",
                "duration": "10m",
                "factorName": "goal_combo__factor_a__factor_b",
                "factorDisplayName": "组合：factor_a + factor_b",
                "description": "组合：factor_a + factor_b",
                "formula": "oriented_zscore_pair_threshold_v1(factor_a, factor_b)",
                "category": "performance",
                "source": "factor_combo_ranking",
                "members": [
                    {"name": "factor_a", "displayName": "factor_a", "orientation": 1},
                    {"name": "factor_b", "displayName": "factor_b", "orientation": 1},
                ],
                "metrics": {"winRate": 0.8, "profitFactor": 2.0},
                "promotionCount": 2,
                "lastSeenAt": "2026-05-15T10:00:00Z",
            }
        ],
    }
