from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from typing import Any

from app.services import factor_combination_live_service as combo_live
from app.services import factor_combination_signal_service as combo_signal
from app.services import factor_combination_service as combo_service
from app.services import factor_learning_signal_filter
from app.services import rule_signal_service
from app.services.agent_mined_factor_library import AGENT_FACTOR_SOURCE_FILE
from app.services.factor_combination_service import CombinationSearchConfig
from app.services.factor_combination_signal_service import (
    LIVE_MIN_PROFIT_FACTOR,
    LIVE_MIN_WIN_RATE,
    build_live_signal_from_ranking,
)
from app.services.factor_combination_ranker import pairwise_diversity_payload
from app.services.factor_combo_simulation_keys import (
    factor_combo_shadow_strategy_key,
    high_winrate_factor_combo_simulation_strategy_key,
)
from app.services.factor_combination_walk_forward import walk_forward_validation
from app.services.factor_mined_candidates import MINED_FACTOR_SOURCE_FILE
from app.services.factor_mined_candidates import MinedCandidateResult, MinedFrameResult
from app.services.factor_registry import FactorCategory, FactorDefinition, FactorDirection
from app.services.strategy_registry import FACTOR_COMBO_STRATEGY_KEY, HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY

ROWS = 1300
HORIZON = 1


@pytest.fixture(autouse=True)
def no_high_winrate_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(combo_live, "get_cached_high_winrate_combo_ranking", lambda *_args: None)


@pytest.fixture
def synthetic_frame() -> pd.DataFrame:
    idx = np.arange(ROWS, dtype=float)
    returns = 0.006 * np.sin(idx / 7.0) + 0.003 * np.cos(idx / 13.0)
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
    def empty(frame: pd.DataFrame, *, symbol: str, duration: str, **_kwargs) -> MinedCandidateResult:
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


def test_combination_ranking_applies_loss_memory_filters(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_frame: pd.DataFrame,
    synthetic_factors: list[FactorDefinition],
) -> None:
    monkeypatch.setattr(combo_service, "list_factors", lambda: synthetic_factors)
    monkeypatch.setattr(
        combo_service,
        "load_factor_learning_memory_for",
        lambda *_args: {
            "factorMining": {"forbiddenRegions": []},
            "lossMemory": {"patterns": [{"feature": "factor_b"}]},
            "weights": {"factor_a": 0.7, "factor_c": 0.3},
        },
    )

    report = combo_service.run_factor_combination_ranking_on_frame(
        synthetic_frame,
        symbol="BTCUSDT",
        duration="10m",
        config=CombinationSearchConfig(base_factor_limit=3, combo_sizes=(2,), result_limit=5),
    )

    base_names = [item["name"] for item in report["baseFactors"]]
    factor_a = next(item for item in report["baseFactors"] if item["name"] == "factor_a")

    assert "factor_b" not in base_names
    assert report["baseFactorCount"] == 2
    assert factor_a["learningWeight"] == 0.7
    assert factor_a["learningScore"] == pytest.approx(factor_a["factorScore"] + 0.7)


def test_combination_ranking_ignores_mining_forbidden_regions(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_frame: pd.DataFrame,
    synthetic_factors: list[FactorDefinition],
) -> None:
    monkeypatch.setattr(combo_service, "list_factors", lambda: synthetic_factors)
    monkeypatch.setattr(
        combo_service,
        "load_factor_learning_memory_for",
        lambda *_args: {
            "factorMining": {"forbiddenRegions": [{"members": ["factor_b"]}]},
            "lossMemory": {"patterns": []},
            "weights": {},
        },
    )

    report = combo_service.run_factor_combination_ranking_on_frame(
        synthetic_frame,
        symbol="BTCUSDT",
        duration="10m",
        config=CombinationSearchConfig(base_factor_limit=3, combo_sizes=(2,), result_limit=5),
    )

    base_names = [item["name"] for item in report["baseFactors"]]

    assert "factor_b" in base_names
    assert report["baseFactorCount"] == 3


def test_combination_ranking_skips_mined_combo_candidates_but_keeps_agent_factors(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_frame: pd.DataFrame,
    synthetic_factors: list[FactorDefinition],
) -> None:
    monkeypatch.setattr(combo_service, "list_factors", lambda: synthetic_factors[:2])
    monkeypatch.setattr(
        combo_service,
        "build_mined_candidates",
        lambda frame, **_kwargs: MinedCandidateResult(
            frame,
            (
                _mined_candidate(
                    "combo__factor_a__factor_b",
                    "组合：未来收益动量 + 平滑收益动量",
                    MINED_FACTOR_SOURCE_FILE,
                ),
                _mined_candidate("agent_alpha", "Agent单因子", AGENT_FACTOR_SOURCE_FILE),
            ),
            2,
            (),
        ),
    )

    report = combo_service.run_factor_combination_ranking_on_frame(
        synthetic_frame,
        symbol="BTCUSDT",
        duration="10m",
        config=CombinationSearchConfig(base_factor_limit=3, combo_sizes=(2,), result_limit=5),
    )

    base_names = [item["name"] for item in report["baseFactors"]]
    ranking_names = [item["factorName"] for item in report["ranking"]]

    assert "agent_alpha" in base_names
    assert "combo__factor_a__factor_b" not in base_names
    assert all("combo__combo__" not in name for name in ranking_names)


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


def test_walk_forward_validation_fails_when_median_split_has_no_edge() -> None:
    factor = _factor("factor_a", "弱边缘因子", FactorDirection.HIGHER_BETTER)
    flat_edge_frame = pd.DataFrame(
        {
            "factor_a": np.tile([0.0, 1.0], 200),
            "fwd_ret": np.full(400, 0.001, dtype=float),
        }
    )

    result = walk_forward_validation(flat_edge_frame, factor, "10m")

    assert result.passed is False
    assert result.failure_reason == "validation_win_rate_below_min"
    assert result.payload["validation"]["winRate"] == pytest.approx(0.5, abs=0.05)


def test_walk_forward_validation_reports_purge_embargo_diagnostics() -> None:
    factor = FactorDefinition(
        name="factor_a",
        category=FactorCategory.RETURN,
        description="带窗口因子",
        formula="factor_a",
        direction=FactorDirection.HIGHER_BETTER,
        parameters={"featureWindow": 7},
    )
    frame = pd.DataFrame({"factor_a": np.arange(600, dtype=float), "fwd_ret": np.full(600, 0.002)})

    result = walk_forward_validation(frame, factor, "10m")
    diagnostics = result.payload["splitDiagnostics"]

    assert diagnostics["policy"] == "chronological_train_validation_test_with_purge_embargo"
    assert diagnostics["purgeGapBars"] == 7
    assert diagnostics["embargoBars"] == 7
    assert diagnostics["purgedSampleCount"] == 14
    assert diagnostics["windows"][1]["name"] == "validation"
    assert diagnostics["windows"][1]["start"] > diagnostics["windows"][0]["end"]


def test_combination_ranking_learns_non_negative_member_weights(
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
    weights = [member["weight"] for member in report["ranking"][0]["members"]]

    assert all(weight >= 0 for weight in weights)
    assert sum(weights) == pytest.approx(1.0)


def test_neutral_base_candidate_flips_low_winrate_orientation() -> None:
    factor = _factor("inverse_alpha", "反向信号", FactorDirection.NEUTRAL)
    metrics = {"winRate": 0.18, "ir": 0.2, "totalPeriods": ROWS}

    orientation = combo_service._factor_orientation(factor, metrics)

    assert orientation == -1


def test_base_rank_key_uses_directional_winrate() -> None:
    weak_direct = _base_candidate("weak_direct", 0.54, 10.0, 1)
    strong_inverse = _base_candidate("strong_inverse", 0.18, 9.0, -1)

    ranked = sorted([weak_direct, strong_inverse], key=combo_service._base_rank_key, reverse=True)

    assert [item.factor.name for item in ranked] == ["strong_inverse", "weak_direct"]


def test_rank_combinations_excludes_walk_forward_failures(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_frame: pd.DataFrame,
) -> None:
    candidates = [_rank_filter_candidate(name) for name in ("factor_a", "factor_b", "factor_c")]

    def fake_result(_context, members):
        names = [member.factor.name for member in members]
        passed = names == ["factor_a", "factor_b"]
        return _rank_filter_row("__".join(names), passed), None

    monkeypatch.setattr(combo_service, "_combination_result", fake_result)
    ranking, tested_count, failures = combo_service._rank_combinations(
        combo_service._CombinationContext(synthetic_frame, "BTCUSDT", "10m"),
        candidates,
        CombinationSearchConfig(base_factor_limit=3, combo_sizes=(2,), result_limit=5),
    )

    assert tested_count == 3
    assert [row["factorName"] for row in ranking] == ["combo__factor_a__factor_b"]
    assert {item["error"] for item in failures} == {"validation_win_rate_below_min"}


def test_rank_combinations_returns_empty_when_no_walk_forward_combo_passes(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_frame: pd.DataFrame,
) -> None:
    candidates = [_rank_filter_candidate(name) for name in ("factor_a", "factor_b")]
    monkeypatch.setattr(
        combo_service,
        "_combination_result",
        lambda _context, members: (_rank_filter_row("__".join(member.factor.name for member in members), False), None),
    )

    ranking, tested_count, failures = combo_service._rank_combinations(
        combo_service._CombinationContext(synthetic_frame, "BTCUSDT", "10m"),
        candidates,
        CombinationSearchConfig(base_factor_limit=2, combo_sizes=(2,), result_limit=5),
    )

    assert tested_count == 1
    assert ranking == []
    assert [item["error"] for item in failures] == [
        "validation_win_rate_below_min",
        "no_walk_forward_combo_passed",
    ]


def test_combination_search_reports_prefilter_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_frame: pd.DataFrame,
) -> None:
    candidates = [_rank_filter_candidate(name) for name in ("factor_a", "factor_b", "factor_c", "factor_d")]
    seen = []

    def fake_result(_context, members):
        seen.append(tuple(member.factor.name for member in members))
        return _rank_filter_row("__".join(member.factor.name for member in members), True), None

    monkeypatch.setattr(combo_service, "_combination_result", fake_result)
    result = combo_service._rank_combinations_with_diagnostics(
        combo_service._CombinationContext(synthetic_frame, "BTCUSDT", "10m"),
        candidates,
        CombinationSearchConfig(
            base_factor_limit=4,
            combo_sizes=(2,),
            result_limit=5,
            prefilter_limit=2,
            beam_width=10,
            parallel_workers=1,
        ),
    )

    diagnostics = result.diagnostics

    assert result.tested_count == 2
    assert len(seen) == 2
    assert diagnostics["mode"] == "staged_layered_pairwise_diversity_v1"
    assert diagnostics["fullCombinationEstimate"] == 6
    assert diagnostics["generatedCombinationCount"] == 6
    assert diagnostics["prefilteredCombinationCount"] == 4
    assert diagnostics["parallelWorkers"] == 1
    assert diagnostics["searchStages"][0]["stage"] == "size_2"
    assert diagnostics["searchStages"][0]["generated"] == 6


def test_combination_search_retains_diverse_failed_pairs_for_expansion(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_frame: pd.DataFrame,
) -> None:
    candidates = [_rank_filter_candidate(name) for name in ("factor_a", "factor_b", "factor_c", "factor_d")]
    seen = []

    def fake_result(_context, members):
        names = tuple(member.factor.name for member in members)
        seen.append(names)
        passed = len(names) == 3 or names == ("factor_a", "factor_b")
        return _rank_filter_row("__".join(names), passed), None

    monkeypatch.setattr(combo_service, "_combination_result", fake_result)
    result = combo_service._rank_combinations_with_diagnostics(
        combo_service._CombinationContext(synthetic_frame, "BTCUSDT", "10m"),
        candidates,
        CombinationSearchConfig(
            base_factor_limit=4,
            combo_sizes=(2, 3),
            result_limit=10,
            prefilter_limit=10,
            beam_width=10,
            parallel_workers=1,
        ),
    )

    size_three = [names for names in seen if len(names) == 3]

    assert size_three
    assert any({"factor_a", "factor_b"} <= set(names) for names in size_three)
    assert any({"factor_a", "factor_b"} - set(names) for names in size_three)
    assert result.diagnostics["searchStages"][0]["survivors"] == 1
    assert result.diagnostics["searchStages"][0]["retainedForExpansion"] > 0
    assert result.diagnostics["searchStages"][1]["evaluated"] == len(size_three)


def test_combination_ranking_report_includes_search_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_frame: pd.DataFrame,
    synthetic_factors: list[FactorDefinition],
) -> None:
    monkeypatch.setattr(combo_service, "list_factors", lambda: synthetic_factors)

    report = combo_service.run_factor_combination_ranking_on_frame(
        synthetic_frame,
        symbol="BTCUSDT",
        duration="10m",
        config=CombinationSearchConfig(
            base_factor_limit=3,
            combo_sizes=(2,),
            result_limit=5,
            prefilter_limit=3,
            beam_width=3,
            parallel_workers=1,
        ),
    )

    assert report["searchConfig"]["prefilterLimit"] == 3
    assert report["searchConfig"]["beamWidth"] == 3
    assert report["searchConfig"]["parallelWorkers"] == 1
    assert report["searchDiagnostics"]["evaluatedCombinationCount"] == 3
    assert "failureReasonCounts" in report["searchDiagnostics"]


def test_pairwise_diversity_uses_member_to_member_correlation() -> None:
    frame = pd.DataFrame(
        {
            "factor_a": np.arange(ROWS, dtype=float),
            "factor_b": np.arange(ROWS, dtype=float) * 2.0,
            "factor_c": np.sin(np.arange(ROWS, dtype=float)),
        }
    )
    correlated = (_rank_filter_candidate("factor_a"), _rank_filter_candidate("factor_b"))
    mixed = (_rank_filter_candidate("factor_a"), _rank_filter_candidate("factor_c"))

    correlated_payload = pairwise_diversity_payload(frame, correlated)
    mixed_payload = pairwise_diversity_payload(frame, mixed)

    assert correlated_payload["pairwiseMaxAbsCorrelation"] == pytest.approx(1.0)
    assert correlated_payload["pairwiseDiversityScore"] == pytest.approx(0.0)
    assert mixed_payload["pairwiseDiversityScore"] > correlated_payload["pairwiseDiversityScore"]


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


def test_live_signal_hard_blocks_learned_loss_pattern_without_quality_gate(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_frame: pd.DataFrame,
) -> None:
    monkeypatch.setattr(
        factor_learning_signal_filter,
        "load_factor_learning_memory",
        lambda *_args: {
            "lossMemory": {"patterns": [{"feature": "factor_a", "direction": "high", "threshold": -999.0}]},
            "filters": {},
            "weights": {},
        },
    )
    row = {
        "factorName": "goal_combo__factor_a",
        "factorDisplayName": "组合：factor_a",
        "members": [{"name": "factor_a", "category": "return", "orientation": 1}],
        "method": "test",
        "threshold": 0.0,
        "winRate": 0.70,
        "profitFactor": 1.20,
        "totalPeriods": ROWS,
        "walkForwardPassed": True,
    }

    signal = build_live_signal_from_ranking(
        synthetic_frame,
        row,
        symbol="BTCUSDT",
        duration="10m",
        apply_quality_gate=False,
    )

    assert signal["qualityPassed"] is False
    assert signal["qualityGateReason"] == "factor_learning_loss_pattern_blocked"


def test_live_signal_requires_walk_forward_for_regular_combo(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_frame: pd.DataFrame,
) -> None:
    monkeypatch.setattr(factor_learning_signal_filter, "load_factor_learning_memory", lambda *_args: None)
    row = {
        "factorName": "combo__factor_a",
        "factorDisplayName": "组合：factor_a",
        "members": [{"name": "factor_a", "category": "return", "orientation": 1}],
        "method": "test",
        "threshold": 0.0,
        "winRate": 0.70,
        "profitFactor": 1.20,
        "totalPeriods": ROWS,
    }

    missing = build_live_signal_from_ranking(synthetic_frame, row, symbol="BTCUSDT", duration="10m")
    failed = build_live_signal_from_ranking(
        synthetic_frame,
        {**row, "walkForwardPassed": False, "walkForwardFailureReason": "test_win_rate_below_min"},
        symbol="BTCUSDT",
        duration="10m",
    )

    assert missing["qualityPassed"] is False
    assert missing["qualityGateReason"] == "walk_forward_missing"
    assert failed["qualityPassed"] is False
    assert failed["qualityGateReason"] == "test_win_rate_below_min"


def test_live_signal_requires_thresholded_score_for_trade_quality(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_frame: pd.DataFrame,
) -> None:
    monkeypatch.setattr(factor_learning_signal_filter, "load_factor_learning_memory", lambda *_args: None)
    frame = synthetic_frame.assign(factor_static=np.sin(np.arange(ROWS) / 11.0))
    row = {
        "factorName": "goal_combo__factor_static",
        "factorDisplayName": "组合：固定弱信号",
        "members": [{"name": "factor_static", "category": "return", "orientation": 1}],
        "method": "oriented_expanding_zscore_pair_threshold_v1",
        "threshold": 10.0,
        "winRate": 0.70,
        "profitFactor": 1.20,
        "totalPeriods": ROWS,
        "walkForwardPassed": True,
    }

    signal = build_live_signal_from_ranking(frame, row, symbol="BTCUSDT", duration="10m")

    assert signal["qualityPassed"] is False
    assert signal["qualityThresholdPassed"] is False
    assert signal["qualityGateReason"] == "signal_threshold_not_met"
    assert signal["signalThreshold"] == 10.0


def test_live_signal_uses_row_min_trades_for_goal_combo(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_frame: pd.DataFrame,
) -> None:
    monkeypatch.setattr(factor_learning_signal_filter, "load_factor_learning_memory", lambda *_args: None)
    row = {
        "factorName": "goal_combo__factor_a__factor_b",
        "factorDisplayName": "组合：factor_a + factor_b",
        "members": [{"name": "factor_a", "category": "return", "orientation": 1}],
        "method": "oriented_expanding_zscore_pair_threshold_v1",
        "threshold": 0.0,
        "winRate": 0.70,
        "profitFactor": 1.20,
        "totalPeriods": 64,
        "minTrades": 60,
    }

    signal = build_live_signal_from_ranking(synthetic_frame, row, symbol="BTCUSDT", duration="10m")

    assert signal["qualityPassed"] is True
    assert signal["qualityMinPeriods"] == 60


def test_live_signal_direction_uses_score_above_historical_median(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(factor_learning_signal_filter, "load_factor_learning_memory", lambda *_args: None)
    frame = _median_direction_frame(130)
    scores = pd.Series([-2.0] * 129 + [-1.0], index=frame.index)
    monkeypatch.setattr(combo_signal, "combination_score", lambda _frame, _members: scores)

    signal = build_live_signal_from_ranking(frame, _median_direction_row(), symbol="BTCUSDT", duration="10m")

    assert signal["score"] == -1.0
    assert signal["historicalMedianScore"] == -2.0
    assert signal["direction"] == "up"
    assert signal["probabilityUp"] == 0.7


def test_live_signal_direction_uses_score_below_historical_median(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(factor_learning_signal_filter, "load_factor_learning_memory", lambda *_args: None)
    frame = _median_direction_frame(130)
    scores = pd.Series([2.0] * 129 + [1.0], index=frame.index)
    monkeypatch.setattr(combo_signal, "combination_score", lambda _frame, _members: scores)

    signal = build_live_signal_from_ranking(frame, _median_direction_row(), symbol="BTCUSDT", duration="10m")

    assert signal["score"] == 1.0
    assert signal["historicalMedianScore"] == 2.0
    assert signal["direction"] == "down"
    assert signal["probabilityUp"] == 0.3


def test_goal_combo_live_direction_uses_threshold_sign(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(factor_learning_signal_filter, "load_factor_learning_memory", lambda *_args: None)
    frame = _median_direction_frame(130)
    scores = pd.Series([-2.0] * 129 + [-1.5], index=frame.index)
    monkeypatch.setattr(combo_signal, "combination_score", lambda _frame, _members: scores)
    row = {
        **_median_direction_row(),
        "factorName": "goal_combo__factor_signal",
        "threshold": 1.0,
    }

    signal = build_live_signal_from_ranking(frame, row, symbol="BTCUSDT", duration="10m")

    assert signal["score"] == -1.5
    assert signal["historicalMedianScore"] == -2.0
    assert signal["direction"] == "down"
    assert signal["probabilityUp"] == 0.3
    assert signal["qualityPassed"] is True


def test_live_signal_direction_requires_historical_median(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(factor_learning_signal_filter, "load_factor_learning_memory", lambda *_args: None)
    frame = _median_direction_frame(100)
    scores = pd.Series([1.0] * 100, index=frame.index)
    monkeypatch.setattr(combo_signal, "combination_score", lambda _frame, _members: scores)

    with pytest.raises(ValueError, match="insufficient historical median"):
        build_live_signal_from_ranking(frame, _median_direction_row(), symbol="BTCUSDT", duration="10m")


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
    entry_open_time = 260 * 60_000
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
        "walkForwardPassed": True,
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
    monkeypatch.setattr(combo_live, "load_factor_frame", lambda _symbol, _duration: synthetic_frame)
    monkeypatch.setattr(
        combo_live,
        "get_cached_combination_ranking",
        lambda _symbol, dur: _usable_cache(rows) if dur == "10m" else None,
    )
    monkeypatch.setattr(
        combo_live,
        "materialize_mined_factor_frame_for_rows",
        lambda frame, **_kwargs: MinedFrameResult(frame, 0, ()),
    )
    monkeypatch.setattr(combo_live, "mined_factor_rows_for_duration", lambda *_args: [])
    monkeypatch.setattr(combo_live, "build_live_signal_from_ranking", _fake_live_signal)

    payload = combo_live.rebuild_combination_signal_watchlist("BTCUSDT", limit=12, top_per_duration=3)

    assert [item["comboRank"] for item in payload["signals"]] == [1, 2, 3]
    assert [item["factorName"] for item in payload["signals"]] == ["combo_0", "combo_1", "combo_2"]
    assert payload["signals"][1]["simulationStrategyKey"] == factor_combo_shadow_strategy_key(2)
    assert payload["signals"][2]["simulationMode"] == "paper_live"
    assert payload["topPerDuration"] == 3


def test_signal_watchlist_marks_goal_combos_as_high_winrate_strategy(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_frame: pd.DataFrame,
) -> None:
    rows = [{"factorName": "goal_combo__alpha__beta"}]
    monkeypatch.setattr(combo_live, "load_factor_frame", lambda _symbol, _duration: synthetic_frame)
    monkeypatch.setattr(
        combo_live,
        "get_cached_combination_ranking",
        lambda _symbol, dur: _usable_cache(rows) if dur == "10m" else None,
    )
    monkeypatch.setattr(
        combo_live,
        "materialize_mined_factor_frame_for_rows",
        lambda frame, **_kwargs: MinedFrameResult(frame, 0, ()),
    )
    monkeypatch.setattr(combo_live, "mined_factor_rows_for_duration", lambda *_args: [])
    monkeypatch.setattr(combo_live, "build_live_signal_from_ranking", _fake_live_signal)

    payload = combo_live.rebuild_combination_signal_watchlist("BTCUSDT", limit=12, top_per_duration=3)

    assert payload["signals"][0]["comboStrategyFamily"] == "high_winrate_goal"
    assert payload["signals"][0]["simulationStrategyKey"] == high_winrate_factor_combo_simulation_strategy_key(1)


def test_signal_watchlist_cache_read_does_not_rebuild(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(combo_live, "get_cached_combination_signals", lambda _symbol: None)
    monkeypatch.setattr(combo_live, "rebuild_combination_signal_watchlist", _fail_rebuild_watchlist)

    payload = combo_live.build_combination_signal_watchlist("BTCUSDT", limit=12, top_per_duration=3)

    assert payload["source"] == "none"
    assert payload["signals"] == []
    assert payload["signalCacheStatus"]["reason"] == "signal_cache_missing"


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


def test_rule_signal_routes_high_winrate_combo_strategy(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def predict_high_winrate_combo(symbol: str, duration: str, **kwargs) -> dict:
        captured.update({"symbol": symbol, "duration": duration, **kwargs})
        return {"strategy_key": HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY}

    monkeypatch.setattr(rule_signal_service, "predict_high_winrate_factor_combo_direction", predict_high_winrate_combo)

    result = rule_signal_service.predict_rule_direction(
        "btcusdt",
        "10m",
        entry_open_time=123,
        strategy_key=HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY,
    )

    assert result["strategy_key"] == HIGH_WINRATE_FACTOR_COMBO_STRATEGY_KEY
    assert captured["symbol"] == "BTCUSDT"
    assert captured["entry_open_time"] == 123


def _fake_live_signal(
    _frame: pd.DataFrame,
    row: dict,
    *,
    symbol: str,
    duration: str,
    context=None,
    **_kwargs,
) -> dict:
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


def _median_direction_frame(rows: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open_time": np.arange(rows) * 60_000,
            "close": np.linspace(100.0, 101.0, rows),
            "factor_signal": np.linspace(-1.0, 1.0, rows),
        }
    )


def _median_direction_row() -> dict[str, Any]:
    return {
        "factorName": "combo__factor_signal",
        "factorDisplayName": "组合：factor_signal",
        "members": [{"name": "factor_signal", "category": "return", "orientation": 1}],
        "method": "test",
        "threshold": 0.0,
        "winRate": 0.70,
        "profitFactor": 1.20,
        "totalPeriods": ROWS,
        "walkForwardPassed": True,
    }


def _rank_filter_candidate(name: str):
    factor = FactorDefinition(
        name=name,
        category=FactorCategory.RETURN,
        description=name,
        formula=name,
        direction=FactorDirection.HIGHER_BETTER,
    )
    metrics = {"factorScore": 10.0, "winRate": 0.70, "sharpe": 1.0, "totalPeriods": ROWS}
    return combo_service._BaseCandidate(factor, metrics, 1)


def _base_candidate(name: str, win_rate: float, score: float, orientation: int):
    factor = FactorDefinition(
        name=name,
        category=FactorCategory.RETURN,
        description=name,
        formula=name,
        direction=FactorDirection.NEUTRAL,
    )
    return combo_service._BaseCandidate(
        factor,
        {"factorScore": score, "winRate": win_rate, "sharpe": 1.0, "totalPeriods": ROWS},
        orientation,
    )


def _rank_filter_row(name: str, passed: bool) -> dict[str, Any]:
    return {
        "factorName": "combo__" + name,
        "factorDisplayName": name,
        "comboSize": 2,
        "method": "test",
        "members": [],
        "winRate": 0.70,
        "profitFactor": 1.20,
        "sharpe": 1.0,
        "ir": 1.0,
        "totalPeriods": ROWS,
        "avgAbsCorrelation": 0.10,
        "walkForward": {},
        "walkForwardPassed": passed,
        "walkForwardFailureReason": None if passed else "validation_win_rate_below_min",
    }


def _mined_candidate(name: str, description: str, source_file: str):
    factor = FactorDefinition(
        name=name,
        category=FactorCategory.PERFORMANCE,
        description=description,
        formula=name,
        source_file=source_file,
        direction=FactorDirection.HIGHER_BETTER,
    )
    return combo_service._BaseCandidate(
        factor=factor,
        metrics={"factorScore": 5.0, "winRate": 0.62, "sharpe": 1.1, "totalPeriods": ROWS},
        orientation=1,
    )
