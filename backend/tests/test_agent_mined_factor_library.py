from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.services import agent_mined_factor_library as agent_lib
from app.services import factor_mined_candidates

ROWS = 1300
HORIZON = 10


def test_agent_candidate_is_recorded_and_promoted(monkeypatch, tmp_path: Path) -> None:
    _patch_paths(monkeypatch, tmp_path)
    memory = _memory("factor_a")
    result = agent_lib.process_agent_factor_candidates(memory, _frame())

    promotion = result["agentCandidatePromotion"]
    library = agent_lib.load_agent_factor_library()
    history = json.loads((tmp_path / "agent_factor_candidate_history.json").read_text(encoding="utf-8"))

    assert promotion["candidateCount"] == 1
    assert promotion["promoted"] == 1
    assert result["agentCandidateEvaluation"]["generatedCount"] == 1
    assert result["agentCandidateEvaluation"]["promotedCount"] == 1
    assert result["llmAgent"]["review"]["evaluation"]["promotedCount"] == 1
    assert library["factors"][0]["source"] == agent_lib.AGENT_FACTOR_SOURCE_FILE
    assert history["runs"][0]["evaluation"]["promotedCount"] == 1
    assert history["runs"][0]["candidates"][0]["status"] == "promoted"


def test_existing_agent_factor_is_not_promoted_again(monkeypatch, tmp_path: Path) -> None:
    _patch_paths(monkeypatch, tmp_path)
    agent_lib.process_agent_factor_candidates(_memory("factor_a"), _frame())
    result = agent_lib.process_agent_factor_candidates(_memory("factor_a"), _frame())

    library = agent_lib.load_agent_factor_library()
    statuses = [row["status"] for row in result["agentCandidatePromotion"]["records"]]

    assert len(library["factors"]) == 1
    assert statuses == ["duplicate_existing"]


def test_evaluation_counts_only_current_records(monkeypatch, tmp_path: Path) -> None:
    _patch_paths(monkeypatch, tmp_path)
    _write_existing_library(tmp_path, count=5)

    result = agent_lib.process_agent_factor_candidates(_memory("missing_column"), _frame())
    evaluation = result["agentCandidateEvaluation"]

    assert evaluation["generatedCount"] == 1
    assert evaluation["promotedCount"] == 0
    assert evaluation["rejectedCount"] == 1
    assert evaluation["statusCounts"]["failed"] == 1
    assert evaluation["topPromotedFactors"] == []


def test_agent_factor_participates_in_combination_candidates(monkeypatch, tmp_path: Path) -> None:
    _patch_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(factor_mined_candidates, "mined_factor_rows_for_duration", lambda *_args: [])
    agent_lib.process_agent_factor_candidates(_memory("factor_a"), _frame())

    result = factor_mined_candidates.build_mined_candidates(_frame(), symbol="BTCUSDT", duration="10m")
    names = [item.factor.name for item in result.candidates]

    assert result.source_count == 1
    assert names == [agent_lib.load_agent_factor_library()["factors"][0]["factorName"]]


def test_agent_candidate_with_ema_can_materialize_and_reach_backtest(monkeypatch, tmp_path: Path) -> None:
    _patch_paths(monkeypatch, tmp_path)
    result = agent_lib.process_agent_factor_candidates(_memory("EMA(close, 12)"), _frame())
    record = result["agentCandidatePromotion"]["records"][0]

    assert record["status"] != "failed"
    assert "unsupported formula function" not in str(record)
    assert isinstance(record["metrics"], dict)


def test_agent_candidate_with_vwap_can_materialize_and_reach_backtest(monkeypatch, tmp_path: Path) -> None:
    _patch_paths(monkeypatch, tmp_path)
    result = agent_lib.process_agent_factor_candidates(_memory("VWAP(close, volume, 20)"), _frame())
    record = result["agentCandidatePromotion"]["records"][0]

    assert record["status"] != "failed"
    assert "unsupported formula function" not in str(record)
    assert isinstance(record["metrics"], dict)


def _patch_paths(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(agent_lib, "AGENT_FACTOR_LIBRARY_PATH", tmp_path / "agent_mined_factor_library.json")
    monkeypatch.setattr(agent_lib, "AGENT_CANDIDATE_HISTORY_PATH", tmp_path / "agent_factor_candidate_history.json")


def _write_existing_library(tmp_path: Path, count: int) -> None:
    rows = [_existing_library_row(index) for index in range(count)]
    payload = {"version": agent_lib.AGENT_FACTOR_LIBRARY_VERSION, "factors": rows}
    path = tmp_path / "agent_mined_factor_library.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _existing_library_row(index: int) -> dict:
    return {
        "symbol": "BTCUSDT",
        "duration": "10m",
        "factorName": f"existing_{index}",
        "factorDisplayName": f"Existing {index}",
        "formula": f"existing_formula_{index}",
        "source": agent_lib.AGENT_FACTOR_SOURCE_FILE,
        "idea": {},
        "metrics": {"winRate": 0.6, "profitFactor": 1.2},
        "score": 10.0,
        "firstSeenAt": "2026-05-17T00:00:00+00:00",
        "lastSeenAt": "2026-05-17T00:00:00+00:00",
        "promotionCount": 1,
    }


def _memory(formula: str) -> dict:
    return {
        "symbol": "BTCUSDT",
        "duration": "10m",
        "llmAgent": {
            "review": {
                "factorMiningPlan": {
                    "candidateFactorIdeas": [
                        {
                            "nameHint": "agentMomentum",
                            "displayNameZh": "Agent动量",
                            "formulaHint": formula,
                            "operatorTrace": [],
                        }
                    ]
                }
            }
        },
    }


def _frame() -> pd.DataFrame:
    idx = np.arange(ROWS, dtype=float)
    returns = 0.012 * np.sin(idx / 7.0) + 0.006 * np.cos(idx / 13.0)
    close = 100.0 * np.cumprod(1.0 + returns)
    future = pd.Series(close).pct_change(HORIZON).shift(-HORIZON).fillna(0.0)
    noise = np.random.default_rng(1).normal(0.0, 0.001, ROWS)
    return pd.DataFrame(
        {
            "open_time": np.arange(ROWS) * 60_000,
            "close": close,
            "high": close * 1.002,
            "low": close * 0.998,
            "volume": 100.0 + 20.0 * (1.0 + np.sin(idx / 11.0)),
            "factor_a": future + noise,
        }
    )
