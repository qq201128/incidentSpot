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
    assert library["factors"][0]["source"] == agent_lib.AGENT_FACTOR_SOURCE_FILE
    assert history["runs"][0]["candidates"][0]["status"] == "promoted"


def test_existing_agent_factor_is_not_promoted_again(monkeypatch, tmp_path: Path) -> None:
    _patch_paths(monkeypatch, tmp_path)
    agent_lib.process_agent_factor_candidates(_memory("factor_a"), _frame())
    result = agent_lib.process_agent_factor_candidates(_memory("factor_a"), _frame())

    library = agent_lib.load_agent_factor_library()
    statuses = [row["status"] for row in result["agentCandidatePromotion"]["records"]]

    assert len(library["factors"]) == 1
    assert statuses == ["duplicate_existing"]


def test_agent_factor_participates_in_combination_candidates(monkeypatch, tmp_path: Path) -> None:
    _patch_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(factor_mined_candidates, "mined_factor_rows_for_duration", lambda *_args: [])
    agent_lib.process_agent_factor_candidates(_memory("factor_a"), _frame())

    result = factor_mined_candidates.build_mined_candidates(_frame(), symbol="BTCUSDT", duration="10m")
    names = [item.factor.name for item in result.candidates]

    assert result.source_count == 1
    assert names == [agent_lib.load_agent_factor_library()["factors"][0]["factorName"]]


def _patch_paths(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(agent_lib, "AGENT_FACTOR_LIBRARY_PATH", tmp_path / "agent_mined_factor_library.json")
    monkeypatch.setattr(agent_lib, "AGENT_CANDIDATE_HISTORY_PATH", tmp_path / "agent_factor_candidate_history.json")


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
    returns = 0.001 * np.sin(idx / 7.0) + 0.0005 * np.cos(idx / 13.0)
    close = 100.0 * np.cumprod(1.0 + returns)
    future = pd.Series(close).pct_change(HORIZON).shift(-HORIZON).fillna(0.0)
    noise = np.random.default_rng(1).normal(0.0, 0.001, ROWS)
    return pd.DataFrame(
        {
            "open_time": np.arange(ROWS) * 60_000,
            "close": close,
            "factor_a": future + noise,
        }
    )
