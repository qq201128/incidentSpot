from __future__ import annotations

import json
from pathlib import Path

from app.services import agent_formula_dedup as dedup
from app.services import agent_mined_factor_library as agent_lib


def test_known_agent_formulas_include_rejected_library_rows(monkeypatch, tmp_path: Path) -> None:
    library_path = tmp_path / "agent_mined_factor_library.json"
    library_path.write_text(
        json.dumps(
            {
                "factors": [
                    {
                        "symbol": "BTCUSDT",
                        "duration": "10m",
                        "formula": "TsZScore(FundingZ(20), 60)",
                        "qualityPassed": False,
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(dedup, "AGENT_FACTOR_LIBRARY_PATH", library_path)
    monkeypatch.setattr(
        dedup,
        "load_agent_candidate_history",
        lambda *_args, **_kwargs: {"runs": []},
    )

    known = dedup.known_agent_formulas("BTCUSDT", "10m")

    assert "TsZScore(FundingZ(20), 60)" in known


def test_known_agent_formulas_ignore_rejected_history_only(monkeypatch, tmp_path: Path) -> None:
    library_path = tmp_path / "agent_mined_factor_library.json"
    library_path.write_text(
        json.dumps({"factors": []}, ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(dedup, "AGENT_FACTOR_LIBRARY_PATH", library_path)
    monkeypatch.setattr(
        dedup,
        "load_agent_candidate_history",
        lambda *_args, **_kwargs: {
            "runs": [
                {
                    "symbol": "BTCUSDT",
                    "duration": "10m",
                    "candidates": [
                        {
                            "formula": "TsRank(obv_slope_20, 40)",
                            "status": "rejected_metrics",
                        },
                        {
                            "formula": "TsZScore(FundingZ(20), 60)",
                            "status": "duplicate_existing",
                        },
                    ],
                }
            ]
        },
    )

    known = dedup.known_agent_formulas("BTCUSDT", "10m")

    assert known == frozenset({"TsZScore(FundingZ(20), 60)"})


def test_filter_duplicate_factor_ideas_drops_known_formulas() -> None:
    known = frozenset({"TsZScore(FundingZ(20), 60)"})
    ideas = [
        {"formulaHint": "TsZScore(FundingZ(20), 60)", "nameHint": "dup"},
        {"formulaHint": "TsRank(obv_slope_20, 40)", "nameHint": "fresh"},
    ]

    kept, dropped = dedup.filter_duplicate_factor_ideas(ideas, known)

    assert len(kept) == 1
    assert kept[0]["nameHint"] == "fresh"
    assert len(dropped) == 1


def test_process_agent_factor_candidates_skips_known_formulas(monkeypatch, tmp_path: Path) -> None:
    library_path = tmp_path / "agent_mined_factor_library.json"
    library_path.write_text(
        json.dumps(
            {
                "factors": [
                    {
                        "symbol": "BTCUSDT",
                        "duration": "10m",
                        "factorName": "agent__funding_z_20_tszscore_60__23572d7c1f",
                        "factorDisplayName": "资金费率Z值的时间序列Z值",
                        "formula": "TsZScore(FundingZ(20), 60)",
                        "candidateStatus": "rejected_metrics",
                        "qualityPassed": False,
                        "metrics": {"winRate": 0.2, "profitFactor": 0.5},
                        "score": 0.0,
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(agent_lib, "AGENT_FACTOR_LIBRARY_PATH", library_path)
    monkeypatch.setattr(agent_lib, "AGENT_CANDIDATE_HISTORY_PATH", tmp_path / "agent_factor_candidate_history.json")
    monkeypatch.setattr(dedup, "AGENT_FACTOR_LIBRARY_PATH", library_path)
    monkeypatch.setattr(
        dedup,
        "load_agent_candidate_history",
        lambda *_args, **_kwargs: {"runs": []},
    )

    memory = {
        "symbol": "BTCUSDT",
        "duration": "10m",
        "llmAgent": {
            "review": {
                "factorMiningPlan": {
                    "candidateFactorIdeas": [
                        {
                            "nameHint": "funding_z_20_tszscore_60",
                            "displayNameZh": "资金费率Z值的时间序列Z值",
                            "formulaHint": "TsZScore(FundingZ(20), 60)",
                        }
                    ]
                }
            }
        },
    }

    result = agent_lib.process_agent_factor_candidates(memory, _frame())

    assert result["agentCandidatePromotion"]["candidateCount"] == 0
    assert result["agentCandidatePromotion"]["records"] == []


def _frame():
    import numpy as np
    import pandas as pd

    rows = 200
    idx = np.arange(rows, dtype=float)
    close = 100.0 + np.sin(idx / 5.0)
    return pd.DataFrame(
        {
            "open_time": np.arange(rows) * 60_000,
            "close": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "volume": 100.0 + idx,
            "factor_a": np.random.default_rng(1).normal(0.0, 0.001, rows),
        }
    )
