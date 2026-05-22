from __future__ import annotations

from app.services.mining_overview_service import _agent_candidate_rows


def test_agent_candidate_rows_match_promotion_by_name_hint() -> None:
    memory = {
        "updatedAt": "2026-05-22T15:16:34.455751+00:00",
        "llmAgent": {
            "reviewedAt": "2026-05-22T15:18:34.986676+00:00",
            "review": {
                "factorMiningPlan": {
                    "candidateFactorIdeas": [
                        {
                            "displayNameZh": "OBV斜率的时序排名",
                            "formulaHint": "TsRank(obv_slope_20, 40)",
                            "nameHint": "obv_slope_20_tsrank_40",
                        }
                    ]
                }
            },
        },
        "agentCandidatePromotion": {
            "records": [
                {
                    "displayName": "OBV斜率的时序排名",
                    "factorName": "agent__obv_slope_20_tsrank_40__5e71c388f5",
                    "formula": "TsRank(obv_slope_20, 40)",
                    "idea": {"nameHint": "obv_slope_20_tsrank_40", "formulaHint": "TsRank(obv_slope_20, 40)"},
                    "seenAt": "2026-05-22T15:18:34.986676+00:00",
                    "status": "rejected_metrics",
                }
            ]
        },
    }

    rows = _agent_candidate_rows(memory)
    assert len(rows) == 1
    assert rows[0]["validationStatusKey"] == "rejected_metrics"
    assert rows[0]["createdAt"] == "2026-05-22T15:18:34.986676+00:00"


def test_agent_candidate_rows_do_not_duplicate_promotion_records() -> None:
    memory = {
        "llmAgent": {
            "review": {
                "factorMiningPlan": {
                    "candidateFactorIdeas": [{"nameHint": "funding_z_20_tszscore_60", "displayNameZh": "资金费率"}]
                }
            }
        },
        "agentCandidatePromotion": {
            "records": [
                {
                    "factorName": "agent__funding_z_20_tszscore_60__23572d7c1f",
                    "displayName": "资金费率",
                    "idea": {"nameHint": "funding_z_20_tszscore_60"},
                    "seenAt": "2026-05-22T15:18:34.986676+00:00",
                    "status": "duplicate_existing",
                }
            ]
        },
    }

    rows = _agent_candidate_rows(memory)
    assert len(rows) == 1
    assert rows[0]["validationStatusKey"] == "duplicate"
