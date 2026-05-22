from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from app.services.factor_learning_llm_agent import (
    AGENT_RUNNING_STALE_SECONDS,
    _compact_memory,
    _user_prompt,
    is_llm_agent_run_stale,
)


def test_compact_agent_prompt_stays_small() -> None:
    memory = {
        "symbol": "BTCUSDT",
        "duration": "10m",
        "source": {"status": "ready", "rankingRefreshSource": "cache", "minedFrameFailureCount": 0},
        "factorMining": {
            "successPatterns": [{"pattern": "x" * 500, "members": ["a" * 80]}],
            "forbiddenRegions": [{"region": "y" * 500, "members": ["b" * 80]}],
        },
        "lossMemory": {"status": "ok", "patterns": []},
        "weights": {"factor_a": 0.8},
        "minedFactorLibrary": {
            "total": 1,
            "factors": [
                {
                    "factorName": "combo__" + ("nested__" * 40),
                    "factorDisplayName": "组合" * 80,
                    "formula": "f(" + "x," * 200 + ")",
                    "members": [{"name": "member__" + ("long__" * 20)}],
                    "metrics": {"winRate": 0.6, "profitFactor": 1.2, "ir": 0.3},
                }
            ],
        },
        "agentMinedFactorLibrary": {
            "total": 1,
            "factors": [
                {
                    "factorName": "agent__demo",
                    "factorDisplayName": "演示",
                    "formula": "TsZScore(close, 60)",
                    "metrics": {"winRate": 0.55, "profitFactor": 1.1},
                    "qualityPassed": True,
                }
            ],
        },
        "monitoring": {"status": "ok", "issues": [{"level": "warn", "message": "m" * 300}]},
    }

    prompt = _user_prompt(memory)

    assert len(prompt) < 30_000
    assert len(json.dumps(_compact_memory(memory), ensure_ascii=False)) < 12_000


def test_stale_running_agent_is_detected() -> None:
    stamp = (datetime.now(timezone.utc) - timedelta(seconds=AGENT_RUNNING_STALE_SECONDS + 5)).isoformat()
    agent = {"status": "running", "updatedAt": stamp}

    assert is_llm_agent_run_stale(agent) is True
