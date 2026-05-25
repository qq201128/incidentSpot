from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from app.services import factor_learning_llm_agent as llm_agent
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


def test_agent_prompt_does_not_grow_with_large_libraries(monkeypatch) -> None:
    monkeypatch.setattr(
        llm_agent,
        "limited_do_not_suggest_formulas",
        lambda *_args: [f"TsZScore(factor_{index}, 60)" for index in range(500)],
    )
    memory = _large_prompt_memory()

    prompt = _user_prompt(memory)
    compact = _compact_memory(memory)

    assert len(prompt) < 30_000
    assert len(json.dumps(compact, ensure_ascii=False)) < 12_000
    assert len(compact["doNotSuggestFactorNames"]) == llm_agent.AGENT_PROMPT_NAME_BLOCK_LIMIT
    assert compact["doNotSuggestFactorNameTotal"] == 500
    assert len(compact["retrieval"]["miningExcludedFactorNames"]) == llm_agent.AGENT_PROMPT_RETRIEVAL_NAME_LIMIT
    assert compact["retrieval"]["miningExcludedFactorNameTotal"] == 4800
    assert len(compact["filters"]["blockedFeatures"]) == llm_agent.AGENT_PROMPT_RETRIEVAL_NAME_LIMIT
    assert compact["filters"]["blockedFeaturesTotal"] == 800


def test_stale_running_agent_is_detected() -> None:
    stamp = (datetime.now(timezone.utc) - timedelta(seconds=AGENT_RUNNING_STALE_SECONDS + 5)).isoformat()
    agent = {"status": "running", "updatedAt": stamp}

    assert is_llm_agent_run_stale(agent) is True


def _large_prompt_memory() -> dict:
    return {
        "symbol": "BTCUSDT",
        "duration": "10m",
        "source": {"status": "ready", "rankingRefreshSource": "cache", "minedFrameFailureCount": 0},
        "factorMining": {
            "successPatterns": [_pattern("pattern", index) for index in range(200)],
            "forbiddenRegions": [_pattern("region", index) for index in range(200)],
        },
        "lossMemory": {
            "status": "ok",
            "sampleCount": 1000,
            "lossCount": 400,
            "patterns": [_loss_pattern(index) for index in range(800)],
        },
        "filters": {"blockedFeatures": [f"blocked_feature_{index}" for index in range(800)]},
        "weights": {f"factor_{index}": 1.0 / (index + 1) for index in range(800)},
        "minedFactorLibrary": {"total": 500, "factors": [_library_row("combo", index) for index in range(500)]},
        "agentMinedFactorLibrary": {
            "total": 500,
            "factors": [_library_row("agent", index) for index in range(500)],
        },
        "monitoring": {"status": "ok", "issues": [{"message": "m" * 300} for _ in range(100)]},
    }


def _pattern(key: str, index: int) -> dict:
    return {
        key: f"{key}_{index}_" + ("x" * 500),
        "support": index,
        "members": [f"member_{index}_{member}_" + ("y" * 120) for member in range(20)],
    }


def _loss_pattern(index: int) -> dict:
    return {"feature": f"loss_feature_{index}", "pattern": "loss_" + ("z" * 500), "support": index}


def _library_row(prefix: str, index: int) -> dict:
    return {
        "factorName": f"{prefix}__factor_{index}__" + ("n" * 120),
        "factorDisplayName": "因子" * 120,
        "formula": "TsZScore(" + ("x," * 200) + "60)",
        "members": [{"name": f"member_{index}_{member}"} for member in range(20)],
        "metrics": {"winRate": 0.55, "profitFactor": 1.2, "ir": 0.3},
        "qualityPassed": index % 2 == 0,
    }
