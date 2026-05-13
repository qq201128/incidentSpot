from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Protocol

from app.services.factor_operator_library import factor_operator_prompt_payload
from app.services.siliconflow_chat_client import SiliconFlowChatClient

AGENT_NAME = "siliconflow_kimi_factor_agent_v1"
AGENT_PROVIDER = "siliconflow"
AGENT_TEMPERATURE = 0.2
AGENT_MAX_TOKENS = 2400


class ChatCompletionClient(Protocol):
    @property
    def model(self) -> str: ...

    def create_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]: ...


def attach_llm_agent_review(
    memory: dict[str, Any],
    client: ChatCompletionClient | None = None,
) -> dict[str, Any]:
    active_client = client or SiliconFlowChatClient()
    completion = active_client.create_chat_completion(_agent_payload(memory))
    review = _review_from_completion(completion)
    updated = deepcopy(memory)
    updated["llmAgent"] = {
        "agent": AGENT_NAME,
        "provider": AGENT_PROVIDER,
        "status": "completed",
        "model": active_client.model,
        "completionId": completion.get("id"),
        "usage": completion.get("usage") or {},
        "review": review,
    }
    return updated


def _agent_payload(memory: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": _user_prompt(memory)},
        ],
        "temperature": AGENT_TEMPERATURE,
        "max_tokens": AGENT_MAX_TOKENS,
        "response_format": {"type": "json_object"},
    }


def _system_prompt() -> str:
    return (
        "你是量化因子挖掘 LLM Agent。你必须只输出 JSON 对象。"
        "不要编造已验证结果，不要声称真实回测通过；只能基于输入记忆提出候选研究方向。"
        "重点学习 FactorMiner 思路：成功模式、禁区、亏损模式、多重过滤、自动权重。"
        "所有给交易员看的因子名称、理由、风控建议必须使用中文。"
    )


def _user_prompt(memory: dict[str, Any]) -> str:
    return json.dumps(
        {
            "task": "review_factor_learning_memory_and_plan_next_factor_mining",
            "required_schema": _required_schema(),
            "operator_library": factor_operator_prompt_payload(),
            "memory": _compact_memory(memory),
        },
        ensure_ascii=False,
    )


def _required_schema() -> dict[str, Any]:
    return {
        "factorMiningPlan": {
            "successfulPatternsToExpand": ["string"],
            "forbiddenRegionsToAvoid": ["string"],
            "operatorFamiliesUsed": ["string"],
            "candidateFactorIdeas": [
                {
                    "nameHint": "string",
                    "displayNameZh": "string",
                    "formulaHint": "string",
                    "operatorTrace": ["string"],
                    "rationale": "string",
                    "rationaleZh": "string",
                    "requiredColumns": ["string"],
                    "validationChecks": ["string"],
                }
            ],
        },
        "lossPatternReview": {
            "featuresToRemember": ["string"],
            "hardFilterAdvice": ["string"],
            "downweightAdvice": ["string"],
        },
        "multiFilterPolicy": {
            "requiredConfirmations": "number",
            "hardBlocks": ["string"],
            "softDownweights": ["string"],
        },
        "notes": ["string"],
    }


def _compact_memory(memory: dict[str, Any]) -> dict[str, Any]:
    return {
        "symbol": memory.get("symbol"),
        "duration": memory.get("duration"),
        "source": memory.get("source") or {},
        "factorMining": memory.get("factorMining") or {},
        "lossMemory": memory.get("lossMemory") or {},
        "filters": memory.get("filters") or {},
        "weights": _top_weights(memory.get("weights") or {}),
        "minedFactorLibrary": memory.get("minedFactorLibrary") or {},
        "monitoring": memory.get("monitoring") or {},
    }


def _top_weights(weights: dict[str, Any]) -> dict[str, Any]:
    pairs = sorted(weights.items(), key=lambda item: float(item[1]), reverse=True)
    return dict(pairs[:20])


def _review_from_completion(completion: dict[str, Any]) -> dict[str, Any]:
    content = _assistant_content(completion)
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError("SiliconFlow factor agent returned invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("SiliconFlow factor agent JSON must be an object")
    return parsed


def _assistant_content(completion: dict[str, Any]) -> str:
    choices = completion.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("SiliconFlow response missing choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("SiliconFlow response missing assistant content")
    return content
