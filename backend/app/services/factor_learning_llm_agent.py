from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from typing import Any, Protocol

from app.services.factor_operator_library import AGENT_FORMULA_RULES, factor_operator_prompt_payload
from app.services.factor_learning_retrieval import build_factor_learning_retrieval
from app.services.siliconflow_chat_client import SiliconFlowChatClient

AGENT_NAME = "siliconflow_kimi_factor_agent_v1"
AGENT_PROVIDER = "siliconflow"
AGENT_TEMPERATURE = 0.2
# Chinese-heavy JSON can exceed a few thousand characters; 2400 tokens often truncates mid-object.
AGENT_MAX_TOKENS_DEFAULT = 8192


def _agent_max_tokens() -> int:
    raw = os.getenv("FACTOR_LEARNING_AGENT_MAX_TOKENS", "").strip()
    if raw:
        try:
            value = int(raw)
        except ValueError:
            return AGENT_MAX_TOKENS_DEFAULT
        if 256 <= value <= 32768:
            return value
    return AGENT_MAX_TOKENS_DEFAULT


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
        "max_tokens": _agent_max_tokens(),
        "response_format": {"type": "json_object"},
    }


def _system_prompt() -> str:
    return (
        "你是量化因子挖掘 LLM Agent。你必须只输出 JSON 对象。"
        "不要编造已验证结果，不要声称真实回测通过；只能基于输入记忆提出候选研究方向。"
        "不要再次提出 doNotSuggestFactorNames 中已经入库的单因子。"
        "必须优先使用 memory.retrieval 中整理好的成功模式、禁区、亏损模式和权重。"
        "重点学习 FactorMiner 思路：成功模式、禁区、亏损模式、多重过滤、自动权重。"
        "候选必须能落到现有算子库和现有特征列，不可物化的想法直接拒绝。"
        "formulaHint 必须遵守 formula_constraints；尤其禁止生成 PctChange(x, 1)。"
        "每个候选都要说明回流到因子库的验证路径、需要检查的列与过滤条件。"
        "所有给交易员看的因子名称、理由、风控建议必须使用中文。"
    )


def _user_prompt(memory: dict[str, Any]) -> str:
    return json.dumps(
        {
            "task": "review_factor_learning_memory_and_plan_next_factor_mining",
            "required_schema": _required_schema(),
            "formula_constraints": list(AGENT_FORMULA_RULES),
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
        "retrieval": build_factor_learning_retrieval(memory),
        "factorMining": memory.get("factorMining") or {},
        "lossMemory": memory.get("lossMemory") or {},
        "filters": memory.get("filters") or {},
        "weights": _top_weights(memory.get("weights") or {}),
        "minedFactorLibrary": memory.get("minedFactorLibrary") or {},
        "agentMinedFactorLibrary": memory.get("agentMinedFactorLibrary") or {},
        "doNotSuggestFactorNames": _factor_names(memory.get("agentMinedFactorLibrary") or {}),
        "monitoring": memory.get("monitoring") or {},
    }


def _top_weights(weights: dict[str, Any]) -> dict[str, Any]:
    pairs = sorted(weights.items(), key=lambda item: float(item[1]), reverse=True)
    return dict(pairs[:20])


def _factor_names(library: dict[str, Any]) -> list[str]:
    return [str(row.get("factorName")) for row in library.get("factors") or [] if row.get("factorName")]


def _review_from_completion(completion: dict[str, Any]) -> dict[str, Any]:
    content, finish_reason = _assistant_message(completion)
    try:
        parsed = _parse_factor_agent_json(content)
    except (json.JSONDecodeError, ValueError) as exc:
        tail = content[-500:] if len(content) > 500 else content
        fr = f" finish_reason={finish_reason!r}" if finish_reason else ""
        truncated = finish_reason == "length" or not content.rstrip().endswith("}")
        hint = " (completion may be truncated: raise FACTOR_LEARNING_AGENT_MAX_TOKENS)" if truncated else ""
        raise RuntimeError(
            f"SiliconFlow factor agent returned invalid JSON{fr}{hint}; tail={tail!r}"
        ) from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("SiliconFlow factor agent JSON must be an object")
    return parsed


def _strip_markdown_json_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    # ```json\n{...}\n``` or ```\n{...}\n```
    stripped = re.sub(r"^```[a-zA-Z0-9_-]*\s*\n?", "", stripped, count=1)
    stripped = re.sub(r"\n?```\s*$", "", stripped, count=1)
    return stripped.strip()


def _extract_outer_json_object(text: str) -> str:
    """Return substring from first '{' through matching '}' (handles leading prose)."""
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object start '{' in assistant content")
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise ValueError("unterminated JSON object in assistant content")


def _parse_factor_agent_json(content: str) -> Any:
    trimmed = _strip_markdown_json_fence(content)
    try:
        return json.loads(trimmed)
    except json.JSONDecodeError:
        candidate = _extract_outer_json_object(trimmed)
        return json.loads(candidate)


def _assistant_message(completion: dict[str, Any]) -> tuple[str, str | None]:
    choices = completion.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("SiliconFlow response missing choices")
    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("SiliconFlow response missing assistant content")
    finish_reason = first.get("finish_reason") if isinstance(first.get("finish_reason"), str) else None
    return content, finish_reason
