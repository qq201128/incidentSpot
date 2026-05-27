from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Protocol

from app.services.agent_formula_dedup import filter_agent_review_duplicates, limited_do_not_suggest_formulas
from app.services.factor_operator_library import AGENT_FORMULA_RULES, factor_operator_prompt_payload
from app.services.factor_learning_common import utc_now
from app.services.factor_learning_llm_memory import compact_memory
from app.services.siliconflow_chat_client import SiliconFlowChatClient, siliconflow_config_from_env

AGENT_NAME = "siliconflow_factor_agent_v1"
AGENT_PROVIDER = "siliconflow"
AGENT_TEMPERATURE = 0.2
# Chinese-heavy JSON can exceed a few thousand characters; 2400 tokens often truncates mid-object.
AGENT_MAX_TOKENS_DEFAULT = 8192
AGENT_TIMEOUT_SECONDS_DEFAULT = 300
AGENT_TIMEOUT_ENV = "FACTOR_LEARNING_SILICONFLOW_TIMEOUT_SECONDS"
AGENT_PROMPT_FORMULA_BLOCK_LIMIT = 48
AGENT_PROMPT_NAME_BLOCK_LIMIT = 16
AGENT_PROMPT_LIBRARY_ROW_LIMIT = 4
AGENT_PROMPT_COMBO_ROW_LIMIT = 3
AGENT_PROMPT_RETRIEVAL_NAME_LIMIT = 16
AGENT_PROMPT_RETRIEVAL_ROW_LIMIT = 8
AGENT_PROMPT_PATTERN_ROW_LIMIT = 3
AGENT_PROMPT_PATTERN_MEMBER_LIMIT = 2
AGENT_PROMPT_TEXT_LIMIT = 96
AGENT_PROMPT_SHORT_TEXT_LIMIT = 72
AGENT_RUNNING_STALE_SECONDS = 600

def _factor_agent_chat_client() -> SiliconFlowChatClient:
    base = siliconflow_config_from_env()
    timeout_seconds = _agent_timeout_seconds(base.timeout_seconds)
    if timeout_seconds == base.timeout_seconds:
        return SiliconFlowChatClient(base)
    return SiliconFlowChatClient(replace(base, timeout_seconds=timeout_seconds))

def _agent_timeout_seconds(fallback: int) -> int:
    raw = os.getenv(AGENT_TIMEOUT_ENV, "").strip()
    if raw:
        try:
            value = int(raw)
        except ValueError:
            return max(fallback, AGENT_TIMEOUT_SECONDS_DEFAULT)
        if value > 0:
            return value
    return max(fallback, AGENT_TIMEOUT_SECONDS_DEFAULT)

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
    active_client = client or _factor_agent_chat_client()
    completion = active_client.create_chat_completion(_agent_payload(memory))
    review = filter_agent_review_duplicates(
        _review_from_completion(completion),
        str(memory["symbol"]),
        str(memory["duration"]),
    )
    updated = deepcopy(memory)
    updated["llmAgent"] = {
        "agent": AGENT_NAME,
        "provider": AGENT_PROVIDER,
        "status": "completed",
        "model": active_client.model,
        "reviewedAt": utc_now(),
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
        "不要再次提出 doNotSuggestFactorNames 中已经入库的单因子，"
        "也不要再次提出 doNotSuggestFormulas 中已经出现过的 formulaHint（含未达标、已存在、物化失败的历史公式）。"
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
    symbol = str(memory.get("symbol") or "")
    duration = str(memory.get("duration") or "")
    blocklist = limited_do_not_suggest_formulas(symbol, duration)
    return compact_memory(memory, blocklist)

def is_llm_agent_run_stale(agent: dict[str, Any], *, now: datetime | None = None) -> bool:
    if str(agent.get("status") or "") not in {"pending", "running"}:
        return False
    stamp = str(agent.get("agentStartedAt") or agent.get("updatedAt") or "")
    if not stamp:
        return True
    try:
        started = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return True
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    current = now or datetime.now(timezone.utc)
    return (current - started).total_seconds() >= AGENT_RUNNING_STALE_SECONDS

def stale_llm_agent_error(agent: dict[str, Any]) -> str:
    stamp = str(agent.get("agentStartedAt") or agent.get("updatedAt") or "")
    return (
        "Agent 联网挖掘超时或中断（状态停留在 pending/running）。"
        f" 开始于 {stamp or '未知'}，超过 {AGENT_RUNNING_STALE_SECONDS} 秒未写回 review。"
        " 请重新点击联网挖掘；若反复失败，请检查 SILICONFLOW_API_KEY / 网络，"
        " 或在 backend/.env 设置 FACTOR_LEARNING_SILICONFLOW_TIMEOUT_SECONDS=420。"
    )

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
