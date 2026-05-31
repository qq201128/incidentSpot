from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

DEFAULT_LLM_PROVIDER = "siliconflow"
DEFAULT_SILICONFLOW_MODEL = "deepseek-ai/DeepSeek-V3.2"
DEFAULT_CHAT_COMPLETIONS_URL = "https://api.siliconflow.cn/v1/chat/completions"
SILICONFLOW_API_KEY_ENV = "SILICONFLOW_API_KEY"
SILICONFLOW_MODEL_ENV = "SILICONFLOW_MODEL"
SILICONFLOW_URL_ENV = "SILICONFLOW_CHAT_COMPLETIONS_URL"


@dataclass(frozen=True)
class LlmModelProfile:
    provider: str
    model: str
    capabilities: tuple[str, ...]
    default: bool = False


@dataclass(frozen=True)
class LlmProviderProfile:
    provider: str
    label: str
    api_key_env: str
    model_env: str
    url_env: str
    default_model: str
    default_url: str
    capabilities: tuple[str, ...]


SILICONFLOW_CAPABILITIES = (
    "chat_completions",
    "json_object_response",
    "factor_mining_review",
    "chinese_reasoning",
)

LLM_PROVIDER_REGISTRY = {
    DEFAULT_LLM_PROVIDER: LlmProviderProfile(
        provider=DEFAULT_LLM_PROVIDER,
        label="SiliconFlow",
        api_key_env=SILICONFLOW_API_KEY_ENV,
        model_env=SILICONFLOW_MODEL_ENV,
        url_env=SILICONFLOW_URL_ENV,
        default_model=DEFAULT_SILICONFLOW_MODEL,
        default_url=DEFAULT_CHAT_COMPLETIONS_URL,
        capabilities=SILICONFLOW_CAPABILITIES,
    )
}

LLM_MODEL_REGISTRY = {
    (DEFAULT_LLM_PROVIDER, DEFAULT_SILICONFLOW_MODEL): LlmModelProfile(
        provider=DEFAULT_LLM_PROVIDER,
        model=DEFAULT_SILICONFLOW_MODEL,
        capabilities=SILICONFLOW_CAPABILITIES,
        default=True,
    )
}


def llm_provider_profile(provider: str = DEFAULT_LLM_PROVIDER) -> LlmProviderProfile:
    key = provider.strip().lower()
    profile = LLM_PROVIDER_REGISTRY.get(key)
    if profile is None:
        raise ValueError(f"unsupported LLM provider: {provider}")
    return profile


def resolved_llm_model(provider: str = DEFAULT_LLM_PROVIDER) -> str:
    profile = llm_provider_profile(provider)
    return os.getenv(profile.model_env, profile.default_model).strip() or profile.default_model


def resolved_llm_url(provider: str = DEFAULT_LLM_PROVIDER) -> str:
    profile = llm_provider_profile(provider)
    return os.getenv(profile.url_env, profile.default_url).strip() or profile.default_url


def llm_model_metadata(provider: str = DEFAULT_LLM_PROVIDER, model: str | None = None) -> dict[str, Any]:
    profile = llm_provider_profile(provider)
    selected = (model or resolved_llm_model(profile.provider)).strip()
    registered = LLM_MODEL_REGISTRY.get((profile.provider, selected))
    capabilities = registered.capabilities if registered else profile.capabilities
    return {
        "provider": profile.provider,
        "providerLabel": profile.label,
        "model": selected,
        "registeredModel": registered is not None,
        "defaultModel": bool(registered.default) if registered else selected == profile.default_model,
        "capabilities": list(capabilities),
    }


def llm_provider_availability(provider: str = DEFAULT_LLM_PROVIDER, model: str | None = None) -> dict[str, Any]:
    profile = llm_provider_profile(provider)
    api_key = os.getenv(profile.api_key_env, "").strip()
    metadata = llm_model_metadata(profile.provider, model)
    missing_env = [] if api_key else [profile.api_key_env]
    return {
        **metadata,
        "status": "available" if not missing_env else "unavailable",
        "probe": "config_only",
        "missingEnv": missing_env,
        "urlConfigured": bool(resolved_llm_url(profile.provider)),
    }
