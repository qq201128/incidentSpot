from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import requests
from requests import RequestException

DEFAULT_SILICONFLOW_MODEL = "Pro/moonshotai/Kimi-K2.6"
DEFAULT_CHAT_COMPLETIONS_URL = "https://api.siliconflow.cn/v1/chat/completions"
DEFAULT_TIMEOUT_SECONDS = 180
TIMEOUT_ENV_NAME = "SILICONFLOW_TIMEOUT_SECONDS"


@dataclass(frozen=True)
class SiliconFlowConfig:
    api_key: str
    model: str = DEFAULT_SILICONFLOW_MODEL
    url: str = DEFAULT_CHAT_COMPLETIONS_URL
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS


class SiliconFlowChatClient:
    def __init__(self, config: SiliconFlowConfig | None = None) -> None:
        self.config = config or siliconflow_config_from_env()

    @property
    def model(self) -> str:
        return self.config.model

    def create_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        request_payload = {"model": self.config.model, **payload}
        try:
            response = requests.post(
                self.config.url,
                headers=_headers(self.config.api_key),
                json=request_payload,
                timeout=self.config.timeout_seconds,
            )
            response.raise_for_status()
        except RequestException as exc:
            raise RuntimeError(f"SiliconFlow chat completion failed: {exc}") from exc
        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError("SiliconFlow chat completion returned non-object JSON")
        return data


def siliconflow_config_from_env() -> SiliconFlowConfig:
    api_key = os.getenv("SILICONFLOW_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("missing SILICONFLOW_API_KEY in environment or .env")
    model = os.getenv("SILICONFLOW_MODEL", DEFAULT_SILICONFLOW_MODEL).strip()
    url = os.getenv("SILICONFLOW_CHAT_COMPLETIONS_URL", DEFAULT_CHAT_COMPLETIONS_URL).strip()
    return SiliconFlowConfig(
        api_key=api_key,
        model=model,
        url=url,
        timeout_seconds=_timeout_seconds_from_env(),
    )


def _timeout_seconds_from_env() -> int:
    raw_value = os.getenv(TIMEOUT_ENV_NAME, "").strip()
    if not raw_value:
        return DEFAULT_TIMEOUT_SECONDS
    try:
        timeout_seconds = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"{TIMEOUT_ENV_NAME} must be a positive integer") from exc
    if timeout_seconds <= 0:
        raise RuntimeError(f"{TIMEOUT_ENV_NAME} must be a positive integer")
    return timeout_seconds


def _headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
