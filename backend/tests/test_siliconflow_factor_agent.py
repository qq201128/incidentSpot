from __future__ import annotations

import json

import pytest

from app.services.factor_learning_llm_agent import (
    _parse_factor_agent_json,
    attach_llm_agent_review,
)
from app.services.factor_operator_library import factor_operator_payload
from app.services.siliconflow_chat_client import (
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_SILICONFLOW_MODEL,
    SiliconFlowChatClient,
    siliconflow_config_from_env,
)


def test_siliconflow_config_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="SILICONFLOW_API_KEY"):
        siliconflow_config_from_env()


def test_siliconflow_config_reads_timeout_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SILICONFLOW_API_KEY", "sk-test")
    monkeypatch.setenv("SILICONFLOW_TIMEOUT_SECONDS", "240")

    config = siliconflow_config_from_env()

    assert config.timeout_seconds == 240


def test_siliconflow_config_rejects_invalid_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SILICONFLOW_API_KEY", "sk-test")
    monkeypatch.setenv("SILICONFLOW_TIMEOUT_SECONDS", "0")

    with pytest.raises(RuntimeError, match="SILICONFLOW_TIMEOUT_SECONDS"):
        siliconflow_config_from_env()


def test_siliconflow_client_sends_bearer_request(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"choices": [{"message": {"content": "{}"}}]}

    def fake_post(url: str, **kwargs) -> Response:
        calls.append({"url": url, **kwargs})
        return Response()

    monkeypatch.setenv("SILICONFLOW_API_KEY", "sk-test")
    monkeypatch.delenv("SILICONFLOW_TIMEOUT_SECONDS", raising=False)
    monkeypatch.setattr("app.services.siliconflow_chat_client.requests.post", fake_post)

    client = SiliconFlowChatClient()
    client.create_chat_completion({"messages": [{"role": "user", "content": "hi"}]})

    assert calls[0]["json"]["model"] == DEFAULT_SILICONFLOW_MODEL
    assert calls[0]["headers"]["Authorization"] == "Bearer sk-test"
    assert calls[0]["timeout"] == DEFAULT_TIMEOUT_SECONDS


def test_parse_factor_agent_json_strips_markdown_fence() -> None:
    raw = '```json\n{"notes": ["x"]}\n```'
    assert _parse_factor_agent_json(raw) == {"notes": ["x"]}


def test_parse_factor_agent_json_skips_leading_prose() -> None:
    raw = 'Here you go:\n{"notes": ["y"]}\ntrailing'
    assert _parse_factor_agent_json(raw) == {"notes": ["y"]}


def test_factor_agent_attaches_json_review(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FACTOR_LEARNING_AGENT_MAX_TOKENS", raising=False)
    client = FakeClient()
    memory = {"symbol": "BTCUSDT", "duration": "10m", "weights": {"factor_a": 0.7}}

    updated = attach_llm_agent_review(memory, client=client)

    assert updated["llmAgent"]["model"] == DEFAULT_SILICONFLOW_MODEL
    assert updated["llmAgent"]["review"]["notes"] == ["keep_loss_memory_explicit"]
    assert client.payload["response_format"] == {"type": "json_object"}
    assert client.payload["max_tokens"] == 8192
    assert "factor_a" in client.payload["messages"][1]["content"]
    assert "operator_library" in client.payload["messages"][1]["content"]
    assert "formula_constraints" in client.payload["messages"][1]["content"]
    assert "PctChange(x, 1) is invalid" in client.payload["messages"][1]["content"]
    assert "\"retrieval\"" in client.payload["messages"][1]["content"]
    assert "llmAgent" not in memory


def test_factor_operator_library_exposes_many_categories() -> None:
    payload = factor_operator_payload()
    categories = {item["key"] for item in payload["categories"]}

    assert payload["total"] >= 60
    assert {"time_series", "regression", "logical", "microstructure"} <= categories


class FakeClient:
    model = DEFAULT_SILICONFLOW_MODEL

    def __init__(self) -> None:
        self.payload = {}

    def create_chat_completion(self, payload: dict) -> dict:
        self.payload = payload
        content = json.dumps({"notes": ["keep_loss_memory_explicit"]})
        return {
            "id": "cmpl-test",
            "choices": [{"message": {"content": content}}],
            "usage": {"total_tokens": 12},
        }
