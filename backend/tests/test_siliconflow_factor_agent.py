from __future__ import annotations

import json

import pytest

from app.services.factor_learning_llm_agent import attach_llm_agent_review
from app.services.factor_operator_library import factor_operator_payload
from app.services.siliconflow_chat_client import (
    DEFAULT_SILICONFLOW_MODEL,
    SiliconFlowChatClient,
    siliconflow_config_from_env,
)


def test_siliconflow_config_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="SILICONFLOW_API_KEY"):
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
    monkeypatch.setattr("app.services.siliconflow_chat_client.requests.post", fake_post)

    client = SiliconFlowChatClient()
    client.create_chat_completion({"messages": [{"role": "user", "content": "hi"}]})

    assert calls[0]["json"]["model"] == DEFAULT_SILICONFLOW_MODEL
    assert calls[0]["headers"]["Authorization"] == "Bearer sk-test"


def test_factor_agent_attaches_json_review() -> None:
    client = FakeClient()
    memory = {"symbol": "BTCUSDT", "duration": "10m", "weights": {"factor_a": 0.7}}

    updated = attach_llm_agent_review(memory, client=client)

    assert updated["llmAgent"]["model"] == DEFAULT_SILICONFLOW_MODEL
    assert updated["llmAgent"]["review"]["notes"] == ["keep_loss_memory_explicit"]
    assert client.payload["response_format"] == {"type": "json_object"}
    assert "factor_a" in client.payload["messages"][1]["content"]
    assert "operator_library" in client.payload["messages"][1]["content"]
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
