from __future__ import annotations

import pytest

from app.services.wxpusher_app_client import (
    DEFAULT_WXPUSHER_SEND_URL,
    WXPUSHER_MARKDOWN_CONTENT_TYPE,
    WxPusherAppClient,
    wxpusher_app_config_from_env,
    wxpusher_app_configured,
)


def test_wxpusher_config_requires_app_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WXPUSHER_APP_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="WXPUSHER_APP_TOKEN"):
        wxpusher_app_config_from_env()


def test_wxpusher_config_reads_app_recipients(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WXPUSHER_APP_TOKEN", "AT_test")
    monkeypatch.setenv("WXPUSHER_UIDS", "UID_a, UID_b")
    monkeypatch.setenv("WXPUSHER_TOPIC_IDS", "1,2")

    config = wxpusher_app_config_from_env()

    assert wxpusher_app_configured() is True
    assert config.app_token == "AT_test"
    assert config.uids == ("UID_a", "UID_b")
    assert config.topic_ids == (1, 2)
    assert config.send_url == DEFAULT_WXPUSHER_SEND_URL
    assert config.verify_pay_type == 0


def test_wxpusher_send_markdown_posts_app_message(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"code": 1000, "msg": "ok"}

    def fake_post(url, *, headers, json, timeout, **_kwargs):
        calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return Response()

    monkeypatch.setenv("WXPUSHER_APP_TOKEN", "AT_test")
    monkeypatch.setenv("WXPUSHER_UIDS", "UID_a")
    monkeypatch.setattr("app.services.wxpusher_app_client.requests.post", fake_post)

    result = WxPusherAppClient().send_markdown(summary="标题", content="# 内容")

    assert result["code"] == 1000
    assert calls[0]["url"] == DEFAULT_WXPUSHER_SEND_URL
    assert calls[0]["json"]["appToken"] == "AT_test"
    assert calls[0]["json"]["uids"] == ["UID_a"]
    assert calls[0]["json"]["contentType"] == WXPUSHER_MARKDOWN_CONTENT_TYPE
    assert calls[0]["json"]["verifyPayType"] == 0
    assert calls[0]["json"]["summary"] == "标题"
