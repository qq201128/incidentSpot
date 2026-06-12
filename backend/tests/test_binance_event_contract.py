from __future__ import annotations

import pytest
import requests

from app.services import binance_event_contract as service


def test_load_auth_accepts_full_cookie(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BINANCE_EVENT_CSRF_TOKEN", "csrf-token")
    monkeypatch.setenv("BINANCE_EVENT_COOKIE", "p20t=session-token; bnc-uuid=uuid-1")
    monkeypatch.delenv("BINANCE_EVENT_P20T", raising=False)

    headers, cookies = service._load_auth()

    assert headers["csrftoken"] == "csrf-token"
    assert cookies["p20t"] == "session-token"
    assert cookies["bnc-uuid"] == "uuid-1"


def test_load_auth_keeps_p20t_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BINANCE_EVENT_CSRF_TOKEN", "csrf-token")
    monkeypatch.delenv("BINANCE_EVENT_COOKIE", raising=False)
    monkeypatch.setenv("BINANCE_EVENT_P20T", "session-token")

    _headers, cookies = service._load_auth()

    assert cookies == {"p20t": "session-token"}


def test_load_auth_adds_p20t_when_full_cookie_omits_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BINANCE_EVENT_CSRF_TOKEN", "csrf-token")
    monkeypatch.setenv("BINANCE_EVENT_COOKIE", "bnc-uuid=uuid-1")
    monkeypatch.setenv("BINANCE_EVENT_P20T", "session-token")

    _headers, cookies = service._load_auth()

    assert cookies["bnc-uuid"] == "uuid-1"
    assert cookies["p20t"] == "session-token"


def test_request_error_message_includes_response_body() -> None:
    response = requests.Response()
    response.status_code = 401
    response._content = b'{"code":"401","message":"unauthorized"}'
    error = requests.HTTPError("401 Client Error: Unauthorized", response=response)

    message = service._request_error_message(error)

    assert "401 Client Error: Unauthorized" in message
    assert "response_body=" in message
    assert "unauthorized" in message
