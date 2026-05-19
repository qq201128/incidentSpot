from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import requests
from requests import RequestException

DEFAULT_WXPUSHER_SEND_URL = "https://wxpusher.zjiecode.com/api/send/message"
DEFAULT_WXPUSHER_TIMEOUT_SECONDS = 15
WXPUSHER_MARKDOWN_CONTENT_TYPE = 3


@dataclass(frozen=True)
class WxPusherAppConfig:
    app_token: str
    uids: tuple[str, ...]
    topic_ids: tuple[int, ...]
    send_url: str = DEFAULT_WXPUSHER_SEND_URL
    timeout_seconds: int = DEFAULT_WXPUSHER_TIMEOUT_SECONDS
    verify_pay_type: int = 0


class WxPusherAppClient:
    def __init__(self, config: WxPusherAppConfig | None = None) -> None:
        self.config = config or wxpusher_app_config_from_env()

    def send_markdown(self, *, summary: str, content: str) -> dict[str, Any]:
        payload = {
            "appToken": self.config.app_token,
            "content": content,
            "summary": summary[:100],
            "contentType": WXPUSHER_MARKDOWN_CONTENT_TYPE,
            "uids": list(self.config.uids),
            "topicIds": list(self.config.topic_ids),
            "verifyPayType": self.config.verify_pay_type,
        }
        try:
            response = requests.post(
                self.config.send_url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=self.config.timeout_seconds,
            )
            response.raise_for_status()
        except RequestException as exc:
            raise RuntimeError(f"WxPusher app message failed: {exc}") from exc
        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError("WxPusher app message returned non-object JSON")
        if int(data.get("code") or 0) != 1000:
            raise RuntimeError(f"WxPusher app message rejected: {data}")
        return data


def wxpusher_app_config_from_env() -> WxPusherAppConfig:
    app_token = os.getenv("WXPUSHER_APP_TOKEN", "").strip()
    if not app_token:
        raise RuntimeError("missing WXPUSHER_APP_TOKEN in environment or .env")
    uids = _csv_strings(os.getenv("WXPUSHER_UIDS", ""))
    topic_ids = _csv_ints(os.getenv("WXPUSHER_TOPIC_IDS", ""))
    if not uids and not topic_ids:
        raise RuntimeError("missing WXPUSHER_UIDS or WXPUSHER_TOPIC_IDS in environment or .env")
    return WxPusherAppConfig(
        app_token=app_token,
        uids=uids,
        topic_ids=topic_ids,
        send_url=os.getenv("WXPUSHER_SEND_URL", DEFAULT_WXPUSHER_SEND_URL).strip(),
        timeout_seconds=_timeout_seconds_from_env(),
        verify_pay_type=_verify_pay_type_from_env(),
    )


def wxpusher_app_configured() -> bool:
    return bool(os.getenv("WXPUSHER_APP_TOKEN", "").strip()) and bool(
        _csv_strings(os.getenv("WXPUSHER_UIDS", ""))
        or _csv_strings(os.getenv("WXPUSHER_TOPIC_IDS", ""))
    )


def _timeout_seconds_from_env() -> int:
    raw_value = os.getenv("WXPUSHER_TIMEOUT_SECONDS", "").strip()
    if not raw_value:
        return DEFAULT_WXPUSHER_TIMEOUT_SECONDS
    value = int(raw_value)
    if value <= 0:
        raise RuntimeError("WXPUSHER_TIMEOUT_SECONDS must be a positive integer")
    return value


def _csv_strings(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _csv_ints(raw: str) -> tuple[int, ...]:
    return tuple(int(part) for part in _csv_strings(raw))


def _verify_pay_type_from_env() -> int:
    raw = os.getenv("WXPUSHER_VERIFY_PAY_TYPE", "").strip()
    if not raw:
        return 0
    value = int(raw)
    if value not in {0, 1, 2}:
        raise RuntimeError("WXPUSHER_VERIFY_PAY_TYPE must be 0, 1, or 2")
    return value
