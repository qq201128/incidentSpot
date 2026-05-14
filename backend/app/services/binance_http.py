from __future__ import annotations

import time
from typing import Any

import requests
from requests.exceptions import RequestException

FAPI_BASE_URL = "https://fapi.binance.com"
DEFAULT_MAX_ATTEMPTS = 6
DEFAULT_TIMEOUT = (10, 40)
SHORT_TIMEOUT = (10, 20)
MAX_RETRY_SLEEP_SECONDS = 20
RETRY_BACKOFF_BASE = 2


def retry_get(
    url: str,
    params: dict[str, Any],
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    timeout: tuple[int, int] = DEFAULT_TIMEOUT,
) -> dict | list:
    last_error: Exception | None = None
    for attempt in range(max_attempts):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except RequestException as exc:
            last_error = exc
            if attempt == max_attempts - 1:
                break
            time.sleep(min(RETRY_BACKOFF_BASE ** attempt, MAX_RETRY_SLEEP_SECONDS))
    raise last_error  # type: ignore[misc]
