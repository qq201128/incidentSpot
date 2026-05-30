from __future__ import annotations

import json
from typing import Any


class CachePayloadDecodeError(ValueError):
    def __init__(self, *, cache_name: str, identity: dict[str, Any], cause: json.JSONDecodeError) -> None:
        self.details = {
            "cacheName": cache_name,
            **identity,
            "error": str(cause),
            "exceptionType": type(cause).__name__,
        }
        label = " ".join(str(value) for value in identity.values())
        super().__init__(f"{cache_name} corrupt JSON for {label}: {cause}")


def decode_cache_payload(raw: str, *, cache_name: str, identity: dict[str, Any]) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CachePayloadDecodeError(cache_name=cache_name, identity=identity, cause=exc) from exc
