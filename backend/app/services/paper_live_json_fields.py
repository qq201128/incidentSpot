from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParsedJsonField:
    value: Any
    error: dict[str, str] | None


def parse_json_field(field: str, value: Any) -> ParsedJsonField:
    if not isinstance(value, str):
        return ParsedJsonField(value, None)
    try:
        return ParsedJsonField(json.loads(value), None)
    except json.JSONDecodeError as exc:
        return ParsedJsonField(
            value,
            {
                "field": field,
                "error": str(exc),
                "exceptionType": type(exc).__name__,
            },
        )


def parse_details_json(value: Any) -> ParsedJsonField:
    if value in (None, ""):
        return ParsedJsonField({}, None)
    return parse_json_field("details", value)
