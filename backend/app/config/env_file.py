from __future__ import annotations

import os
from pathlib import Path

BACKEND_ROOT_PARENT_INDEX = 2
ENV_FILE_NAME = ".env"
COMMENT_PREFIX = "#"
KEY_VALUE_SEPARATOR = "="
EXPORT_PREFIX = "export "


def load_backend_env_file() -> None:
    env_path = Path(__file__).resolve().parents[BACKEND_ROOT_PARENT_INDEX] / ENV_FILE_NAME
    if not env_path.exists():
        return

    for line_number, raw_line in enumerate(env_path.read_text(encoding="utf-8").splitlines(), start=1):
        parsed = _parse_env_line(raw_line, line_number)
        if parsed is None:
            continue
        key, value = parsed
        os.environ.setdefault(key, value)


def _parse_env_line(raw_line: str, line_number: int) -> tuple[str, str] | None:
    line = raw_line.strip()
    if not line or line.startswith(COMMENT_PREFIX):
        return None
    if line.startswith(EXPORT_PREFIX):
        line = line[len(EXPORT_PREFIX) :].strip()
    if KEY_VALUE_SEPARATOR not in line:
        raise RuntimeError(f"invalid .env line {line_number}: missing '='")

    key, value = line.split(KEY_VALUE_SEPARATOR, maxsplit=1)
    key = key.strip()
    if not key:
        raise RuntimeError(f"invalid .env line {line_number}: empty key")
    return key, _clean_value(value)


def _clean_value(value: str) -> str:
    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {"'", '"'}:
        return stripped[1:-1]
    return stripped
