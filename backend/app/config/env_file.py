from __future__ import annotations

import os
from pathlib import Path

BACKEND_ROOT_PARENT_INDEX = 2
ENV_FILE_NAME = ".env"
ENV_LOCAL_FILE_NAME = ".env.local"
COMMENT_PREFIX = "#"
KEY_VALUE_SEPARATOR = "="
EXPORT_PREFIX = "export "


def load_backend_env_file() -> None:
    """
    加载环境配置文件

    优先级（从低到高）：
    1. .env - 通用配置
    2. .env.local - 本地开发配置（会覆盖.env中的同名变量）
    """
    backend_root = Path(__file__).resolve().parents[BACKEND_ROOT_PARENT_INDEX]

    # 先加载 .env
    env_path = backend_root / ENV_FILE_NAME
    if env_path.exists():
        _load_single_env_file(env_path)

    # 再加载 .env.local（会覆盖.env中的配置）
    env_local_path = backend_root / ENV_LOCAL_FILE_NAME
    if env_local_path.exists():
        _load_single_env_file(env_local_path)


def _load_single_env_file(env_path: Path) -> None:
    """加载单个环境文件"""
    for line_number, raw_line in enumerate(env_path.read_text(encoding="utf-8").splitlines(), start=1):
        parsed = _parse_env_line(raw_line, line_number)
        if parsed is None:
            continue
        key, value = parsed
        # 改为直接覆盖（而非setdefault），让.env.local能覆盖.env
        os.environ[key] = value


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
