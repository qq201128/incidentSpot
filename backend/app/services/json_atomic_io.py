from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_json_object(
    path: Path,
    *,
    retries: int = 6,
    retry_delay_sec: float = 0.05,
) -> Any:
    last_error: json.JSONDecodeError | None = None
    for attempt in range(max(1, retries)):
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except json.JSONDecodeError as exc:
            last_error = exc
            if attempt + 1 >= retries:
                break
            time.sleep(retry_delay_sec * (attempt + 1))
    assert last_error is not None
    raise last_error


def save_json_object(
    path: Path,
    payload: Any,
    *,
    indent: int = 2,
    sort_keys: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=indent, sort_keys=sort_keys)
            handle.write("\n")
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                logger.exception("failed to remove temporary JSON file: %s", tmp_path)
