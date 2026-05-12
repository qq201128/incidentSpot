from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"
FACTOR_LEARNING_DIR = MODEL_DIR / "factor_learning"
FACTOR_LEARNING_VERSION = "factor_learning_v1"


def factor_learning_memory_path(symbol: str, duration: str) -> Path:
    safe_symbol = symbol.strip().upper()
    safe_duration = duration.strip()
    if not safe_symbol or not safe_duration:
        raise ValueError("symbol and duration are required")
    return FACTOR_LEARNING_DIR / f"{safe_symbol}_{safe_duration}.json"


def load_factor_learning_memory(symbol: str, duration: str) -> dict[str, Any] | None:
    path = factor_learning_memory_path(symbol, duration)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"factor learning memory is not an object: {path}")
    return payload


def save_factor_learning_memory(memory: dict[str, Any]) -> Path:
    symbol = str(memory["symbol"])
    duration = str(memory["duration"])
    path = factor_learning_memory_path(symbol, duration)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(memory, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return path
