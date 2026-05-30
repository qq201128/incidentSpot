from __future__ import annotations

import os


def float_env(name: str, default: float, *, min_value: float) -> float:
    raw = os.getenv(name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric: {raw!r}") from exc
    return max(min_value, value)
