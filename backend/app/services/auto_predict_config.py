from __future__ import annotations

import os


def predict_initial_delay_seconds() -> float:
    raw = os.getenv("PREDICT_INITIAL_DELAY_SECONDS", "8")
    try:
        return max(0.0, float(raw))
    except ValueError as exc:
        raise ValueError(f"PREDICT_INITIAL_DELAY_SECONDS must be numeric: {raw!r}") from exc
