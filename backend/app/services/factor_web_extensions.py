from __future__ import annotations

from typing import Any

from app.services.kline_web_factor_specs import WEB_FACTOR_SPECS

SOURCE_FILE = "kline_web_factors.py"


def _factor_payload(spec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "category": spec.category,
        "description": spec.description,
        "formula": spec.formula,
        "source_file": SOURCE_FILE,
        "direction": "neutral",
        "parameters": {"window": spec.window, "kind": spec.kind},
    }


WEB_FACTORS: tuple[dict[str, Any], ...] = tuple(_factor_payload(spec) for spec in WEB_FACTOR_SPECS)
