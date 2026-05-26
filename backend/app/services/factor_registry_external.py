from __future__ import annotations

from typing import Any

from app.services.factor_extensions import ADDITIONAL_FACTORS
from app.services.factor_external_extensions import EXTERNAL_DATA_FACTORS
from app.services.factor_registry_core import FactorCategory, FactorDefinition, FactorDirection


def _factor_from_payload(payload: dict[str, Any]) -> FactorDefinition:
    data = dict(payload)
    data["category"] = FactorCategory(str(data["category"]))
    data["direction"] = FactorDirection(str(data.get("direction", FactorDirection.NEUTRAL)))
    return FactorDefinition(**data)


EXTENDED_KLINE_FACTORS = tuple(
    _factor_from_payload(payload) for payload in (*ADDITIONAL_FACTORS, *EXTERNAL_DATA_FACTORS)
)
