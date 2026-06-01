from __future__ import annotations

from typing import Any

def with_simulation_status(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from app.services.simulation_slot_observability_service import simulation_statuses_for_slots

    statuses = simulation_statuses_for_slots(payloads)
    return [
        {
            **payload,
            "simulationStatus": statuses.get(_payload_key(payload)),
        }
        for payload in payloads
    ]


def _payload_key(payload: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(payload["strategyKey"]),
        str(payload["symbol"]).upper(),
        str(payload["duration"]),
    )
