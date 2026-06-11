from __future__ import annotations

PAYOUT_RATIO_BY_DURATION = {
    "10m": 0.8,
    "30m": 0.85,
    "60m": 0.85,
    "1d": 0.85,
}

FIXED_PAYOUT_RATIO = PAYOUT_RATIO_BY_DURATION["10m"]


def payout_ratio_for_duration(duration: str) -> float:
    try:
        return PAYOUT_RATIO_BY_DURATION[duration]
    except KeyError as exc:
        raise ValueError(f"unsupported payout duration: {duration}") from exc
