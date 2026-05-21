from __future__ import annotations

import hashlib

FACTOR_CANDIDATE_SIGNAL_PREFIX = "factor_candidate_signal_"


def factor_candidate_signal_key(factor_name: str) -> str:
    digest = hashlib.sha1(str(factor_name).encode("utf-8")).hexdigest()[:12]
    return f"{FACTOR_CANDIDATE_SIGNAL_PREFIX}{digest}"


def is_factor_candidate_signal_key(signal_key: str | None) -> bool:
    return str(signal_key or "").startswith(FACTOR_CANDIDATE_SIGNAL_PREFIX)


def factor_candidate_signal_strategy_key(factor_name: str) -> str:
    return factor_candidate_signal_key(factor_name)


def is_factor_candidate_signal_strategy(strategy_key: str | None) -> bool:
    return is_factor_candidate_signal_key(strategy_key)
