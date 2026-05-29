from __future__ import annotations

import os

DEFAULT_RUNTIME_SYMBOLS: tuple[str, ...] = ("BTCUSDT", "ETHUSDT")
DEFAULT_RUNTIME_SYMBOLS_ENV = "FACTOR_RANKING_SYMBOLS"


def configured_runtime_symbols(env_var: str = DEFAULT_RUNTIME_SYMBOLS_ENV) -> tuple[str, ...]:
    raw = os.getenv(env_var)
    if raw is None:
        return DEFAULT_RUNTIME_SYMBOLS
    symbols = parse_symbol_csv(raw)
    if not symbols:
        raise ValueError(f"{env_var} must include at least one symbol when set")
    return symbols


def parse_symbol_csv(raw: str) -> tuple[str, ...]:
    symbols = tuple(dict.fromkeys(part.strip().upper() for part in raw.split(",") if part.strip()))
    return symbols
