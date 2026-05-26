from __future__ import annotations

import logging
from typing import Any, Callable

from app.services.auto_trade_types import AutoTradeSettings
from app.services.model_family_config import MODEL_FAMILIES, is_model_family_shadow_strategy, parse_model_family_strategy
from app.services.strategy_registry import DEFAULT_STRATEGY_KEY, FACTOR_COMBO_STRATEGY_KEY


def prediction_targets(
    settings: list[AutoTradeSettings],
    *,
    get_default_settings: Callable[[str], AutoTradeSettings],
    readiness: Callable[..., Any],
    supports_duration: Callable[[str, str], bool],
    logger: logging.Logger,
) -> list[AutoTradeSettings]:
    enabled = enabled_prediction_targets(settings, supports_duration)
    if enabled:
        return enabled
    if any(item.enabled for item in settings):
        return []
    return default_prediction_targets(get_default_settings, readiness, logger)


def default_prediction_targets(
    get_default_settings: Callable[[str], AutoTradeSettings],
    readiness: Callable[..., Any],
    logger: logging.Logger,
) -> list[AutoTradeSettings]:
    default = get_default_settings(DEFAULT_STRATEGY_KEY)
    status = readiness(default.strategy_key, default.symbol, default.duration, attempt_recovery=True)
    if status.ready:
        return [default]
    logger.warning(
        "default predict target skipped strategy=%s symbol=%s duration=%s reason=%s recoveryAttempted=%s recoveryStatus=%s diagnostics=%s",
        default.strategy_key,
        default.symbol,
        default.duration,
        status.reason,
        status.recovery_attempted,
        status.recovery_status,
        status.diagnostics,
    )
    return []


def enabled_prediction_targets(
    settings: list[AutoTradeSettings],
    supports_duration: Callable[[str, str], bool],
) -> list[AutoTradeSettings]:
    return [item for item in settings if item.enabled and supports_duration(item.strategy_key, item.duration)]


def ready_due_prediction_targets(
    targets: list[AutoTradeSettings],
    *,
    due_targets: Callable[[list[AutoTradeSettings]], list[AutoTradeSettings]],
    readiness: Callable[..., Any],
    logger: logging.Logger,
) -> list[AutoTradeSettings]:
    ready = []
    for settings in due_targets(targets):
        status = readiness(settings.strategy_key, settings.symbol, settings.duration, attempt_recovery=False)
        if status.ready:
            ready.append(settings)
            continue
        log_readiness_skip(logger, settings, status)
    return ready


def log_readiness_skip(logger: logging.Logger, settings: AutoTradeSettings, status: Any) -> None:
    logger.warning(
        "predict due target skipped strategy=%s symbol=%s duration=%s reason=%s recoveryAttempted=%s recoveryStatus=%s diagnostics=%s",
        settings.strategy_key,
        settings.symbol,
        settings.duration,
        status.reason,
        status.recovery_attempted,
        status.recovery_status,
        status.diagnostics,
    )


def candidate_collection_targets(
    targets: list[AutoTradeSettings],
    *,
    current_bucket: Callable[[str], int],
    collection_settings: Callable[[AutoTradeSettings], AutoTradeSettings],
    collection_due: Callable[[AutoTradeSettings, int], bool],
    supports_duration: Callable[[str, str], bool],
) -> list[AutoTradeSettings]:
    collection = []
    for settings in unique_collection_settings(targets, collection_settings, supports_duration):
        bucket = current_bucket(settings.duration)
        if collection_due(settings, bucket):
            collection.append(settings)
    return collection


def unique_collection_settings(
    targets: list[AutoTradeSettings],
    collection_settings: Callable[[AutoTradeSettings], AutoTradeSettings],
    supports_duration: Callable[[str, str], bool],
) -> list[AutoTradeSettings]:
    selected: dict[tuple[str, str], AutoTradeSettings] = {}
    for settings in targets:
        if not supports_duration(settings.strategy_key, settings.duration):
            continue
        selected[(settings.symbol.upper(), settings.duration)] = collection_settings(settings)
    return list(selected.values())


def collection_settings(settings: AutoTradeSettings) -> AutoTradeSettings:
    return AutoTradeSettings(
        strategy_key=FACTOR_COMBO_STRATEGY_KEY,
        enabled=True,
        symbol=settings.symbol.upper(),
        duration=settings.duration,
        duration_minutes=settings.duration_minutes,
        qty=settings.qty,
        live_trading_enabled=False,
    )


def should_predict_entry(
    settings: AutoTradeSettings,
    *,
    current_bucket: Callable[[str], int],
    prediction_exists: Callable[..., bool],
    ready_any_shadow_due: Callable[[AutoTradeSettings, int], bool],
    ready_family_strategy_due: Callable[[AutoTradeSettings, int], bool],
) -> bool:
    bucket = current_bucket(settings.duration)
    if current_bucket_prediction_exists(settings, bucket, prediction_exists):
        return ready_any_shadow_due(settings, bucket)
    if is_model_family_shadow_strategy(settings.strategy_key):
        return ready_family_strategy_due(settings, bucket)
    return True


def current_bucket_prediction_exists(
    settings: AutoTradeSettings,
    bucket: int,
    prediction_exists: Callable[..., bool],
) -> bool:
    return prediction_exists(strategy_key=settings.strategy_key, symbol=settings.symbol, duration=settings.duration, open_time=bucket)


def factor_combo_shadow_due(
    settings: AutoTradeSettings,
    bucket: int,
    *,
    eligible_rows: Callable[[str, str], list[dict[str, Any]]],
    prediction_exists: Callable[..., bool],
    simulation_key: Callable[[str], str],
) -> bool:
    for row in eligible_rows(settings.symbol, settings.duration):
        if not prediction_exists(
            strategy_key=simulation_key(str(row["factorName"])),
            symbol=settings.symbol,
            duration=settings.duration,
            open_time=bucket,
        ):
            return True
    return False


def factor_candidate_signal_due(
    settings: AutoTradeSettings,
    bucket: int,
    *,
    signal_keys: Callable[[str, str], tuple[str, ...]],
    prediction_exists: Callable[..., bool],
) -> bool:
    for signal_key in signal_keys(settings.symbol, settings.duration):
        if not prediction_exists(signal_key=signal_key, symbol=settings.symbol, duration=settings.duration, open_time=bucket):
            return True
    return False


def ready_model_family_strategy_due(
    settings: AutoTradeSettings,
    bucket: int,
    ready_shadow_due: Callable[[str, AutoTradeSettings, int], bool],
) -> bool:
    parsed = parse_model_family_strategy(settings.strategy_key)
    if parsed is None:
        return False
    family, _duration = parsed
    return ready_shadow_due(family, settings, bucket)


def unique_model_family_shadow_targets(settings_list: list[AutoTradeSettings]) -> list[tuple[str, str, str]]:
    targets = set()
    for settings in settings_list:
        if settings.strategy_key == FACTOR_COMBO_STRATEGY_KEY:
            for family in MODEL_FAMILIES:
                targets.add((family, settings.symbol.upper(), settings.duration))
            continue
        parsed = parse_model_family_strategy(settings.strategy_key)
        if parsed is not None:
            family, _duration = parsed
            targets.add((family, settings.symbol.upper(), settings.duration))
    return sorted(targets)
