from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.services.factor_combination_service import (
    CombinationSearchConfig,
    DEFAULT_AGENT_FACTOR_LIMIT,
    DEFAULT_BASE_FACTOR_LIMIT,
    DEFAULT_BEAM_WIDTH,
    DEFAULT_MINED_FACTOR_LIMIT,
    DEFAULT_NATIVE_FACTOR_LIMIT,
    DEFAULT_PARALLEL_WORKERS,
    DEFAULT_PREFILTER_LIMIT,
    DEFAULT_LOOKBACK_BARS,
    DEFAULT_RESULT_LIMIT,
)
from app.services.high_winrate_strategy_demotion import (
    STATUS_PAPER_LIVE_PASSED,
    STATUS_TRADABLE,
    high_winrate_demotion_status,
)
from app.services.lstm_config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_FEATURE_WINDOW,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MIN_MOVE_BPS,
    DEFAULT_NUM_LAYERS,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VAL_RATIO,
    LstmTrainingConfig,
)
from app.services.lstm_shadow_learning import lstm_shadow_learning_summary

EXPERIMENT_PROFILE_FAST = "fast"
EXPERIMENT_PROFILE_FULL = "full"
EXPERIMENT_PROFILES = (EXPERIMENT_PROFILE_FAST, EXPERIMENT_PROFILE_FULL)

FAST_COMBINATION_CONFIG = CombinationSearchConfig(
    base_factor_limit=72,
    native_factor_limit=72,
    mined_factor_limit=6,
    agent_factor_limit=6,
    combo_sizes=(2,),
    result_limit=250,
    prefilter_limit=200,
    beam_width=200,
    parallel_workers=1,
    lookback_bars=DEFAULT_LOOKBACK_BARS,
)
FULL_COMBINATION_CONFIG = CombinationSearchConfig(
    base_factor_limit=DEFAULT_BASE_FACTOR_LIMIT,
    native_factor_limit=DEFAULT_NATIVE_FACTOR_LIMIT,
    mined_factor_limit=DEFAULT_MINED_FACTOR_LIMIT,
    agent_factor_limit=DEFAULT_AGENT_FACTOR_LIMIT,
    combo_sizes=(2,),
    result_limit=DEFAULT_RESULT_LIMIT,
    prefilter_limit=DEFAULT_PREFILTER_LIMIT,
    beam_width=DEFAULT_BEAM_WIDTH,
    parallel_workers=DEFAULT_PARALLEL_WORKERS,
    lookback_bars=DEFAULT_LOOKBACK_BARS,
)

FAST_LSTM_CONFIG = {
    "feature_window": 32,
    "epochs": 2,
    "batch_size": 64,
    "hidden_size": 32,
    "num_layers": 1,
    "min_samples": 80,
    "min_move_bps": DEFAULT_MIN_MOVE_BPS,
    "learning_rate": DEFAULT_LEARNING_RATE,
    "train_ratio": DEFAULT_TRAIN_RATIO,
    "val_ratio": DEFAULT_VAL_RATIO,
}
FULL_LSTM_CONFIG = {
    "feature_window": DEFAULT_FEATURE_WINDOW,
    "epochs": DEFAULT_EPOCHS,
    "batch_size": DEFAULT_BATCH_SIZE,
    "hidden_size": DEFAULT_HIDDEN_SIZE,
    "num_layers": DEFAULT_NUM_LAYERS,
    "min_samples": 120,
    "min_move_bps": DEFAULT_MIN_MOVE_BPS,
    "learning_rate": DEFAULT_LEARNING_RATE,
    "train_ratio": DEFAULT_TRAIN_RATIO,
    "val_ratio": DEFAULT_VAL_RATIO,
}

LSTM_SHADOW_MIN_SAMPLE_COUNT = 20
LSTM_SHADOW_MIN_WIN_RATE = 0.55
LSTM_SHADOW_MIN_RECENT_WIN_RATE = 0.50
LSTM_SHADOW_MIN_AVG_RETURN = 0.0


@dataclass(frozen=True)
class ShadowGateResult:
    ready: bool
    reason: str
    diagnostics: dict[str, Any]


def normalize_experiment_profile(profile: str | None) -> str:
    value = (profile or EXPERIMENT_PROFILE_FULL).strip().lower()
    if value not in EXPERIMENT_PROFILES:
        raise ValueError(f"unsupported experiment profile: {profile}")
    return value


def combination_search_config_for_profile(
    profile: str,
    *,
    base_factor_limit: int | None = None,
    native_factor_limit: int | None = None,
    mined_factor_limit: int | None = None,
    agent_factor_limit: int | None = None,
    combo_sizes: tuple[int, ...] | None = None,
    result_limit: int | None = None,
    lookback_days: int | None = None,
    lookback_bars: int | None = None,
) -> CombinationSearchConfig:
    selected = _combination_profile(profile)
    return CombinationSearchConfig(
        base_factor_limit=base_factor_limit if base_factor_limit is not None else selected.base_factor_limit,
        native_factor_limit=native_factor_limit if native_factor_limit is not None else selected.native_factor_limit,
        mined_factor_limit=mined_factor_limit if mined_factor_limit is not None else selected.mined_factor_limit,
        agent_factor_limit=agent_factor_limit if agent_factor_limit is not None else selected.agent_factor_limit,
        combo_sizes=combo_sizes if combo_sizes is not None else selected.combo_sizes,
        result_limit=result_limit if result_limit is not None else selected.result_limit,
        prefilter_limit=selected.prefilter_limit,
        beam_width=selected.beam_width,
        parallel_workers=selected.parallel_workers,
        lookback_days=lookback_days if lookback_days is not None else selected.lookback_days,
        lookback_bars=lookback_bars if lookback_bars is not None else selected.lookback_bars,
    )


def lstm_training_config_for_profile(
    symbol: str,
    duration: str,
    profile: str,
    *,
    feature_window: int | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    min_samples: int | None = None,
    learning_rate: float | None = None,
    hidden_size: int | None = None,
    num_layers: int | None = None,
    min_move_bps: float | None = None,
    train_ratio: float | None = None,
    val_ratio: float | None = None,
    seed: int | None = None,
) -> LstmTrainingConfig:
    selected = _lstm_profile(profile)
    kwargs = {
        "feature_window": feature_window if feature_window is not None else selected["feature_window"],
        "epochs": epochs if epochs is not None else selected["epochs"],
        "batch_size": batch_size if batch_size is not None else selected["batch_size"],
        "min_samples": min_samples if min_samples is not None else selected["min_samples"],
        "learning_rate": learning_rate if learning_rate is not None else selected["learning_rate"],
        "hidden_size": hidden_size if hidden_size is not None else selected["hidden_size"],
        "num_layers": num_layers if num_layers is not None else selected["num_layers"],
        "min_move_bps": min_move_bps if min_move_bps is not None else selected["min_move_bps"],
        "train_ratio": train_ratio if train_ratio is not None else selected["train_ratio"],
        "val_ratio": val_ratio if val_ratio is not None else selected["val_ratio"],
    }
    if seed is not None:
        kwargs["seed"] = seed
    return LstmTrainingConfig(symbol=symbol.strip().upper(), duration=duration, **kwargs)


def shadow_gate_for_full_profile(symbol: str, duration: str) -> ShadowGateResult:
    sym = symbol.strip().upper()
    lstm = lstm_shadow_learning_summary(sym, duration)
    high = high_winrate_demotion_status(sym, duration)
    lstm_ready = _lstm_shadow_ready(lstm)
    high_ready = _high_winrate_ready(high)
    if lstm_ready and high_ready:
        return ShadowGateResult(True, "passed", {"lstmShadow": lstm, "highWinrate": high})
    reason = "lstm_shadow_not_ready" if not lstm_ready else "high_winrate_shadow_not_ready"
    return ShadowGateResult(False, reason, {"lstmShadow": lstm, "highWinrate": high})


def _combination_profile(profile: str) -> CombinationSearchConfig:
    value = normalize_experiment_profile(profile)
    return FAST_COMBINATION_CONFIG if value == EXPERIMENT_PROFILE_FAST else FULL_COMBINATION_CONFIG


def _lstm_profile(profile: str) -> dict[str, Any]:
    value = normalize_experiment_profile(profile)
    return FAST_LSTM_CONFIG if value == EXPERIMENT_PROFILE_FAST else FULL_LSTM_CONFIG


def _lstm_shadow_ready(summary: dict[str, Any]) -> bool:
    sample_count = int(summary.get("sampleCount") or 0)
    win_rate = _finite_float(summary.get("winRate"))
    recent_win_rate = _finite_float(summary.get("recentWinRate"))
    avg_return = _finite_float(summary.get("avgReturn"))
    return (
        sample_count >= LSTM_SHADOW_MIN_SAMPLE_COUNT
        and win_rate is not None
        and win_rate >= LSTM_SHADOW_MIN_WIN_RATE
        and recent_win_rate is not None
        and recent_win_rate >= LSTM_SHADOW_MIN_RECENT_WIN_RATE
        and avg_return is not None
        and avg_return > LSTM_SHADOW_MIN_AVG_RETURN
    )


def _high_winrate_ready(status: dict[str, Any]) -> bool:
    return str(status.get("status") or "") in {STATUS_PAPER_LIVE_PASSED, STATUS_TRADABLE}


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and number not in (float("inf"), float("-inf")) else None
