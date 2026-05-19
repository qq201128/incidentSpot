from __future__ import annotations

from app.services.lstm_config import LstmTrainingConfig


def search_key_for_config(config: LstmTrainingConfig, profile: str) -> str:
    bps = f"{float(config.min_move_bps):g}"
    return (
        f"profile={profile}|duration={config.duration}|window={config.feature_window}|"
        f"move_bps={bps}|epochs={config.epochs}|seed={config.seed}"
    )
