from __future__ import annotations

import pandas as pd

from app.services.enhanced_features import (
    build_enhanced_feature_frame,
    load_funding_features,
    load_klines,
    load_orderbook_features,
)
from app.services.external_factor_data import load_external_feature_frames
from app.services.rule_config import SUPPORTED_RULE_DURATIONS

FACTOR_FRAME_MIN_HISTORY = 240


def load_factor_frame(
    symbol: str,
    duration: str = "10m",
    *,
    min_history: int = FACTOR_FRAME_MIN_HISTORY,
) -> pd.DataFrame:
    if duration not in SUPPORTED_RULE_DURATIONS:
        raise ValueError(f"unsupported factor frame duration: {duration}")
    df = load_klines(symbol, duration)
    orderbook = load_orderbook_features(symbol)
    funding = load_funding_features(symbol)
    frame, _ = build_enhanced_feature_frame(
        df,
        ob_df=orderbook,
        funding_df=funding,
        external_frames=load_external_feature_frames(symbol),
        min_history=min_history,
    )
    return frame
