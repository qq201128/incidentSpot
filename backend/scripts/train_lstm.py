from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.experiment_profiles import lstm_training_config_for_profile, normalize_experiment_profile
from app.services.lstm_training_service import train_lstm_model
from app.services.runtime_symbols import parse_symbol_csv


def main() -> None:
    args = _parse_args()
    profile = normalize_experiment_profile(args.profile)
    symbol = _selected_symbol(args)
    report = train_lstm_model(
        lstm_training_config_for_profile(
            symbol,
            args.duration,
            profile,
            feature_window=args.feature_window,
            epochs=args.epochs,
            batch_size=args.batch_size,
            min_samples=args.min_samples,
            learning_rate=args.learning_rate,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            min_move_bps=args.min_move_bps,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an LSTM shadow strategy model.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--duration", default="10m", choices=("10m", "30m", "60m", "1d"))
    parser.add_argument("--profile", default="full", choices=("fast", "full"))
    parser.add_argument("--feature-window", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--min-samples", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--min-move-bps", type=float, default=None)
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def _selected_symbol(args: argparse.Namespace) -> str:
    symbols = parse_symbol_csv(args.symbol)
    if len(symbols) != 1:
        raise ValueError("train_lstm.py accepts exactly one --symbol; use queued model-search jobs for batches")
    return symbols[0]


if __name__ == "__main__":
    main()
