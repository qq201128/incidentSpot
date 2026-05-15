from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.lstm_config import LstmTrainingConfig
from app.services.lstm_training_service import train_lstm_model


def main() -> None:
    args = _parse_args()
    report = train_lstm_model(
        LstmTrainingConfig(
            symbol=args.symbol,
            duration=args.duration,
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
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--duration", default="10m", choices=("10m", "30m", "60m", "1d"))
    parser.add_argument("--feature-window", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--min-samples", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--min-move-bps", type=float, default=8.0)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260513)
    return parser.parse_args()


if __name__ == "__main__":
    main()
