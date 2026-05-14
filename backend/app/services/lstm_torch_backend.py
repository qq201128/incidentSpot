from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TorchLstmOptions:
    input_size: int
    hidden_size: int
    num_layers: int
    learning_rate: float
    batch_size: int
    epochs: int
    seed: int


class TorchLstmBackend:
    def train(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        val_x: np.ndarray,
        val_y: np.ndarray,
        *,
        options: TorchLstmOptions,
        model_path: Path,
    ) -> dict[str, Any]:
        torch, nn = _torch_modules()
        torch.manual_seed(options.seed)
        model = _model_class(nn)(options.input_size, options.hidden_size, options.num_layers)
        optimizer = torch.optim.Adam(model.parameters(), lr=options.learning_rate)
        loss_fn = nn.BCEWithLogitsLoss()
        best_state, losses = _train_epochs(
            model, optimizer, loss_fn, _tensor_pack(torch, train_x, train_y),
            _tensor_pack(torch, val_x, val_y), options,
        )
        if best_state is not None:
            model.load_state_dict(best_state)
        _save_model(torch, model, options, model_path)
        return {"trainLoss": losses["train"], "valLoss": losses["val"]}

    def predict(self, model_path: Path, x: np.ndarray) -> np.ndarray:
        torch, nn = _torch_modules()
        payload = torch.load(model_path, map_location="cpu", weights_only=True)
        model = _model_class(nn)(
            int(payload["input_size"]),
            int(payload["hidden_size"]),
            int(payload["num_layers"]),
        )
        model.load_state_dict(payload["state_dict"])
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(x, dtype=torch.float32))
            probs = torch.sigmoid(logits).detach().cpu().numpy()
        return probs.astype(np.float32)


def _train_epochs(model, optimizer, loss_fn, train_pack, val_pack, options: TorchLstmOptions):
    torch, _nn = _torch_modules()
    best_loss = float("inf")
    best_state = None
    last_train_loss = None
    for _epoch in range(options.epochs):
        last_train_loss = _train_one_epoch(torch, model, optimizer, loss_fn, train_pack, options.batch_size)
        val_loss = _validation_loss(model, loss_fn, val_pack)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    return best_state, {"train": float(last_train_loss or 0.0), "val": float(best_loss)}


def _train_one_epoch(torch, model, optimizer, loss_fn, train_pack, batch_size: int) -> float:
    x, y = train_pack
    model.train()
    permutation = torch.randperm(x.shape[0])
    losses = []
    for start in range(0, x.shape[0], batch_size):
        idx = permutation[start:start + batch_size]
        optimizer.zero_grad()
        loss = loss_fn(model(x[idx]), y[idx])
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def _validation_loss(model, loss_fn, val_pack) -> float:
    x, y = val_pack
    model.eval()
    with _no_grad():
        return float(loss_fn(model(x), y).detach().cpu())


def _tensor_pack(torch, x: np.ndarray, y: np.ndarray):
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def _save_model(torch, model, options: TorchLstmOptions, model_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_size": options.input_size,
            "hidden_size": options.hidden_size,
            "num_layers": options.num_layers,
        },
        model_path,
    )


def _model_class(nn):
    class SequenceClassifier(nn.Module):
        def __init__(self, input_size: int, hidden_size: int, num_layers: int) -> None:
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _state = self.lstm(x)
            return self.head(out[:, -1, :]).squeeze(-1)

    return SequenceClassifier


def is_torch_available() -> bool:
    return bool(torch_availability()["available"])


def torch_availability() -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        return {"available": False, "error": str(exc)}
    return {
        "available": True,
        "version": getattr(torch, "__version__", None),
        "modulePath": getattr(torch, "__file__", None),
    }


def _torch_modules():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise ImportError("missing dependency: torch is required for LSTM training/prediction") from exc
    return torch, nn


def _no_grad():
    torch, _nn = _torch_modules()
    return torch.no_grad()
