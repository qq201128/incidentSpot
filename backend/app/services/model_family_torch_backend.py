from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TorchSequenceOptions:
    family: str
    input_size: int
    hidden_size: int
    num_layers: int
    learning_rate: float
    batch_size: int
    epochs: int
    seed: int


class TorchSequenceBackend:
    def train(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        val_x: np.ndarray,
        val_y: np.ndarray,
        *,
        options: TorchSequenceOptions,
        model_path: Path,
        persist_model: bool = True,
    ) -> dict[str, Any]:
        torch, nn = _torch_modules()
        torch.manual_seed(options.seed)
        model = _model_for_family(nn, options)
        optimizer = torch.optim.Adam(model.parameters(), lr=options.learning_rate)
        loss_fn = nn.BCEWithLogitsLoss()
        best_state, losses = _train_epochs(
            torch,
            model,
            optimizer,
            loss_fn,
            _tensor_pack(torch, train_x, train_y),
            _tensor_pack(torch, val_x, val_y),
            options,
        )
        model.load_state_dict(best_state or model.state_dict())
        self._trained_model = model
        self._trained_options = options
        if persist_model:
            _save_model(torch, model, options, model_path)
        return {"trainLoss": losses["train"], "valLoss": losses["val"]}

    def predict_trained(self, x: np.ndarray) -> np.ndarray:
        torch, _nn = _torch_modules()
        model = getattr(self, "_trained_model", None)
        if model is None:
            raise RuntimeError("torch sequence model has not been trained")
        return _predict_model(torch, model, x)

    def predict(self, model_path: Path, x: np.ndarray) -> np.ndarray:
        torch, nn = _torch_modules()
        payload = torch.load(model_path, map_location="cpu", weights_only=True)
        options = TorchSequenceOptions(
            family=str(payload["family"]),
            input_size=int(payload["input_size"]),
            hidden_size=int(payload["hidden_size"]),
            num_layers=int(payload["num_layers"]),
            learning_rate=0.0,
            batch_size=1,
            epochs=1,
            seed=0,
        )
        model = _model_for_family(nn, options)
        model.load_state_dict(payload["state_dict"])
        return _predict_model(torch, model, x)


def _predict_model(torch, model, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32))
    return torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)


def _train_epochs(torch, model, optimizer, loss_fn, train_pack, val_pack, options):
    best_loss = float("inf")
    best_state = None
    last_train_loss = 0.0
    for _epoch in range(options.epochs):
        last_train_loss = _train_one_epoch(torch, model, optimizer, loss_fn, train_pack, options.batch_size)
        val_loss = _validation_loss(model, loss_fn, val_pack)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    return best_state, {"train": float(last_train_loss), "val": float(best_loss)}


def _train_one_epoch(torch, model, optimizer, loss_fn, train_pack, batch_size: int) -> float:
    x, y = train_pack
    losses = []
    model.train()
    for start in range(0, x.shape[0], batch_size):
        idx = torch.randperm(x.shape[0])[start:start + batch_size]
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


def _model_for_family(nn, options: TorchSequenceOptions):
    if options.family == "gru":
        return _GruClassifier(nn, options)
    if options.family == "cnn":
        return _CnnClassifier(nn, options)
    if options.family == "transformer":
        return _TransformerClassifier(nn, options)
    return _LstmClassifier(nn, options)


class _LstmClassifier:
    def __new__(cls, nn, options):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.rnn = nn.LSTM(options.input_size, options.hidden_size, options.num_layers, batch_first=True)
                self.head = nn.Linear(options.hidden_size, 1)

            def forward(self, x):
                out, _state = self.rnn(x)
                return self.head(out[:, -1, :]).squeeze(-1)

        return Model()


class _GruClassifier:
    def __new__(cls, nn, options):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.rnn = nn.GRU(options.input_size, options.hidden_size, options.num_layers, batch_first=True)
                self.head = nn.Linear(options.hidden_size, 1)

            def forward(self, x):
                out, _state = self.rnn(x)
                return self.head(out[:, -1, :]).squeeze(-1)

        return Model()


class _CnnClassifier:
    def __new__(cls, nn, options):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv1d(options.input_size, options.hidden_size, kernel_size=3, padding=1)
                self.head = nn.Linear(options.hidden_size, 1)

            def forward(self, x):
                out = self.conv(x.transpose(1, 2)).amax(dim=2)
                return self.head(out).squeeze(-1)

        return Model()


class _TransformerClassifier:
    def __new__(cls, nn, options):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.proj = nn.Linear(options.input_size, options.hidden_size)
                layer = nn.TransformerEncoderLayer(
                    d_model=options.hidden_size,
                    nhead=4,
                    batch_first=True,
                    dim_feedforward=options.hidden_size * 2,
                )
                self.encoder = nn.TransformerEncoder(layer, num_layers=options.num_layers)
                self.head = nn.Linear(options.hidden_size, 1)

            def forward(self, x):
                out = self.encoder(self.proj(x))
                return self.head(out[:, -1, :]).squeeze(-1)

        return Model()


def _tensor_pack(torch, x: np.ndarray, y: np.ndarray):
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def _save_model(torch, model, options: TorchSequenceOptions, model_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "family": options.family,
            "input_size": options.input_size,
            "hidden_size": options.hidden_size,
            "num_layers": options.num_layers,
        },
        model_path,
    )


def _torch_modules():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise ImportError("missing dependency: torch is required for neural model training/prediction") from exc
    return torch, nn


def _no_grad():
    torch, _nn = _torch_modules()
    return torch.no_grad()
