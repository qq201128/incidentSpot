from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.services.model_family_torch_helpers import position_tensor, returns_or_zeros, weighted_loss

DEFAULT_DROPOUT = 0.0
DEFAULT_TRANSFORMER_HEADS = 4
PREDICT_BATCH_SIZE = 1024


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
    dropout: float = DEFAULT_DROPOUT
    weight_decay: float = 0.0
    early_stopping_patience: int = 0
    class_weight_mode: str = "none"
    return_weight_mode: str = "none"
    transformer_nhead: int = DEFAULT_TRANSFORMER_HEADS
    use_positional_encoding: bool = False


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
        train_returns: np.ndarray | None = None,
        val_returns: np.ndarray | None = None,
        scaler: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        torch, nn = _torch_modules()
        torch.manual_seed(options.seed)
        mean, std = _scaler_arrays(scaler)
        model = _model_for_family(nn, options)
        optimizer = torch.optim.Adam(model.parameters(), lr=options.learning_rate, weight_decay=options.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        best_state, losses = _train_epochs(
            torch,
            model,
            optimizer,
            loss_fn,
            ArrayPack(train_x, train_y, returns_or_zeros(train_y, train_returns), options, mean, std),
            ArrayPack(val_x, val_y, returns_or_zeros(val_y, val_returns), options, mean, std),
            options,
        )
        model.load_state_dict(best_state or model.state_dict())
        self._trained_model = model
        self._trained_options = options
        self._trained_scaler = (mean, std) if mean is not None and std is not None else None
        if persist_model:
            _save_model(torch, model, options, model_path)
        return {"trainLoss": losses["train"], "valLoss": losses["val"]}

    def predict_trained(self, x: np.ndarray) -> np.ndarray:
        torch, _nn = _torch_modules()
        model = getattr(self, "_trained_model", None)
        if model is None:
            raise RuntimeError("torch sequence model has not been trained")
        return _predict_model(torch, model, x, getattr(self, "_trained_scaler", None))

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
            dropout=float(payload.get("dropout") or DEFAULT_DROPOUT),
            transformer_nhead=int(payload.get("transformer_nhead") or DEFAULT_TRANSFORMER_HEADS),
            use_positional_encoding=bool(payload.get("use_positional_encoding") or False),
        )
        model = _model_for_family(nn, options)
        model.load_state_dict(payload["state_dict"])
        return _predict_model(torch, model, x, None)


def _predict_model(torch, model, x: np.ndarray, scaler) -> np.ndarray:
    model.eval()
    batches = []
    with torch.no_grad():
        for start in range(0, x.shape[0], PREDICT_BATCH_SIZE):
            batch = _tensor_batch(torch, x, slice(start, start + PREDICT_BATCH_SIZE), scaler)
            logits = model(batch)
            batches.append(torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(batches) if batches else np.empty(0, dtype=np.float32)


@dataclass(frozen=True)
class ArrayPack:
    x: np.ndarray
    y: np.ndarray
    returns: np.ndarray
    options: TorchSequenceOptions
    mean: np.ndarray | None
    std: np.ndarray | None


def _train_epochs(torch, model, optimizer, loss_fn, train_pack, val_pack, options):
    best_loss = float("inf")
    best_state = None
    last_train_loss = 0.0
    stale_epochs = 0
    epochs_run = 0
    for _epoch in range(options.epochs):
        last_train_loss = _train_one_epoch(torch, model, optimizer, loss_fn, train_pack, options.batch_size)
        val_loss = _validation_loss(model, loss_fn, val_pack)
        epochs_run += 1
        if val_loss < best_loss:
            best_loss = val_loss
            stale_epochs = 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            continue
        stale_epochs += 1
        if options.early_stopping_patience > 0 and stale_epochs >= options.early_stopping_patience:
            break
    return best_state, {"train": float(last_train_loss), "val": float(best_loss), "epochsRun": epochs_run}


def _train_one_epoch(torch, model, optimizer, loss_fn, train_pack, batch_size: int) -> float:
    losses = []
    model.train()
    permutation = torch.randperm(train_pack.x.shape[0]).numpy()
    for start in range(0, train_pack.x.shape[0], batch_size):
        idx = permutation[start:start + batch_size]
        optimizer.zero_grad()
        batch_x, batch_y, batch_returns = _batch_pack(torch, train_pack, idx)
        loss = weighted_loss(loss_fn, model(batch_x), batch_y, batch_returns, train_pack.options)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def _validation_loss(model, loss_fn, val_pack) -> float:
    import torch

    losses = []
    model.eval()
    with torch.no_grad():
        for start in range(0, val_pack.x.shape[0], val_pack.options.batch_size):
            idx = slice(start, start + val_pack.options.batch_size)
            batch_x, batch_y, batch_returns = _batch_pack(torch, val_pack, idx)
            loss = weighted_loss(loss_fn, model(batch_x), batch_y, batch_returns, val_pack.options)
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("inf")


def _batch_pack(torch, pack: ArrayPack, idx):
    scaler = (pack.mean, pack.std) if pack.mean is not None and pack.std is not None else None
    return (
        _tensor_batch(torch, pack.x, idx, scaler),
        _tensor_vector(torch, pack.y, idx),
        _tensor_vector(torch, pack.returns, idx),
    )


def _tensor_batch(torch, values: np.ndarray, idx, scaler=None):
    batch = np.array(values[idx], dtype=np.float32, copy=True)
    if scaler is not None:
        mean, std = scaler
        batch = ((batch - mean) / std).astype(np.float32)
    return torch.as_tensor(batch, dtype=torch.float32)


def _tensor_vector(torch, values: np.ndarray, idx):
    return torch.as_tensor(np.array(values[idx], dtype=np.float32, copy=True), dtype=torch.float32)


def _scaler_arrays(scaler: dict[str, Any] | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    if scaler is None:
        return None, None
    return np.asarray(scaler["mean"], dtype=np.float32), np.asarray(scaler["std"], dtype=np.float32)


def _model_for_family(nn, options: TorchSequenceOptions):
    if options.family == "transformer" and options.hidden_size % options.transformer_nhead != 0:
        raise ValueError("transformer hidden_size must be divisible by transformer_nhead")
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
                self.dropout = nn.Dropout(options.dropout)
                self.head = nn.Linear(options.hidden_size, 1)

            def forward(self, x):
                out, _state = self.rnn(x)
                return self.head(self.dropout(out[:, -1, :])).squeeze(-1)

        return Model()


class _GruClassifier:
    def __new__(cls, nn, options):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.rnn = nn.GRU(options.input_size, options.hidden_size, options.num_layers, batch_first=True)
                self.dropout = nn.Dropout(options.dropout)
                self.head = nn.Linear(options.hidden_size, 1)

            def forward(self, x):
                out, _state = self.rnn(x)
                return self.head(self.dropout(out[:, -1, :])).squeeze(-1)

        return Model()


class _CnnClassifier:
    def __new__(cls, nn, options):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv1d(options.input_size, options.hidden_size, kernel_size=3, padding=1)
                self.dropout = nn.Dropout(options.dropout)
                self.head = nn.Linear(options.hidden_size, 1)

            def forward(self, x):
                out = self.conv(x.transpose(1, 2)).amax(dim=2)
                return self.head(self.dropout(out)).squeeze(-1)

        return Model()


class _TransformerClassifier:
    def __new__(cls, nn, options):
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.proj = nn.Linear(options.input_size, options.hidden_size)
                self.positional = _PositionalEncoding(nn, options.hidden_size) if options.use_positional_encoding else None
                layer = nn.TransformerEncoderLayer(
                    d_model=options.hidden_size,
                    nhead=options.transformer_nhead,
                    batch_first=True,
                    dim_feedforward=options.hidden_size * 2,
                    dropout=options.dropout,
                )
                self.encoder = nn.TransformerEncoder(layer, num_layers=options.num_layers)
                self.head = nn.Linear(options.hidden_size, 1)

            def forward(self, x):
                projected = self.proj(x)
                encoded = projected if self.positional is None else self.positional(projected)
                out = self.encoder(encoded)
                return self.head(out[:, -1, :]).squeeze(-1)

        return Model()


class _PositionalEncoding:
    def __new__(cls, nn, hidden_size: int):
        class Module(nn.Module):
            def forward(self, x):
                positions = position_tensor(x)
                return x + positions.to(device=x.device, dtype=x.dtype)

        return Module()


def _save_model(torch, model, options: TorchSequenceOptions, model_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "family": options.family,
            "input_size": options.input_size,
            "hidden_size": options.hidden_size,
            "num_layers": options.num_layers,
            "dropout": options.dropout,
            "transformer_nhead": options.transformer_nhead,
            "use_positional_encoding": options.use_positional_encoding,
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
