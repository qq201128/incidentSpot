from __future__ import annotations

from typing import Any

import numpy as np

RETURN_WEIGHT_CLIP = 3.0
WEIGHT_EPSILON = 1e-6
POSITION_ENCODING_BASE = 10_000.0


def tensor_pack(torch, x: np.ndarray, y: np.ndarray, returns: np.ndarray | None, options: Any):
    return (
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
        torch.tensor(returns_or_zeros(y, returns), dtype=torch.float32),
        options,
    )


def weighted_loss(loss_fn, logits, labels, returns, options: Any):
    losses = loss_fn(logits, labels)
    weights = _class_weights(labels, options) * _return_weights(returns, options)
    return (losses * weights).mean()


def position_tensor(x):
    import torch

    steps = torch.arange(x.shape[1], device=x.device, dtype=x.dtype).unsqueeze(1)
    dims = torch.arange(x.shape[2], device=x.device, dtype=x.dtype).unsqueeze(0)
    base = torch.tensor(POSITION_ENCODING_BASE, device=x.device, dtype=x.dtype)
    scale = torch.pow(base, (2 * (dims // 2)) / x.shape[2])
    angles = steps / scale
    return torch.where((dims % 2) == 0, torch.sin(angles), torch.cos(angles)).unsqueeze(0)


def _class_weights(labels, options: Any):
    if options.class_weight_mode != "balanced":
        return labels.new_ones(labels.shape)
    positives = labels.sum().clamp_min(WEIGHT_EPSILON)
    negatives = (labels.shape[0] - labels.sum()).clamp_min(WEIGHT_EPSILON)
    pos_weight = (negatives / positives).clamp_min(WEIGHT_EPSILON)
    neg_weight = (positives / negatives).clamp_min(WEIGHT_EPSILON)
    return labels * pos_weight + (1.0 - labels) * neg_weight


def _return_weights(returns, options: Any):
    if options.return_weight_mode != "abs_return":
        return returns.new_ones(returns.shape)
    scale = returns.abs().mean().clamp_min(WEIGHT_EPSILON)
    return (returns.abs() / scale).clamp(max=RETURN_WEIGHT_CLIP).clamp_min(WEIGHT_EPSILON)


def returns_or_zeros(y: np.ndarray, returns: np.ndarray | None) -> np.ndarray:
    if returns is None:
        return np.zeros_like(y, dtype=np.float32)
    return returns.astype(np.float32)
