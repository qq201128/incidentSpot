from __future__ import annotations

import ast
from typing import Any

import numpy as np
import pandas as pd

EPSILON = 1e-12


def materialize_agent_formula(frame: pd.DataFrame, formula: str) -> pd.Series:
    tree = ast.parse(formula, mode="eval")
    values = _eval_node(tree.body, frame)
    if not isinstance(values, pd.Series):
        raise ValueError("agent factor formula must produce a series")
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _eval_node(node: ast.AST, frame: pd.DataFrame) -> Any:
    if isinstance(node, ast.Name):
        return _column(frame, node.id)
    if isinstance(node, ast.Constant):
        return _constant(node.value)
    if isinstance(node, ast.BinOp):
        return _binary(node, frame)
    if isinstance(node, ast.UnaryOp):
        return _unary(node, frame)
    if isinstance(node, ast.Compare):
        return _compare(node, frame)
    if isinstance(node, ast.Call):
        return _call(node, frame)
    raise ValueError(f"unsupported formula node: {type(node).__name__}")


def _constant(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("formula constants must be numeric")
    return float(value)


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        raise ValueError(f"formula column not found: {name}")
    return pd.to_numeric(frame[name], errors="coerce")


def _binary(node: ast.BinOp, frame: pd.DataFrame) -> Any:
    left = _eval_node(node.left, frame)
    right = _eval_node(node.right, frame)
    if isinstance(node.op, ast.Add):
        return left + right
    if isinstance(node.op, ast.Sub):
        return left - right
    if isinstance(node.op, ast.Mult):
        return left * right
    if isinstance(node.op, ast.Div):
        return left / _nonzero(right)
    raise ValueError(f"unsupported formula operator: {type(node.op).__name__}")


def _unary(node: ast.UnaryOp, frame: pd.DataFrame) -> Any:
    value = _eval_node(node.operand, frame)
    if isinstance(node.op, ast.USub):
        return -value
    if isinstance(node.op, ast.UAdd):
        return value
    raise ValueError(f"unsupported unary operator: {type(node.op).__name__}")


def _compare(node: ast.Compare, frame: pd.DataFrame) -> pd.Series:
    if len(node.ops) != 1 or len(node.comparators) != 1:
        raise ValueError("formula comparisons must be binary")
    left = _eval_node(node.left, frame)
    right = _eval_node(node.comparators[0], frame)
    if isinstance(node.ops[0], ast.Gt):
        return left > right
    if isinstance(node.ops[0], ast.Lt):
        return left < right
    if isinstance(node.ops[0], ast.GtE):
        return left >= right
    if isinstance(node.ops[0], ast.LtE):
        return left <= right
    raise ValueError(f"unsupported comparison operator: {type(node.ops[0]).__name__}")


def _call(node: ast.Call, frame: pd.DataFrame) -> Any:
    name = _call_name(node)
    args = [_eval_node(arg, frame) for arg in node.args]
    if name == "ATR":
        return _atr(frame, _int_arg(args, name))
    if name == "TsZScore":
        return _ts_zscore(_series_arg(args, name), _int_arg(args[1:], name))
    if name == "Mean":
        return _mean(_series_arg(args, name), _int_arg(args[1:], name))
    if name == "SignedPower":
        return _signed_power(_series_arg(args, name), _float_arg(args[1:], name))
    if name == "Where":
        return _where(args)
    if name == "Slope":
        return _slope(_series_arg(args, name), _int_arg(args[1:], name))
    raise ValueError(f"unsupported formula function: {name}")


def _call_name(node: ast.Call) -> str:
    if not isinstance(node.func, ast.Name):
        raise ValueError("formula functions must be direct names")
    if node.keywords:
        raise ValueError("formula function keywords are not supported")
    return node.func.id


def _atr(frame: pd.DataFrame, period: int) -> pd.Series:
    name = f"atr_{period}"
    if name in frame.columns:
        return _column(frame, name)
    raise ValueError(f"formula ATR column not found: {name}")


def _ts_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std().replace(0.0, np.nan)
    return (series - mean) / std


def _mean(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def _signed_power(series: pd.Series, power: float) -> pd.Series:
    return np.sign(series) * np.power(series.abs(), power)


def _where(args: list[Any]) -> pd.Series:
    if len(args) not in (2, 3) or not isinstance(args[0], pd.Series):
        raise ValueError("Where requires condition, true series, optional false value")
    false_value = args[2] if len(args) == 3 else 0.0
    return args[1].where(args[0].astype(bool), false_value)


def _slope(series: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=float)
    return series.rolling(window).apply(lambda values: float(np.polyfit(x, values, 1)[0]), raw=True)


def _series_arg(args: list[Any], name: str) -> pd.Series:
    if not args or not isinstance(args[0], pd.Series):
        raise ValueError(f"{name} requires a series first argument")
    return args[0]


def _int_arg(args: list[Any], name: str) -> int:
    if not args or isinstance(args[0], pd.Series):
        raise ValueError(f"{name} requires a numeric window argument")
    value = int(args[0])
    if value <= 1:
        raise ValueError(f"{name} window must be greater than 1")
    return value


def _float_arg(args: list[Any], name: str) -> float:
    if not args or isinstance(args[0], pd.Series):
        raise ValueError(f"{name} requires a numeric argument")
    return float(args[0])


def _nonzero(value: Any) -> Any:
    if isinstance(value, pd.Series):
        return value.where(value.abs() > EPSILON, np.nan)
    return value if abs(float(value)) > EPSILON else np.nan
