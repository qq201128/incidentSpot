from __future__ import annotations

import ast
from typing import Any

import numpy as np
import pandas as pd

from app.services.agent_formula_helpers import (
    _float_arg,
    _int_arg,
    _nonzero,
    _series_lag_args,
    _series_lag_window_args,
    _series_series_window_args,
    _series_window_args,
    _signed_power,
    _ts_zscore,
    _vwap,
    _window_arg,
)

EPSILON = 1e-12
SUPPORTED_AGENT_FORMULA_FUNCTIONS = frozenset(
    {
        "ADX", "ATR", "Abs", "Acceleration", "Add", "AutoCorr", "Clip", "Corr",
        "Delay", "Delta", "Div", "DonchianPos", "EMA", "EWMStd", "Equal",
        "FundingZ", "Greater", "GreaterEqual", "IfElse", "Less", "LessEqual",
        "Log", "LongShortRatioZ", "Max", "Mean", "Min", "Mul", "Neg",
        "OpenInterestZ", "PctChange", "SMA", "Sign", "SignedPower", "Slope",
        "Std", "Sub", "Sum", "TrueRange", "TsQuantile", "TsRank", "TsZScore",
        "VWAP", "VWAPDev", "Where",
    }
)
COLUMN_ALIASES = {
    "MicropriceBps": "microprice_bps",
    "OFIRatio": "ofi_ratio",
    "OrderbookImbalance": "orderbook_imbalance",
    "SpreadBps": "spread_bps",
}


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
    if isinstance(node.op, ast.Pow | ast.BitXor):
        return np.power(left, right)
    raise ValueError(f"unsupported formula operator: {type(node.op).__name__}")


def _compare(node: ast.Compare, frame: pd.DataFrame) -> pd.Series:
    if len(node.ops) != 1 or len(node.comparators) != 1:
        raise ValueError("formula comparisons must be binary")
    return _compare_values(_eval_node(node.left, frame), _eval_node(node.comparators[0], frame), node.ops[0])


def _call(node: ast.Call, frame: pd.DataFrame) -> Any:
    name = _call_name(node)
    args = [_eval_node(arg, frame) for arg in node.args]
    if name in {"ATR", "ADX"}:
        return _mapped_column(frame, name.lower(), _int_arg(args, name))
    if name in {"FundingZ", "OpenInterestZ", "LongShortRatioZ"}:
        return _zscore_mapped_column(frame, name, _int_arg(args, name))
    if name in {"Abs", "Sign", "Log", "Clip", "SignedPower", "Neg"}:
        return _arithmetic_call(name, args)
    if name in {"Add", "Sub", "Mul", "Div"}:
        return _binary_call(name, args)
    if name in {"Greater", "Less", "GreaterEqual", "LessEqual", "Equal"}:
        return _comparison_call(name, args)
    if name in {"TsZScore", "TsRank", "Mean", "Std", "Sum", "Min", "Max", "SMA", "EMA", "EWMStd"}:
        return _window_call(name, args)
    if name in {"Delay", "PctChange", "Delta", "Acceleration"}:
        return _lag_call(name, args)
    if name in {"Corr", "AutoCorr", "VWAP", "VWAPDev", "DonchianPos", "TsQuantile"}:
        return _multi_series_call(name, args)
    if name in {"Where", "IfElse"}:
        return _where(args)
    if name == "Slope":
        return _slope(_series_arg(args, name), _window_arg(args[1:], name))
    if name == "TrueRange":
        return _true_range(frame, args)
    raise ValueError(f"unsupported formula function: {name}")


def _arithmetic_call(name: str, args: list[Any]) -> pd.Series:
    series = _series_arg(args, name)
    if name == "Abs":
        return series.abs()
    if name == "Sign":
        return pd.Series(np.sign(series), index=series.index)
    if name == "Log":
        return np.log(series.where(series > 0.0, np.nan))
    if name == "Neg":
        return -series
    if name == "Clip":
        return series.clip(_float_arg(args[1:2], name), _float_arg(args[2:3], name))
    return _signed_power(series, _float_arg(args[1:], name))


def _window_call(name: str, args: list[Any]) -> pd.Series:
    series, window = _series_window_args(args, name)
    if name == "TsZScore":
        return _ts_zscore(series, window)
    if name == "TsRank":
        return series.rolling(window).rank(pct=True)
    if name in {"Mean", "SMA"}:
        return series.rolling(window).mean()
    if name == "Std":
        return series.rolling(window).std()
    if name == "EWMStd":
        return series.ewm(span=window, adjust=False, min_periods=window).std()
    if name == "EMA":
        return series.ewm(span=window, adjust=False, min_periods=window).mean()
    return _rolling_extreme(name, series, window)


def _lag_call(name: str, args: list[Any]) -> pd.Series:
    series, window = _series_lag_args(args, name)
    if name == "Delay":
        return series.shift(window)
    if name == "PctChange":
        return series / _nonzero(series.shift(window)) - 1.0
    if name == "Delta":
        return series - series.shift(window)
    return series.diff(window).diff(window)


def _multi_series_call(name: str, args: list[Any]) -> pd.Series:
    if name == "AutoCorr":
        series, lag, window = _series_lag_window_args(args, name)
        return series.rolling(window).corr(series.shift(lag))
    if name in {"DonchianPos", "TsQuantile"}:
        series, window = _series_window_args(args, name)
        return _donchian_pos(series, window) if name == "DonchianPos" else _ts_quantile(series, window)
    first, second, window = _series_series_window_args(args, name)
    if name == "Corr":
        return first.rolling(window).corr(second)
    if name == "VWAP":
        return _vwap(first, second, window)
    if name == "VWAPDev":
        return first / _nonzero(_vwap(first, second, window)) - 1.0
    raise ValueError(f"unsupported formula function: {name}")


def _binary_call(name: str, args: list[Any]) -> Any:
    left, right = _two_args(args, name)
    if name == "Add":
        return left + right
    if name == "Sub":
        return left - right
    if name == "Mul":
        return left * right
    return left / _nonzero(right)


def _comparison_call(name: str, args: list[Any]) -> pd.Series:
    left, right = _two_args(args, name)
    op = {"Greater": ast.Gt(), "Less": ast.Lt(), "GreaterEqual": ast.GtE(), "LessEqual": ast.LtE(), "Equal": ast.Eq()}[name]
    return _compare_values(left, right, op)


def _compare_values(left: Any, right: Any, op: ast.cmpop) -> pd.Series:
    if isinstance(op, ast.Gt):
        return left > right
    if isinstance(op, ast.Lt):
        return left < right
    if isinstance(op, ast.GtE):
        return left >= right
    if isinstance(op, ast.LtE):
        return left <= right
    if isinstance(op, ast.Eq):
        return left == right
    raise ValueError(f"unsupported comparison operator: {type(op).__name__}")


def _rolling_extreme(name: str, series: pd.Series, window: int) -> pd.Series:
    if name == "Sum":
        return series.rolling(window).sum()
    if name == "Min":
        return series.rolling(window).min()
    if name == "Max":
        return series.rolling(window).max()
    raise ValueError(f"unsupported formula function: {name}")


def _where(args: list[Any]) -> pd.Series:
    if len(args) not in (2, 3) or not isinstance(args[0], pd.Series):
        raise ValueError("Where requires condition, true series, optional false value")
    false_value = args[2] if len(args) == 3 else 0.0
    return args[1].where(args[0].astype(bool), false_value)


def _true_range(frame: pd.DataFrame, args: list[Any]) -> pd.Series:
    high, low, close = _true_range_args(frame, args)
    previous_close = close.shift(1)
    ranges = pd.concat([high - low, (high - previous_close).abs(), (low - previous_close).abs()], axis=1)
    return ranges.max(axis=1)


def _true_range_args(frame: pd.DataFrame, args: list[Any]) -> tuple[pd.Series, pd.Series, pd.Series]:
    if not args:
        return _column(frame, "high"), _column(frame, "low"), _column(frame, "close")
    if len(args) == 3 and all(isinstance(arg, pd.Series) for arg in args):
        return args[0], args[1], args[2]
    raise ValueError("TrueRange requires no arguments or high, low, close series")

def _donchian_pos(series: pd.Series, window: int) -> pd.Series:
    low = series.rolling(window).min()
    high = series.rolling(window).max()
    return (series - low) / _nonzero(high - low)

def _ts_quantile(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).rank(pct=True)

def _slope(series: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=float)
    return series.rolling(window).apply(lambda values: float(np.polyfit(x, values, 1)[0]), raw=True)


def _call_name(node: ast.Call) -> str:
    if not isinstance(node.func, ast.Name):
        raise ValueError("formula functions must be direct names")
    if node.keywords:
        raise ValueError("formula function keywords are not supported")
    return node.func.id


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    name = COLUMN_ALIASES.get(name, name)
    if name not in frame.columns:
        raise ValueError(f"formula column not found: {name}")
    return pd.to_numeric(frame[name], errors="coerce")


def _mapped_column(frame: pd.DataFrame, prefix: str, period: int) -> pd.Series:
    return _column(frame, f"{prefix}_{period}")


def _zscore_mapped_column(frame: pd.DataFrame, name: str, window: int) -> pd.Series:
    columns = {"FundingZ": "funding_rate", "OpenInterestZ": "open_interest", "LongShortRatioZ": "long_short_ratio"}
    return _ts_zscore(_column(frame, columns[name]), window)


def _constant(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("formula constants must be numeric")
    return float(value)


def _unary(node: ast.UnaryOp, frame: pd.DataFrame) -> Any:
    value = _eval_node(node.operand, frame)
    if isinstance(node.op, ast.USub):
        return -value
    if isinstance(node.op, ast.UAdd):
        return value
    raise ValueError(f"unsupported unary operator: {type(node.op).__name__}")


def _series_arg(args: list[Any], name: str) -> pd.Series:
    if not args or not isinstance(args[0], pd.Series):
        raise ValueError(f"{name} requires a series first argument")
    return args[0]


def _two_args(args: list[Any], name: str) -> tuple[Any, Any]:
    if len(args) != 2:
        raise ValueError(f"{name} requires two arguments")
    return args[0], args[1]
