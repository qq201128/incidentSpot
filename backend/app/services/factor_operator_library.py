from __future__ import annotations

from collections import Counter
from typing import Any

from app.services.agent_factor_formula import SUPPORTED_AGENT_FORMULA_FUNCTIONS

OPERATOR_LIBRARY_VERSION = "factor_operator_library_v1"
EXECUTABLE_OPERATOR_NAMES = SUPPORTED_AGENT_FORMULA_FUNCTIONS
WINDOW_CONSTRAINT = "rolling window argument must be an integer greater than 1"
LAG_CONSTRAINT = "lag argument must be a positive integer"
AGENT_FORMULA_RULES = (
    "formulaHint must use only executable operator names listed in operator_library.operators",
    "rolling window arguments must be integer values greater than 1",
    "lag arguments for Delay, Delta, PctChange, Acceleration, AutoCorr may be 1",
    "arithmetic and comparisons may use functions or symbols: Add/Sub/Mul/Div, >, <, >=, <=, ==",
)
WINDOWED_OPERATORS = frozenset(
    {
        "ATR",
        "ADX",
        "Acceleration",
        "AutoCorr",
        "Corr",
        "Delay",
        "Delta",
        "DonchianPos",
        "EMA",
        "EWMStd",
        "FundingZ",
        "LongShortRatioZ",
        "Max",
        "Mean",
        "Min",
        "OpenInterestZ",
        "PctChange",
        "SMA",
        "Slope",
        "Std",
        "Sum",
        "TsQuantile",
        "TsRank",
        "TsZScore",
        "VWAP",
        "VWAPDev",
    }
)

_ROWS: tuple[tuple[str, str, str, str, str], ...] = (
    ("Add", "arithmetic", "x + y", "两个信号线性叠加", "Add(ret_5, vol_z_20)"),
    ("Sub", "arithmetic", "x - y", "差值与偏离", "Sub(close, vwap_20)"),
    ("Mul", "arithmetic", "x * y", "交互项与调制", "Mul(ret_5, vol_ratio_20)"),
    ("Div", "arithmetic", "x / y", "比率与效率", "Div(ret_10, amount_ma_20)"),
    ("Abs", "arithmetic", "abs(x)", "幅度提取", "Abs(ret_1)"),
    ("Neg", "arithmetic", "-x", "方向反转", "Neg(TsRank(ret_5, 24))"),
    ("Log", "arithmetic", "log(x)", "压缩长尾", "Log(volume)"),
    ("Sign", "arithmetic", "sign(x)", "方向离散化", "Sign(ret_3)"),
    ("SignedPower", "arithmetic", "sign(x) * abs(x)^p", "非线性放大", "SignedPower(ret_5, 0.5)"),
    ("Clip", "arithmetic", "clip(x, low, high)", "极值裁剪", "Clip(zscore, -4, 4)"),
    ("Mean", "time_series", "rolling_mean(x, n)", "历史均值", "Mean(ret_1, 20)"),
    ("Std", "time_series", "rolling_std(x, n)", "历史波动率", "Std(ret_1, 60)"),
    ("Median", "time_series", "rolling_median(x, n)", "抗极值中位数", "Median(volume, 20)"),
    ("Skew", "time_series", "rolling_skew(x, n)", "偏度/尾部方向", "Skew(ret_1, 60)"),
    ("Kurt", "time_series", "rolling_kurt(x, n)", "峰度/尾部厚度", "Kurt(ret_1, 60)"),
    ("Sum", "time_series", "rolling_sum(x, n)", "累积量", "Sum(volume, 20)"),
    ("Min", "time_series", "rolling_min(x, n)", "窗口低点", "Min(low, 20)"),
    ("Max", "time_series", "rolling_max(x, n)", "窗口高点", "Max(high, 20)"),
    ("ArgMin", "time_series", "argmin(x, n)", "低点位置", "ArgMin(low, 60)"),
    ("ArgMax", "time_series", "argmax(x, n)", "高点位置", "ArgMax(high, 60)"),
    ("TsRank", "time_series", "rank x within trailing n", "时间序列排名", "TsRank(ret_5, 24)"),
    ("TsZScore", "time_series", "(x - Mean(x,n)) / Std(x,n)", "历史标准分", "TsZScore(volume, 60)"),
    ("TsQuantile", "time_series", "quantile_rank(x,n)", "历史分位", "TsQuantile(spread_bps, 120)"),
    ("Delay", "time_series", "x.shift(n)", "历史延迟", "Delay(ret_1, 5)"),
    ("DecayLinear", "time_series", "linear_decay(x,n)", "近端加权衰减", "DecayLinear(ret_1, 12)"),
    ("Delta", "difference", "x - Delay(x,n)", "变化量", "Delta(close, 6)"),
    ("PctChange", "difference", "x / Delay(x,n) - 1", "变化率", "PctChange(close, 10)"),
    ("DiffRank", "difference", "TsRank(Delta(x,n),m)", "变化排名", "DiffRank(volume, 3, 20)"),
    ("Acceleration", "difference", "Delta(Delta(x,n),n)", "二阶变化", "Acceleration(ret_5, 3)"),
    ("CsRank", "cross_section", "cross-sectional rank", "截面排名", "CsRank(ret_5)"),
    ("CsZScore", "cross_section", "cross-sectional zscore", "截面标准分", "CsZScore(volume_ratio)"),
    ("CsNeutralize", "cross_section", "neutralize by group", "行业/组中性化", "CsNeutralize(ret_5, sector)"),
    ("CsWinsorize", "cross_section", "winsorize by cross-section", "截面去极值", "CsWinsorize(factor, 0.01)"),
    ("Corr", "correlation", "rolling_corr(x,y,n)", "滚动相关", "Corr(ret_1, volume, 40)"),
    ("Cov", "correlation", "rolling_cov(x,y,n)", "滚动协方差", "Cov(close, volume, 40)"),
    ("Beta", "correlation", "Cov(x,y,n) / Var(y,n)", "相对 beta", "Beta(ret_1, market_ret, 60)"),
    ("AutoCorr", "correlation", "corr(x, Delay(x,k), n)", "自相关", "AutoCorr(ret_1, 1, 40)"),
    ("SMA", "smoothing", "simple moving average", "简单平滑", "SMA(close, 20)"),
    ("EMA", "smoothing", "exponential moving average", "指数平滑", "EMA(close, 20)"),
    ("WMA", "smoothing", "weighted moving average", "线性加权平滑", "WMA(close, 20)"),
    ("EWMStd", "smoothing", "ewm std", "指数波动率", "EWMStd(ret_1, 40)"),
    ("Slope", "regression", "rolling linear slope", "趋势斜率", "Slope(close, 60)"),
    ("Rsquare", "regression", "rolling regression R2", "趋势解释度", "Rsquare(close, 60)"),
    ("Resi", "regression", "rolling regression residual", "偏离趋势残差", "Resi(close, 60)"),
    ("TStat", "regression", "rolling regression t-stat", "趋势显著性", "TStat(close, 60)"),
    ("IfElse", "logical", "condition ? x : y", "状态切换", "IfElse(Rsquare(close,60)>0.5, trend, reversal)"),
    ("Greater", "logical", "x > y", "阈值判断", "Greater(vol_z_20, 2)"),
    ("Less", "logical", "x < y", "阈值判断", "Less(spread_bps, 5)"),
    ("And", "logical", "a and b", "多条件共振", "And(high_vol, trend_up)"),
    ("Or", "logical", "a or b", "条件放宽", "Or(sweep_high, fvg_down)"),
    ("Not", "logical", "not a", "条件取反", "Not(high_spread)"),
    ("Where", "logical", "mask x else nan", "显式过滤", "Where(quality_passed, factor)"),
    ("ATR", "risk_shape", "average true range", "真实波幅", "ATR(14) / close"),
    ("TrueRange", "risk_shape", "max high-low gaps", "跳空波幅", "TrueRange(high, low, close)"),
    ("Drawdown", "risk_shape", "x / rolling_max(x,n)-1", "回撤状态", "Drawdown(close, 120)"),
    ("DonchianPos", "risk_shape", "(close-Min(low,n))/(Max(high,n)-Min(low,n))", "通道位置", "DonchianPos(close, 60)"),
    ("VWAP", "volume_price", "sum(price*volume,n)/sum(volume,n)", "成交量加权均价", "VWAP(close, volume, 20)"),
    ("VWAPDev", "volume_price", "close / VWAP(n) - 1", "VWAP 偏离", "VWAPDev(close, volume, 60)"),
    ("OBV", "volume_price", "on balance volume", "量价累积", "Slope(OBV, 60)"),
    ("MFI", "volume_price", "money flow index", "资金流强弱", "MFI(14)"),
    ("CMF", "volume_price", "chaikin money flow", "收盘位置加权资金流", "CMF(20)"),
    ("AmountEfficiency", "volume_price", "returns / amount", "成交效率", "Div(ret_10, amount_ma_20)"),
    ("OrderbookImbalance", "microstructure", "(bid_qty-ask_qty)/total", "盘口不平衡", "OrderbookImbalance"),
    ("SpreadBps", "microstructure", "spread / mid * 10000", "价差成本", "SpreadBps"),
    ("MicropriceBps", "microstructure", "microprice deviation", "微价格偏离", "MicropriceBps"),
    ("OFIRatio", "microstructure", "order flow imbalance ratio", "订单流失衡", "OFIRatio"),
    ("FundingZ", "derivatives", "funding zscore", "资金费率异常", "FundingZ(20)"),
    ("OpenInterestZ", "derivatives", "open interest zscore", "持仓异常", "OpenInterestZ(60)"),
    ("LongShortRatioZ", "derivatives", "long short ratio zscore", "账户多空拥挤", "LongShortRatioZ(60)"),
)


def factor_operator_payload() -> dict[str, Any]:
    operators = [_operator_payload(row) for row in _ROWS]
    category_counts = Counter(item["category"] for item in operators)
    return {
        "version": OPERATOR_LIBRARY_VERSION,
        "total": len(operators),
        "categories": [
            {"key": key, "count": category_counts[key]}
            for key in sorted(category_counts)
        ],
        "operators": operators,
    }


def factor_operator_prompt_payload() -> dict[str, Any]:
    payload = factor_operator_payload()
    operators = [item for item in payload["operators"] if item["name"] in EXECUTABLE_OPERATOR_NAMES]
    category_counts = Counter(item["category"] for item in operators)
    return {
        "version": payload["version"],
        "total": len(operators),
        "categories": [
            {"key": key, "count": category_counts[key]}
            for key in sorted(category_counts)
        ],
        "operators": [
            {
                "name": item["name"],
                "category": item["category"],
                "signature": item["signature"],
                "example": item["example"],
                "constraints": _operator_constraints(item["name"]),
            }
            for item in operators
        ],
        "formulaRules": list(AGENT_FORMULA_RULES),
    }


def factor_operator_summary() -> dict[str, Any]:
    payload = factor_operator_payload()
    return {
        "version": payload["version"],
        "total": payload["total"],
        "categories": payload["categories"],
    }


def _operator_payload(row: tuple[str, str, str, str, str]) -> dict[str, Any]:
    name, category, signature, purpose, example = row
    return {
        "name": name,
        "category": category,
        "signature": signature,
        "purpose": purpose,
        "example": example,
        "executable": name in EXECUTABLE_OPERATOR_NAMES,
    }


def _operator_constraints(name: str) -> list[str]:
    constraints = []
    if name in WINDOWED_OPERATORS:
        constraints.append(WINDOW_CONSTRAINT)
    if name in {"Delay", "Delta", "PctChange", "Acceleration", "AutoCorr"}:
        constraints.append(LAG_CONSTRAINT)
    return constraints
