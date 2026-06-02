from __future__ import annotations

from dataclasses import dataclass

WEB_FACTOR_WINDOWS = (3, 5, 8, 13, 21, 34, 55, 89, 144, 200)


@dataclass(frozen=True)
class WebFactorSpec:
    name: str
    category: str
    description: str
    formula: str
    kind: str
    window: int


@dataclass(frozen=True)
class WebFactorFamily:
    prefix: str
    category: str
    description_template: str
    formula_template: str
    kind: str


WEB_FACTOR_FAMILIES: tuple[WebFactorFamily, ...] = (
    WebFactorFamily("web_ret", "return", "{window}周期收益率", "close.pct_change({window})", "ret_ratio"),
    WebFactorFamily("web_log_ret", "return", "{window}周期对数收益率", "log(close / close.shift({window}))", "log_ret"),
    WebFactorFamily("web_ret_z", "return", "单周期收益率{window}周期Z分数", "zscore(ret_1,{window})", "ret_z"),
    WebFactorFamily("web_ret_rank", "statistic", "单周期收益率{window}周期时序排名", "ts_rank(ret_1,{window})", "ret_rank"),
    WebFactorFamily("web_realized_vol", "volatility", "{window}周期已实现波动率", "std(ret_1,{window})", "realized_vol"),
    WebFactorFamily("web_parkinson_vol", "volatility", "{window}周期Parkinson高低价波动率", "sqrt(mean(log(high/low)^2,{window}) / (4*log(2)))", "parkinson_vol"),
    WebFactorFamily("web_garman_klass_vol", "volatility", "{window}周期Garman-Klass OHLC波动率", "sqrt(mean(0.5*log(high/low)^2-(2*log(2)-1)*log(close/open)^2,{window}))", "garman_klass_vol"),
    WebFactorFamily("web_atr_norm", "volatility", "{window}周期标准化真实波幅", "mean(true_range,{window}) / close", "atr_norm"),
    WebFactorFamily("web_range_z", "volatility", "高低价振幅{window}周期Z分数", "zscore((high-low)/close,{window})", "range_z"),
    WebFactorFamily("web_close_pos", "structure", "{window}周期高低区间收盘位置", "(close-low_min)/(high_max-low_min)", "close_pos"),
    WebFactorFamily("web_sma_ratio", "moving_average", "收盘价相对{window}周期SMA偏离", "close / sma(close,{window}) - 1", "sma_ratio"),
    WebFactorFamily("web_ema_ratio", "moving_average", "收盘价相对{window}周期EMA偏离", "close / ema(close,{window}) - 1", "ema_ratio"),
    WebFactorFamily("web_sma_slope", "moving_average", "{window}周期SMA三周期斜率", "sma(close,{window}).pct_change(3)", "sma_slope"),
    WebFactorFamily("web_roc_smooth", "momentum", "{window}周期ROC平滑动量", "mean(close.pct_change({window}),{window})", "roc_smooth"),
    WebFactorFamily("web_volume_z", "volume", "成交量{window}周期Z分数", "zscore(volume,{window})", "volume_z"),
    WebFactorFamily("web_volume_ratio", "volume", "成交量相对{window}周期均量", "volume / mean(volume,{window})", "volume_ratio"),
    WebFactorFamily("web_dollar_volume_z", "volume", "成交额{window}周期Z分数", "zscore(close*volume,{window})", "dollar_volume_z"),
    WebFactorFamily("web_obv_slope", "volume", "OBV {window}周期斜率", "obv.diff({window}) / sum(volume,{window})", "obv_slope"),
    WebFactorFamily("web_price_volume_corr", "statistic", "收益率与成交量变化{window}周期相关", "corr(ret_1, volume.pct_change(), {window})", "price_volume_corr"),
    WebFactorFamily("web_vwap_dev", "structure", "收盘价相对{window}周期滚动VWAP偏离", "close / rolling_vwap({window}) - 1", "vwap_dev"),
)


def _specs() -> tuple[WebFactorSpec, ...]:
    return tuple(
        WebFactorSpec(
            name=f"{family.prefix}_{window}",
            category=family.category,
            description=family.description_template.format(window=window),
            formula=family.formula_template.format(window=window),
            kind=family.kind,
            window=window,
        )
        for family in WEB_FACTOR_FAMILIES
        for window in WEB_FACTOR_WINDOWS
    )


WEB_FACTOR_SPECS: tuple[WebFactorSpec, ...] = _specs()
WEB_FACTOR_COLUMNS: tuple[str, ...] = tuple(spec.name for spec in WEB_FACTOR_SPECS)
WEB_FACTOR_COUNT = len(WEB_FACTOR_SPECS)
