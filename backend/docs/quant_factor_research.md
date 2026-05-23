# 量化因子研究笔记

本文档把股票与币圈都适用的因子知识，收敛成当前项目可计算、可回测、可审计的落地范围。

## 资料基线

- [101 Formulaic Alphas](https://hedgefundalpha.com/101-formulaic-alphas/)：大量日频 price-volume alpha 使用 open/high/low/close/volume/vwap 的秩、相关、延迟、滚动统计组合，核心启发是不要迷信单一因子，要看低相关组合和样本外稳定性。
- [TA-Lib 官方指标分组](https://ta-lib.github.io/ta-lib-python/funcs.html)：技术指标可按 overlap、momentum、volume、volatility、price transform、statistic 等族组织。
- [ML for Trading: Alpha Factor Research](https://ml4trading.io/second-edition/chapter/4/)：因子本质是把市场、基本面或另类数据转成预测信号，评估重点包括 IC、分位数组合收益和换手率。
- [Alphalens 因子分析示例](https://aaiken1.github.io/fin-data-analysis-text/chapters/18_1_alphalens.html)：单因子验证应看 forward returns、quantile returns、Spearman IC、IC 稳定性和 turnover。
- [pandas rolling 文档](https://pandas.pydata.org/docs/dev/reference/api/pandas.DataFrame.rolling.html) 与 [shift 文档](https://pandas.pydata.org/docs/dev/reference/api/pandas.DataFrame.shift.html)：项目内滚动窗口只使用历史窗口，基础 K 线因子统一 `shift(1)` 后进入回测，避免当前未收完 K 线污染信号。
- [Purged walk-forward 研究框架](https://arxiv.org/html/2603.09219v1)：过拟合控制应按时间顺序做 IS/WFA/OOS，窗口边界使用 purge gap，参数选择不能回看 OOS。

## 因子族落地

当前项目优先落地不依赖外部财报或链上数据的 OHLCV 因子：

- 收益率：简单收益、对数收益、短长动量差、收益率 Z 分数。
- 波动率：滚动标准差、上下行波动率、波动率的波动率、偏度、峰度、ATR、布林带宽、振幅均值和振幅 Z 分数。
- 均线趋势：SMA/EMA 偏离、EMA 交叉、均线斜率。
- 动量振荡：RSI、MACD、ROC、PPO、随机指标、Williams %R、CCI、效率比率。
- 成交量：成交量均值、成交量比率、成交量 Z 分数、成交额均值、MFI、CMF、OBV 斜率。
- K 线结构：上下影线、影线不平衡、实体占振幅、跳空、Donchian 位置、距近端高低点、VWAP 偏离。
- 扩展数据：多时间框架、订单簿、资金费率、合约持仓量、多空账户比、主动买卖量、情绪指数由增强特征帧提供；缺失外部数据时回测报告会显式记录失败或未覆盖，不伪造成功。

新增真实非 K 线来源：

- Binance USD-M Futures Open Interest Statistics：`/futures/data/openInterestHist`，用于持仓量、持仓价值、持仓量 Z 分数。
- Binance USD-M Futures Long/Short Ratio：`/futures/data/globalLongShortAccountRatio`，用于多空账户比、长短账户占比。
- Binance USD-M Futures Taker Buy/Sell Volume：`/futures/data/takerlongshortRatio`，用于主动买卖量比、主动买入占比。
- Alternative.me Fear & Greed Index：`/fng/`，用于币圈情绪因子。

链上外部数据（`onchain_features`）由 CoinMetrics Community API 按日采集：`exchange_netflow`（交易所净流入 = FlowInExNtv - FlowOutExNtv）、`active_addresses`、`transaction_count`、`stablecoin_supply_ratio`（USDT+USDC 流通量 / 标的市值）。采集入口为 `market_context_ingest_service.ingest_market_context_data`，与持仓/情绪数据一同由后台 `market_context_refresh_loop` 周期刷新。当前支持 `BTCUSDT`、`ETHUSDT`；无真实链上数据入库前不生成空值假信号。

## 防未来函数规则

- 基础 OHLCV 因子在 `build_feature_frame` 中统一计算后 `shift(1)`。
- 回测收益只作为标签：`fwd_ret = close.pct_change(horizon).shift(-horizon)`，不参与因子计算。
- 多时间框架特征只使用已完成历史桶和当前桶运行中状态，不读取未来桶收盘、未来高低点或未来成交量总额。
- 测试会验证注册的基础因子都在 `FEATURE_COLUMNS` 内，避免新增因子漏掉统一 shift。
- 测试会突变当前 K 线的 OHLCV，并断言同一时点的已 shift 因子不变。

## 过拟合控制

- 单因子排序只作为筛选，不等于上线策略。
- 同一因子必须同时查看 10m、30m、60m、1d，避免只挑表现最好的单周期。
- 关注 IC 均值、IR、IC 正值率、分位收益、long-short return、turnover 和 p-value。
- 参数、阈值、入场规则应在训练窗口或 walk-forward 验证中确定，最终 OOS 不再调参。
- 对多因子或策略组合，优先使用已有 walk-forward + purge 机制，而不是全样本网格搜索。

## 因子学习记忆

项目内新增的因子学习层借鉴 FactorMiner 的“成功模式 + 禁区”思想，但落地时只读取本项目真实回测、组合排名和已结算预测：

- 因子挖掘记忆：从多因子组合缓存中提炼高胜率、高盈亏比、高夏普或高 IR 的类别/算子模式，并用 Spearman 相关性记录冗余因子邻域。
- 挖掘因子库：组合刷新会把同时满足胜率与盈亏比阈值的好组合写入 `backend/models/factor_learning/mined_factor_library.json`，下一次组合搜索会把这些已验证组合重新作为候选因子参与排序。
- 亏损模式记忆：自动结算到期预测后，把真实亏损交易与入场时因子截面重新对齐，学习“亏损中位数明显偏高/偏低且亏损率抬升”的特征阈值。
- 多重过滤：实时组合信号先过历史胜率和入场窗口，再检查成员确认数和亏损特征命中数；命中亏损模式会显式阻断，不静默降级。
- 自动权重：每天北京时间 00:30 刷新前一日组合排名与复盘记忆，按真实指标重算成员权重，并对已进入亏损模式的成员降权，记忆写入 `backend/models/factor_learning/`。
- 监控诊断：因子学习记忆会记录多因子组合实盘模拟的结算样本、预测成功率、候选成功率、连续亏损和对应解决方案。
- 运算符库：`factor_operator_library.py` 维护 60+ 个挖掘算子，覆盖算术、时序统计、截面、差分、相关、平滑、回归、逻辑条件、风险形态、量价、微观结构和衍生品状态。页面通过 `/api/factor-learning/operators` 展示，Agent prompt 也使用同一份结构化库。
- 联网 LLM Agent：刷新因子学习时会调用 SiliconFlow Chat Completions。`.env` 需要配置 `SILICONFLOW_API_KEY`；用 `SILICONFLOW_MODEL` 指定实际调用的模型（例如 `deepseek-ai/DeepSeek-V3.2`）。未配置时代码默认 `Pro/moonshotai/Kimi-K2.6`。可选覆盖 `SILICONFLOW_CHAT_COMPLETIONS_URL` 和 `SILICONFLOW_TIMEOUT_SECONDS`（默认 180 秒）。页面「模型」展示 `llmAgent.model` 或当前 `SILICONFLOW_MODEL`，与内部 agent 标识 `siliconflow_factor_agent_v1` 无关。Agent 输出只作为候选研究计划写入 `llmAgent.review`，不会伪造成已验证因子。
