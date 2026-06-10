# 事件合约方向预测系统重构汇总

更新时间：2026-06-03

## 目标

本次重构将项目目标从旧的规则胜率/组合缓存驱动，调整为面向 Binance 事件合约的方向预测系统。

核心目标是：在指定事件周期内判断价格方向，输出 `UP / DOWN / SKIP`，并先接管模拟事件合约验证，真实交易保持关闭。

```text
原始行情与外部因子数据
-> 数据完整性修复
-> 环境标签生成
-> 模型族重新训练
-> 最终决定层
-> UP / DOWN / SKIP
-> 模拟事件合约验证
```

## 设计原则

- 旧预测、旧事件、旧因子缓存、旧模型产物视为污染样本。
- 保留 K 线和外部因子等基础原始数据。
- K 线缺口和异常只允许通过 Binance 重新拉取修复。
- 不做插值，不生成合成 K 线，不静默补全。
- 训练和实时预测使用同一套环境识别逻辑。
- 第一版最终决定层只用于模拟验证，不启用真实交易。

## 数据库新增

### `market_data_quality_reports`

用于记录行情质量扫描与修复结果。

主要用途：

- 记录 K 线缺口。
- 记录重复 `open_time`。
- 记录 OHLC 合法性异常。
- 记录价格跳变异常。
- 记录 Binance 重拉区间。
- 记录修复状态和失败原因。

### `event_market_regimes`

用于保存事件入场前的市场环境标签。

核心字段：

- `symbol`
- `duration`
- `open_time`
- `trend_state`
- `volatility_state`
- `regime_label`
- `confidence`
- `reason_codes`
- `metrics_json`

### `event_final_decisions`

用于保存最终裁判层的审计记录。

核心字段：

- `symbol`
- `duration`
- `open_time`
- `decision`
- `direction`
- `probability_up`
- `confidence`
- `final_score`
- `regime_label`
- `candidate_count`
- `reason_codes`
- `settled_at`
- `decision_correct`

## 新增后端模块

### `reset_derived_research_data.py`

新增一次性清理脚本。

位置：

- `backend/scripts/reset_derived_research_data.py`

行为：

- 默认 dry-run，只输出将清理的表、目录和行数。
- 必须显式传入 `--confirm` 才执行删除。
- 保留 schema，不重建数据库文件。
- 不删除原始行情和外部因子基础数据。

清理范围：

- `predictions`
- `events`
- `orders`
- `settlements`
- `ensemble_*`
- `paper_live_*`
- `factor_*_cache`
- `high_winrate_*`
- `event_market_regimes`
- `event_final_decisions`
- `auto_trade_strategies`
- 旧模型 artifact 目录

### `market_data_repair_service.py`

新增行情质量修复服务。

位置：

- `backend/app/services/market_data_repair_service.py`

能力：

- 扫描 `symbol + interval` 的 K 线连续性。
- 识别缺口、重复 K 线、非法 OHLC、异常价格跳变。
- 对缺口和异常窗口统一从 Binance 重新拉取。
- 修复失败时写质量报告并抛出明确错误。
- 不使用插值，不生成假数据。

### `event_regime_detector.py`

新增市场环境识别服务。

位置：

- `backend/app/services/event_regime_detector.py`

输出：

- 趋势状态：`trend_up / trend_down / range / uncertain`
- 波动状态：`high_vol / normal_vol / low_vol`
- 综合标签：`regime_label`
- 环境置信度：`confidence`
- 解释代码：`reason_codes`
- 指标详情：`metrics_json`

关键约束：

- 只使用入场前已经完成的 K 线。
- 数据不足时返回 `insufficient_regime_data`。
- 训练集和实时预测共用同一套环境特征生成逻辑。

### `event_final_decision_service.py`

新增最终决定层。

位置：

- `backend/app/services/event_final_decision_service.py`

策略键：

- `event_final_decision_v1`

输出：

- `UP`
- `DOWN`
- `SKIP`

决策逻辑：

- 聚合同一事件窗口内的模型族 shadow 输出。
- 根据模型概率、方向一致性、环境置信度、候选数量生成最终评分。
- 候选不足时输出 `SKIP`。
- 环境数据不足时输出 `SKIP`。
- 环境不确定或高风险时提高通过门槛。
- `UP / DOWN` 写入 `predictions`，进入现有模拟事件链路。
- `SKIP` 只写入 `event_final_decisions`，不创建预测，不创建模拟事件单。

### `event_final_decision_reporting.py`

新增最终裁判统计服务。

位置：

- `backend/app/services/event_final_decision_reporting.py`

统计内容：

- 总命中率。
- 按周期命中率。
- 按环境命中率。
- `UP / DOWN / SKIP` 分布。
- 高置信度分桶真实命中率。

### `model_family_regime_reports.py`

新增模型族按环境分组报告模块。

位置：

- `backend/app/services/model_family_regime_reports.py`

用途：

- 从模型训练验证结果中生成按市场环境分组的表现报告。
- 降低 `model_family_training_impl.py` 文件复杂度。

## API 新增

新增最终裁判报告 API：

```http
GET /api/event-final-decisions/summary?symbol=BTCUSDT&duration=10m
```

实现位置：

- `backend/app/api/event_final_decision.py`

路由注册：

- `backend/app/app_startup.py`

## 现有后端修改

### 自动预测链路

修改文件：

- `backend/app/services/auto_predict_service.py`

变化：

- `event_final_decision_v1` 接入自动预测。
- 允许最终裁判返回 `None` 表示 `SKIP`。
- `SKIP` 不保存到 `predictions`。
- 避免同一个事件窗口对 `SKIP` 重复评估。
- 只启用最终裁判时，也会先收集模型族 shadow 预测。

### 结算链路

修改文件：

- `backend/app/services/forward_validation_service.py`

变化：

- 结算普通 prediction 时同步结算最终裁判审计记录。
- 返回结果增加 `finalDecisionsSettled`。

### 策略注册

修改文件：

- `backend/app/services/strategy_registry.py`

新增策略：

- key：`event_final_decision_v1`
- 名称：`事件最终裁判模拟`
- 支持周期：`10m / 30m / 60m / 1d`
- `live_trading_enabled=False`

### LSTM / 模型族特征

修改文件：

- `backend/app/services/lstm_feature_builder.py`
- `backend/app/services/model_family_training_impl.py`
- `backend/app/services/model_family_prediction_service.py`
- `backend/app/services/model_family_status_service.py`

变化：

- 移除 `sim_feedback_*` 特征。
- 移除依赖旧 `factor_combo_*` 历史表现缓存的特征。
- 加入环境标签特征。
- 训练报告增加按环境分组的验证结果。
- 旧 artifact 如果不包含新版 `regime_*` 特征，会被阻止继续预测。
- 状态接口会给出 `clean_event_retrain_required`，提示需要按新版事件环境特征重训。

## 前端修改

### 策略标签

修改文件：

- `frontend/src/utils/strategyLabels.js`

新增标签：

- `event_final_decision_v1`：`事件最终裁判模拟`

### 删除不可达旧 UI

已删除不再被当前路由和入口引用的旧 UI 文件：

- `frontend/src/components/EventList.jsx`
- `frontend/src/components/FactorLearningPanel.jsx`
- `frontend/src/components/FactorAdaptiveLearningPanel.jsx`
- `frontend/src/components/FactorLearningCandidateIdeas.jsx`
- `frontend/src/components/FactorLearningMemoryGrid.jsx`
- `frontend/src/components/FactorLearningOperatorLibrary.jsx`
- `frontend/src/components/FactorLearningStatusBoxes.jsx`
- `frontend/src/components/ModelFamilyBoard.jsx`
- `frontend/src/components/useFactorLearningData.js`
- 以上组件对应的 CSS 文件
- `frontend/src/pages/FactorLearningPage.css`
- `frontend/src/pages/FactorsDetail.css`
- `frontend/src/assets/hero.png`
- `frontend/src/assets/vite.svg`
- `frontend/src/assets/typescript.svg`

说明：

- 仍有路由入口的页面没有删除。
- 删除对象均为当前导入图不可达文件或默认旧资产。

## 测试新增与修改

### 新增测试

- `backend/tests/test_event_contract_rebuild.py`
- `backend/tests/test_reset_derived_research_data.py`

覆盖内容：

- 清理脚本 dry-run 不删除数据。
- `--confirm` 只删除派生数据。
- K 线缺口触发 Binance 重拉。
- 重拉失败暴露明确错误。
- 环境识别只使用入场前完成 K 线。
- 候选不足时最终裁判输出 `SKIP`。
- 模型族一致时最终裁判输出 `UP`。
- 模型族特征包含 `regime_*`，不再包含 `sim_feedback_*` 和 `factor_combo_*`。

### 修改测试

- `backend/tests/test_model_family_system.py`

变化：

- fake dataset 改为包含 `regime_*`。
- 旧仿真反馈/旧组合缓存断言改为新版环境特征断言。

## 已执行验证

后端 targeted pytest：

```powershell
cd backend
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest -q tests/test_event_contract_rebuild.py tests/test_reset_derived_research_data.py tests/test_strategy_prediction_readiness.py
```

结果：

- 15 passed

后端模型族与自动预测相关 targeted pytest：

```powershell
cd backend
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest -q tests/test_model_family_system.py::test_model_family_train_can_publish_initial_baseline tests/test_model_family_system.py::test_model_family_report_includes_training_input_observability tests/test_auto_predict_service.py tests/test_model_family_status_progress.py tests/test_db_write_lock.py
```

结果：

- 54 passed

质量门：

```powershell
python backend\scripts\agents_quality_gate.py
```

结果：

- `AGENTS quality gate: total=0 new=0`

前端构建：

```powershell
cd frontend
npm run build
```

结果：

- build passed

## 已知环境问题

较大范围回归中，`tests/test_direction_prediction_metrics.py::test_forward_validation_prediction_correct_tracks_direction_without_cost` 曾失败于 Windows 本地权限问题：

- 无法创建或清理 `backend/runtime/pytest-temp/...`
- `.pytest_cache` 也出现 `WinError 5` 写入警告

该问题表现为本地 runtime/pytest 临时目录 ACL 或文件锁问题，不是本次事件合约重构逻辑失败。

## 当前结果

本次重构后，系统已经具备新版事件合约方向预测链路：

- 可以清理旧派生研究数据。
- 可以扫描并修复 K 线质量问题。
- 可以生成市场环境标签。
- 模型族训练和实时预测已切换到新版环境特征。
- 最终决定层可以输出 `UP / DOWN / SKIP`。
- `UP / DOWN` 进入模拟事件验证。
- `SKIP` 只进入审计，不污染预测和模拟订单。
- 前端已识别最终裁判策略名，并删除一批不可达旧 UI。
