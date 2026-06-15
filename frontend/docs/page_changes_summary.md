# 前端页面调整汇总

更新时间：2026-06-03

## 页面目标

前端页面需要从旧的规则胜率、因子组合、历史缓存展示，逐步收敛到事件合约方向预测工作台。

页面关注点调整为：

- 当前事件周期。
- 最终裁判输出：`UP / DOWN / SKIP`。
- 市场环境标签。
- 模型族是否已按新版环境特征重训。
- 模拟事件合约验证结果。
- 真实交易状态保持关闭。

## 当前保留页面

### 工作台

路径：

- `/`

用途：

- K 线、盘口、近期成交展示。
- 事件合约模拟入口。
- 自动策略控制。
- 事件记录列表。

保留原因：

- 这是事件合约方向预测的主页面。
- 最终裁判策略后续应优先展示在这里。

### 规则命中率

路径：

- `/rule-hit-rate`

用途：

- 展示策略历史命中率。
- 展示近期策略事件。

保留原因：

- 仍可用于观察最终裁判策略的历史表现。
- 后续展示重心应从旧规则胜率转为最终裁判命中率。

### 样本观测

路径：

- `/event-governance`

用途：

- 查看事件样本。
- 观察事件结算、样本状态和治理信息。

保留原因：

- 新系统仍需要事件样本审计。

### 研究驾驶舱

路径：

- `/research-dashboard`

用途：

- 展示模型族、研究状态、验证证据。
- 实盘开启总览显示每个候选最近一次已结算实盘是否正确、入场价、结算价、真实事件开仓时间和事件结束时间。

保留原因：

- 模型族重训后仍需要观察训练状态和分组表现。
- 页面文案应逐步替换旧缓存/旧规则表达。

### 因子库

路径：

- `/factors`

用途：

- 查看因子基础定义、因子概览和因子排名。

保留原因：

- 原始因子定义仍可作为模型族输入。
- 旧的因子组合历史胜率缓存不应再作为新系统核心依据。

### 自动挖掘

路径：

- `/learning`

用途：

- 展示模型族搜索、训练、候选状态。

保留原因：

- 模型族需要按新版环境特征重新训练。
- 页面应显示 `clean_event_retrain_required` 等重训状态。

## 已删除旧 UI

以下文件已经删除，因为它们不在当前路由入口的导入图中，属于不可达旧 UI 或默认旧资产。

### 旧因子学习面板

- `frontend/src/components/FactorLearningPanel.jsx`
- `frontend/src/components/FactorLearningPanel.css`
- `frontend/src/components/FactorLearningCards.css`
- `frontend/src/components/FactorAdaptiveLearningPanel.jsx`
- `frontend/src/components/FactorAdaptiveLearningPanel.css`
- `frontend/src/components/FactorLearningCandidateIdeas.jsx`
- `frontend/src/components/FactorLearningMemoryGrid.jsx`
- `frontend/src/components/FactorLearningOperatorLibrary.jsx`
- `frontend/src/components/FactorLearningStatusBoxes.jsx`
- `frontend/src/components/FactorLearningStatusBoxes.css`
- `frontend/src/components/useFactorLearningData.js`

### 旧模型族面板

- `frontend/src/components/ModelFamilyBoard.jsx`
- `frontend/src/components/ModelFamilyBoard.css`
- `frontend/src/components/ModelFamilyBoardLabels.js`

### 旧事件列表占位

- `frontend/src/components/EventList.jsx`

### 未使用样式和资产

- `frontend/src/pages/FactorLearningPage.css`
- `frontend/src/pages/FactorsDetail.css`
- `frontend/src/assets/hero.png`
- `frontend/src/assets/vite.svg`
- `frontend/src/assets/typescript.svg`

## 已修改页面相关内容

### 策略名称

修改文件：

- `frontend/src/utils/strategyLabels.js`

新增策略标签：

```text
event_final_decision_v1 -> 事件最终裁判模拟
```

影响页面：

- 工作台
- 事件记录列表
- 自动策略控制
- 规则命中率
- 研究驾驶舱中引用策略名的位置

## 页面后续应展示的信息

### 工作台应展示

建议增加“事件最终裁判”区域，展示：

- 当前周期。
- 当前环境：趋势状态、波动状态、环境置信度。
- 最终输出：`UP / DOWN / SKIP`。
- `probability_up`。
- `final_score`。
- 候选模型数量。
- `SKIP` 原因。
- 最近一次结算是否正确。

### 研究驾驶舱应展示

建议增加按环境分组的模型族验证结果：

- `trend_up`
- `trend_down`
- `range`
- `uncertain`
- `high_vol`
- `normal_vol`
- `low_vol`

并明确显示：

- 哪些模型族已经使用 `regime_*` 特征训练。
- 哪些模型族被阻止继续预测。
- 阻止原因：`clean_event_retrain_required`。

### 规则命中率页应调整

展示重心应从旧规则胜率切换为最终裁判验证：

- 总命中率。
- 按周期命中率。
- 按环境命中率。
- `UP / DOWN / SKIP` 分布。
- 高置信度分桶真实命中率。

### 因子库页应调整

页面可以保留因子定义和基础指标展示。

需要弱化或移除的旧表达：

- 旧因子组合历史胜率。
- 旧 `high_winrate_*` 排名。
- 旧 `factor_combo_*` 缓存表现。

## 页面不应继续强化的内容

- 不应继续把旧规则胜率当作核心目标。
- 不应继续把旧因子组合缓存排名当作模型依据。
- 不应把 `SKIP` 当作错误状态。
- 不应显示真实交易已启用，除非用户明确开启。
- 不应用旧仿真反馈特征解释新模型族表现。

## 当前验证

已执行前端构建：

```powershell
cd frontend
npm run build
```

结果：

- build passed
