# 因子挖掘闭环步骤与提示词

## 闭环结论

当前主链路已经具备闭环：多因子组合刷新后会把达标组合写入挖掘因子库，再刷新因子学习记忆并触发 Agent 复盘；Agent 候选经过公式物化和单因子回测达标后写入 Agent 单因子库，下一次组合搜索会按来源配额重新纳入候选。

页面侧需要重点观察四个反馈点：组合回灌数、Agent 入库数、本地复盘来源、Agent 候选结果。若其中任一项长期为 0 或失败，优先按下列步骤排查。

## 详细步骤

1. 本地复盘基线
   - 输入：`symbol`、`duration`、当前组合排名缓存、已结算预测、挖掘库。
   - 动作：执行 `/api/factor-learning/refresh?runAgent=false`。
   - 预期：结算到期预测，刷新亏损模式、自动权重、监控告警、挖掘库摘要。
   - 失败判断：`rankingRefreshSource` 长期无缓存或重建失败；`lossMemoryStatus` 样本不足；`minedFrameFailureCount` 大于 0。

2. 多因子组合重算
   - 输入：增强特征帧、原生因子、组合回灌因子、Agent 单因子。
   - 动作：执行 `/api/factors/combinations/refresh`。
   - 预期：刷新组合排名缓存，达标组合进入 `mined_factor_library.json`。
   - 失败判断：`baseFactorCount` 过低、`minedFactorUsedCount` 为 0、`failureCount` 持续上升。

3. 组合回灌验证
   - 输入：组合排名中的 `winRate`、`profitFactor`、`members`。
   - 动作：只允许同时满足胜率和盈亏比阈值的组合入库。
   - 预期：新组合在因子库显示为“回灌”，并在下一轮组合搜索中作为候选基础因子。
   - 失败判断：组合排名有高分组合但 `组合回灌` 为 0，需要检查阈值、成员字段和公式物化。

4. 联网 Agent 挖掘
   - 输入：本地复盘记忆、算子库、成功模式、禁区、亏损模式、已入库 Agent 因子名单。
   - 动作：执行 `/api/factor-learning/refresh?runAgent=true` 或 `/api/factor-learning/agent/review`。
   - 预期：Agent 输出候选单因子；系统自动物化公式并运行单因子回测。
   - 失败判断：`agentCandidatePromotion.records` 中出现 unsupported function、column not found、rejected_metrics。

5. Agent 单因子回灌
   - 输入：Agent 候选公式、现有特征列、单因子回测指标。
   - 动作：通过物化和指标门槛后写入 `agent_mined_factor_library.json`。
   - 预期：因子库显示为“Agent”，下一轮组合搜索按 Agent 来源配额纳入。
   - 失败判断：候选数量大于 0 但 `Agent入库` 长期为 0，需要收紧提示词到现有算子和列。

6. 实盘模拟监控与再复盘
   - 输入：已结算多因子模拟预测、LSTM 影子策略、监控告警。
   - 动作：当成功率或候选成功率偏低时，先执行本地复盘，再检查组合成员、亏损特征和入库阈值。
   - 预期：亏损特征进入过滤器，权重和组合质量分随复盘更新。
   - 失败判断：连续亏损告警后复盘无变化，检查预测结算、策略 key 和事件对齐。

## 提示词模板

### 本地复盘提示词

```text
请基于 {symbol} {duration} 的因子学习记忆做本地复盘。
重点检查：
1. rankingRefreshSource、rankingTotal、baseFactorCount 是否说明组合排名可用；
2. settledPredictionCount、lossMemoryStatus、lossPatternCount 是否足够学习亏损模式；
3. minedFactorSourceCount、minedFactorUsedCount、minedFrameFailureCount 是否说明回灌因子可物化；
4. monitoring.status、issues、solutions 是否要求重新复盘或人工确认。
输出：
- 当前闭环是否完整；
- 断点字段和原因；
- 下一步只允许给出可执行动作，不要给模拟成功或兜底方案。
```

### 多因子组合提示词

```text
请审查 {symbol} {duration} 的多因子组合排名闭环。
输入包括 ranking、baseFactors、searchConfig、failureCount、minedFactorUsedCount、agentMinedFactorUsedCount。
请逐项判断：
1. 原生、回灌、Agent 三类候选是否都进入 baseFactors；
2. 高胜率组合是否满足 winRate 与 profitFactor 入库阈值；
3. 成员因子是否重复、过度相关或来自不可物化的组合；
4. 缓存是否 stale，是否需要重算。
输出：
- 可入库组合列表；
- 不应入库组合及原因；
- 下一轮组合搜索的候选来源配额建议。
```

### Agent 自动挖掘提示词

```text
你是量化因子挖掘 Agent。只能基于输入记忆提出候选研究方向，不得声称真实回测已通过。
必须遵守：
1. 只使用 operator_library 中存在的算子；
2. 只引用 memory 中可用或明确存在的特征列；
3. 不重复 doNotSuggestFactorNames；
4. 每个候选必须写清 displayNameZh、formulaHint、operatorTrace、requiredColumns、validationChecks；
5. validationChecks 必须覆盖公式物化、单因子回测、相关性、亏损禁区和能否回流到因子库。
输出必须是 JSON 对象，并符合 required_schema。
```

### 因子库复核提示词

```text
请复核因子库展示与回灌闭环。
输入包括 factors、comboFactors、ranking、minedFactorLibrary、agentMinedFactorLibrary。
检查：
1. 因子来源是否能区分原生、组合回灌、Agent；
2. 组合回灌因子是否显示成员、胜率、盈亏比、promotionCount；
3. Agent 单因子是否显示公式、入库指标、失败记录；
4. 排名页是否能看出该因子是否参与下一轮组合搜索。
输出：
- 展示缺失字段；
- 容易误判的字段；
- 必须补到页面上的状态指标。
```
