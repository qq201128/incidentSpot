# 信号裁判架构评估报告

评估日期：2026-06-02

## 1. 结论摘要

当前“信号裁判”对应后端策略键 `ensemble_ranker_v1`，前端展示为“信号裁判 / 综合裁判模拟”。

结论分两层：

- 作为“交易或模拟执行信号”：当前基本没有实际作用。数据库中没有任何 `ensemble_ranker_v1` 预测记录，所有交易对与周期的裁判阶段都停在 `observe`，且未确认 `ensemble_ready`。
- 作为“候选信号治理与诊断面板”：仍然有用。它已经在统计多因子组合、单因子候选、模型族等信号的结算表现，并能暴露哪些信号源样本不足、近期 PF<1、连亏或交易日覆盖不足。

不建议直接删除整个信号裁判。更合理的判断是：保留其“诊断 / 治理”能力，但当前不应把它视为可用的最终执行信号。系统架构确实存在问题，核心问题是裁判状态机、信号源分类和执行策略身份混在了一起。

## 2. 相关实现位置

后端核心：

- `backend/app/services/ensemble_judge_service.py`：刷新裁判、写入阶段状态、写入候选评分、确认阶段。
- `backend/app/services/ensemble_judge_metrics.py`：按已结算预测计算胜率、均值收益、PF、连亏、稳定性和建议权重。
- `backend/app/services/ensemble_ranker_prediction_service.py`：在裁判确认可用后，对同一入场窗口的候选预测做加权投票。
- `backend/app/services/ensemble_signal_identity.py`：用 `strategy_key` 字符串判断信号来源类型。
- `backend/app/api/ensemble.py`：`/api/ensemble/status`、`/api/ensemble/ranking`、`/api/ensemble/refresh`、`/api/ensemble/confirm-stage`。
- `backend/app/services/strategy_registry.py`：把 `ensemble_ranker_v1` 注册为“综合裁判模拟”。
- `backend/app/services/auto_predict_service.py`：自动预测循环会刷新裁判；若裁判策略启用，会调用裁判预测服务。
- `backend/app/services/strategy_prediction_readiness.py`：裁判预测必须满足 `confirmed_stage = ensemble_ready`。
- `backend/app/db/schema.sql`：`ensemble_stage_status` 与 `ensemble_signal_scores` 两张裁判表。

前端核心：

- `frontend/src/components/EnsembleJudgePanel.jsx`：展示阶段、样本覆盖、刷新裁判、确认阶段。
- `frontend/src/components/EnsembleRankingTable.jsx`：展示候选信号排名。
- `frontend/src/components/EventContractPanel.jsx`、`frontend/src/components/EventRecordsTable.jsx`：把裁判面板和候选排名接入事件工作台。

## 3. 当前实际运行状态

本次只读查询了 `backend/data.db`，没有修改数据库。

### 3.1 裁判阶段

`ensemble_stage_status` 中 BTCUSDT 和 ETHUSDT 的 `10m / 30m / 60m / 1d` 全部是：

- `stage = observe`
- `recommended_stage = observe`
- `confirmed_stage = null`
- 原因：`waiting for settled samples across major signal sources`

最新更新时间集中在 2026-06-02 14:40-14:52 UTC。

### 3.2 裁判自身预测

查询结果显示：

- `predictions` 表里没有 `signal_key = ensemble_ranker_v1` 的记录。
- 因此裁判当前没有产生任何自己的模拟预测，也没有自己的结算样本。

这说明“信号裁判”目前不是一个实际运行中的最终信号源。

### 3.3 候选评分已有大量数据

`ensemble_signal_scores` 并不是空的。例如：

- `BTCUSDT 10m`：383 条候选评分，总样本 58,994。
- `BTCUSDT 30m`：275 条候选评分，总样本 20,389。
- `BTCUSDT 60m`：225 条候选评分，总样本 11,189。
- `ETHUSDT 10m`：60 条候选评分，总样本 1,947。

这说明裁判作为候选信号排行榜是有数据基础的。

### 3.4 为什么 BTCUSDT 10m 样本很多但仍不能升阶

`BTCUSDT 10m` 的覆盖情况：

- `factor_combo`：7,343 样本，11 个交易日，最大连亏 5，近期 PF<1。
- `high_winrate_combo`：0 样本。
- `model_family`：16,684 样本，14 个交易日，近期 PF<1。
- `factor_candidate`：34,967 样本，4 个交易日，近期 PF<1。

裁判要求四类主信号源都达标：

- `factor_combo`
- `high_winrate_combo`
- `model_family`
- `factor_candidate`

当前最大硬阻断是 `high_winrate_combo = 0`。其次是多个来源交易日不足、近期 PF<1、或最大连亏达到阈值。

## 4. 当前架构意图

信号裁判的设计意图不是再训练一个模型，而是做后置治理层：

1. 从 `predictions` 读取各类候选信号的已结算结果。
2. 按 `signal_key` 聚合表现。
3. 计算胜率、收益、PF、连亏、稳定性。
4. 给每个候选一个 `weight_suggestion`。
5. 当阶段确认可用后，对同一入场窗口多个候选预测做加权投票，产出 `ensemble_ranker_v1`。

这条思路本身是合理的。问题出在落地状态机和信号源边界。

## 5. 主要架构问题

### 5.1 裁判状态机存在闭环死锁

`ensemble_ranker_prediction_service.py` 要求：

- 只有 `confirmed_stage = ensemble_ready` 才允许生成裁判预测。

但 `ensemble_judge_service.py` 的 `ensemble_ready` 推荐条件又要求：

- 已有至少 100 条已结算的 `ensemble_ranker_v1` 裁判预测。
- 最近窗口收益都为正。

这形成循环依赖：

1. 没有 `ensemble_ready`，不能生成裁判预测。
2. 没有裁判预测，就不能达到 `ensemble_ready`。

测试里是手动插入裁判预测样本来验证 ready 逻辑，但真实后台链路没有看到自然生成这些 shadow 样本的路径。这是当前最严重的架构问题。

### 5.2 `high_winrate_combo` 作为必需信号源，但实际没有样本

裁判升阶要求 `high_winrate_combo` 达标，但当前数据库里该类型样本为 0。

同时，`strategy_registry.py` 已经把 `high_winrate_factor_combo_v1` 标为不可交易，并写明“已并入综合裁判信号层”。这会导致职责冲突：

- 上游高胜率组合独立执行被禁用。
- 裁判又强制等待高胜率组合样本。
- 结果是裁判长期停在观察阶段。

此外，`factor_combo_simulation_keys.py` 当前注释写着 `combo__` 和 `goal_combo__` 共享同一个 batch 前缀，这会让 goal combo 的批量模拟更容易被归类为普通 `factor_combo`，而不是裁判期望的 `high_winrate_combo`。

### 5.3 信号源分类依赖字符串前缀，边界脆弱

`ensemble_signal_identity.py` 主要通过 `strategy_key` 字符串判断类型。

这带来几个问题：

- 策略改名、前缀变更或批量模拟键变更，会影响裁判分类。
- `signal_source` 已经存在于预测 payload 中，但裁判没有优先使用结构化来源字段。
- `goal_combo__` 这种业务身份和 `strategy_key` 执行身份混在一起，导致来源判断容易错位。

更稳的做法是让预测写入时明确保存 `signal_source` / `signal_family` / `candidate_family`，裁判只读结构化字段。

### 5.4 “权重准备”阶段没有实际效果

裁判有 `weight_ready` 阶段，含义是“可启用降权”。但当前链路里：

- `weight_ready` 只记录状态。
- 不会创建裁判 shadow 预测。
- 不会让候选预测应用裁判权重。
- 不会推动 `ensemble_ranker_v1` 积累自身结算样本。

这使 `weight_ready` 更像 UI 文案，而不是可执行状态。

### 5.5 裁判被同时当成报表、策略和执行项

同一个 `ensemble_ranker_v1` 同时承担：

- 候选信号评分报表。
- 阶段治理状态。
- 自动预测目标。
- 自动交易策略配置。

这些职责边界过宽。结果是诊断层还可用，但执行层不可用时，用户会感到“这个裁判到底有没有用”。

### 5.6 加权投票的概率口径不统一

裁判加权时直接平均各候选的 `probability_up`。

但不同来源的 `probability_up` 含义不完全一致：

- 因子组合中可能更多来自历史胜率。
- 模型族来自模型输出概率。
- 单因子候选来自方向胜率。

直接加权平均会把不同口径混在一起。即使状态机修好，也需要校准或统一评分口径，否则“综合裁判”的数学意义偏弱。

### 5.7 性能会随预测表增长变差

刷新裁判会读取并聚合目标 symbol/duration 的已结算预测，然后删除并重写对应评分。

当前 `backend/data.db` 约 3.9GB，`BTCUSDT 10m` 已有大量预测样本。短期还能用，但长期更适合：

- 按 `signal_key + lifecycle_identity` 做增量聚合。
- 避免每次全量扫描。
- 把评分更新时间和数据水位分开记录。

## 6. 是否应该删除

不建议现在直接删除。

理由：

- 它已经承担了候选信号诊断作用，能展示哪些信号源在退化。
- 删除后会失去跨信号源的统一排名和阶段可视化。
- 当前问题主要不是“完全无用”，而是“执行层状态机不闭合、分类口径不一致”。

但如果你的目标是把系统收敛成“只跑单一最优策略 / 模型族，不再做多信号仲裁”，那么可以删除。删除范围会比较大，涉及 API、前端面板、策略注册、数据库表、自动预测链路和测试。

## 7. 建议的优化方向

### 7.1 先拆职责，不先删代码

建议把“信号裁判”拆成两个概念：

- `signal_governance`：只做候选信号评分、覆盖诊断、阶段建议。
- `ensemble_ranker_v1`：只做最终投票预测，且只有在治理层确认后才作为策略暴露。

这样用户看到的“裁判”首先是治理报表，而不是一个看起来可执行但实际没产出的策略。

### 7.2 修正状态机

建议状态变为：

1. `observe`：只收集候选信号表现。
2. `weight_ready`：候选来源足够后，自动生成裁判 shadow 预测，但不创建交易。
3. `ensemble_ready`：裁判 shadow 自身有足够已结算样本且表现达标后，才允许人工确认并进入模拟执行。

关键点：`weight_ready` 必须能产生 `ensemble_ranker_v1` 的 shadow 预测，否则永远无法自然进入 `ensemble_ready`。

### 7.3 修正信号源分类

建议不要再靠字符串前缀判断主类型。预测写入时应明确保存结构化来源：

- `signal_source`
- `signal_family`
- `candidate_family`
- `lifecycle_identity`

如果继续要求 `high_winrate_combo`，就必须保证 `goal_combo__` 相关模拟能被明确归类到 `high_winrate_combo`。如果高胜率组合已经退役，就应从 `MAJOR_SIGNAL_TYPES` 中移除，或改成可选来源。

### 7.4 统一概率与权重口径

短期可以把裁判投票从“直接平均 probability_up”改为：

- 方向投票 + 可靠性权重。
- 或把不同来源概率先校准成统一置信度。

长期可以训练一个轻量 meta-ranker，但前提是不能伪造样本，必须基于真实已结算预测。

### 7.5 改进 UI 阻断说明

前端现在显示“等待样本”，但不够具体。建议直接展示：

- 哪个主信号源为 0。
- 哪个信号源交易日不足。
- 哪个信号源 PF<1。
- 是否存在裁判 shadow 样本死锁。

这样用户能知道是数据问题、策略问题，还是架构状态机问题。

## 8. 决策建议

当前最准确的判断是：

- 信号裁判“还有用”，但它现在主要是诊断工具，不是有效执行信号。
- 系统架构确实有问题，尤其是裁判状态机闭环死锁和 `high_winrate_combo` 来源要求。
- 直接删除会损失已有诊断价值；直接启用也不合理，因为没有裁判自身预测样本。

建议决策优先级：

1. 如果你要保留多策略、多模型、多因子并行探索：优化，不删除。
2. 如果你只想做单策略执行工作台：可以删除裁判执行层，但建议保留候选排行榜。
3. 如果短期不想投入重构：保留现状，但在 UI 或文档中明确标注“当前仅观察，不是有效执行信号”。
