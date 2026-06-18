# incidentSpot

incidentSpot 是一个面向 Binance USD-M 指数事件交易研究的本地工作台。它把指数价、指数 K 线、订单簿、近期成交、事件交易、结算验证、因子研究、多模型候选搜索和纸面实盘观测集中在同一个应用里，目标是让每一笔交易信号的输入、执行状态、失败原因和结算结果都可解释、可复查。

项目当前按 `PRODUCT_TARGET.md` 定义的边界运行：真实交易保持关闭，模拟交易必须明确显示“未调用 Binance”，上游接口、WebSocket、模型训练、测试依赖和后台任务失败都要显式暴露，不能用 mock 成功或静默兜底掩盖问题。

## 核心能力

- 行情工作台：展示 Binance 指数价、指数 K 线、普通 K 线、订单簿、近期聚合成交和最新预测。
- 事件交易闭环：支持手动事件、快速事件、订单记录、自动结算和事件列表检索。
- 模拟/实盘边界：`liveTradingEnabled=false` 时只写本地事件和订单；`liveTradingEnabled=true` 时必须调用 Binance 下单接口，失败会显式报错并记录。
- 规则预测：基于策略窗口刷新 K 线、生成方向预测、写入预测缓存并通过 WebSocket 广播。
- 自动交易观察：按策略槽位读取最新预测，满足入场条件时创建事件和订单；手动开启实盘的稳定 paper-live 单因子候选会用最新 paper-live 稳定性作为执行门控。
- 因子研究：提供单因子目录、单因子回测、因子排名、多因子组合排名和组合信号 watchlist。
- 因子学习：把组合排名、已结算预测、亏损模式、回灌因子库和 Agent 候选写入学习记忆。
- 多模型族搜索：支持 LSTM、GRU、CNN、Transformer、RandomForest、XGBoost、LightGBM、CatBoost、SVM、Bayesian、KNN 等候选搜索任务。
- 研究驾驶舱：聚合纸面实盘候选、模型族状态、已结算 Event 证据、稳定性和失败原因。

## 技术栈

后端：

- FastAPI + Uvicorn
- SQLite，数据库文件为 `backend/data.db`
- pandas、numpy、scikit-learn、joblib
- LightGBM、CatBoost、XGBoost、PyTorch
- Binance USD-M 行情接口与 WebSocket 代理

前端：

- Vite
- React 19
- React Router
- Axios
- lightweight-charts
- 原生 CSS 分模块样式

## 目录结构

```text
incidentSpot/
├── AGENTS.md                         # 项目级 Agent/工程约束
├── PRODUCT_TARGET.md                 # 产品目标、边界和验收标准
├── pytest.ini                        # 后端 pytest 配置
├── README.md                         # 项目入口文档
├── backend/
│   ├── app/
│   │   ├── main.py                   # FastAPI 入口、健康检查、核心 WebSocket
│   │   ├── app_startup.py            # 启动、延迟路由注册、后台循环管理
│   │   ├── api/                      # HTTP API 与 WS 路由
│   │   ├── config/                   # .env 加载
│   │   ├── db/                       # SQLite schema、迁移、连接
│   │   └── services/                 # 行情、预测、交易、因子、模型、后台任务
│   ├── docs/                         # 后端流程、质量门禁、因子研究文档
│   ├── scripts/                      # 回填、训练、worker、诊断和质量门禁脚本
│   ├── tests/                        # 后端测试
│   ├── check_backend.ps1             # 后端质量门禁 + pytest 统一入口
│   ├── requirements.txt              # 后端运行依赖
│   └── dev-requirements.txt          # 后端测试依赖
├── frontend/
│   ├── src/
│   │   ├── App.jsx                   # 前端视图路由与交易工作台状态
│   │   ├── api/                      # 后端 API client
│   │   ├── components/               # 工作台、事件、因子、模型、挖掘组件
│   │   ├── hooks/                    # 行情、预测、图表 UI hooks
│   │   ├── pages/                    # 研究驾驶舱、因子库、自动挖掘页面
│   │   └── utils/                    # 标签、时间、K 线和展示工具
│   ├── vite.config.js                # Vite dev server 与 API/WS 代理
│   └── package.json                  # 前端依赖与构建脚本
└── runtime/                          # 运行日志或临时状态，具体内容由任务生成
```

## 页面入口

前端主导航包含以下页面：

- `/`：工作台，展示行情、图表、预测、事件交易和执行状态。
- `/research-dashboard`：研究驾驶舱，聚合候选、模型和已结算 Event 证据。
- `/factors`：因子库、因子排名和组合因子。
- `/learning`：自动挖掘与因子学习记忆。

## 后端接口概览

核心 HTTP API：

- `GET /health`：服务健康检查，包含启动状态和后台循环状态。
- `GET /api/last-price`：Binance premiumIndex 指数价与标记价。
- `GET /api/index-price`：官方指数价、标记价、资金费率时间。
- `GET /api/index-klines`：Binance indexPriceKlines。
- `GET /api/klines`：本地 K 线缓存；样本不足时回源 Binance 并写库。
- `GET /api/depth`：USD-M 订单簿。
- `GET /api/agg-trades`：近期聚合成交。
- `POST /api/predict`：规则预测。
- `GET /api/predict/latest`：最新预测缓存。
- `GET /api/workbench/summary`：工作台摘要。
- `GET|POST|DELETE /api/events`：事件列表、创建和删除。
- `POST /api/events/quick-trade`：创建快速事件与订单上下文。
- `POST /api/events/{event_id}/settle`：结算事件。
- `GET /api/rules/backtest`：规则回测。
- `GET|POST /api/ensemble/*`：集成判断状态、排名、刷新和阶段确认。
- `GET|PUT /api/auto-trade/*`：自动交易设置、策略槽位、状态和模拟槽位。
- `GET|POST /api/factors/*`：因子列表、详情、回测、排名和刷新。
- `GET|POST /api/factors/combinations/*`：组合排名、组合信号、纸面实盘候选和组合刷新。
- `GET|POST /api/factor-learning/*`：学习记忆、复盘刷新、Agent 复盘、算子库和挖掘库。
- `GET|POST /api/models/*`：模型族状态、候选搜索入队、预测和搜索队列状态。
- `GET /api/mining/overview`：挖掘总览。

WebSocket：

- `WS /ws/klines`
- `WS /ws/index-klines`
- `WS /ws/agg-trades`
- `WS /ws/predictions`

更完整的后端流程说明见 `backend/docs/backend_process_flow.md`。

## 本地环境准备

### 1. Python 环境

项目测试脚本默认使用根目录 `.venv`：

```powershell
cd D:\Desktop\incidentSpot
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt -r backend\dev-requirements.txt
```

如果已经存在 `.venv`，只需要确认依赖已安装。

### 2. 前端依赖

```powershell
cd D:\Desktop\incidentSpot\frontend
npm install
```

### 3. 环境变量

后端启动时会读取仓库根目录 `.env`。缺少 `.env` 不会阻止启动，但需要外部能力时必须显式配置。

常用变量：

```text
# CORS
CORS_ORIGINS=http://127.0.0.1:5173,http://localhost:5173

# 后端并发
ASYNCIO_THREAD_POOL_WORKERS=48

# 运行时标的，默认 BTCUSDT,ETHUSDT
FACTOR_RANKING_SYMBOLS=BTCUSDT,ETHUSDT

# 因子学习 LLM Agent，可选
SILICONFLOW_API_KEY=
SILICONFLOW_MODEL=deepseek-ai/DeepSeek-V3.2
SILICONFLOW_CHAT_COMPLETIONS_URL=https://api.siliconflow.cn/v1/chat/completions
SILICONFLOW_TIMEOUT_SECONDS=180

# LSTM 每日复盘
LSTM_DAILY_REVIEW_ENABLED=1
LSTM_DAILY_REVIEW_TZ=Asia/Shanghai
LSTM_DAILY_REVIEW_AT=02:00

# LSTM 候选重试，默认关闭
LSTM_CANDIDATE_RETRY_ENABLED=0

# Binance WebSocket 连接调试，可选
BINANCE_WS_USE_WINDOWS_PROXY=
BINANCE_FSTREAM_CONNECT_IP=
BINANCE_FSTREAM_CONNECT_PORT=

# 微信通知，可选
WXPUSHER_APP_TOKEN=
WXPUSHER_UIDS=
WXPUSHER_TOPIC_IDS=
```

前端开发服务器默认使用 Vite 同源代理，不需要配置 `VITE_API_BASE_URL`。需要直连远端后端时再配置：

```text
VITE_DIRECT_API=1
VITE_API_BASE_URL=http://127.0.0.1:8000
VITE_WS_BASE_URL=ws://127.0.0.1:8000
```

Vite 代理目标可通过下面变量覆盖：

```text
VITE_DEV_API_PROXY_TARGET=http://127.0.0.1:8000
VITE_DEV_WS_PROXY_TARGET=ws://127.0.0.1:8000
```

## 启动项目

### 1. 启动后端

```powershell
cd D:\Desktop\incidentSpot\backend
$env:PYTHONPATH="D:\Desktop\incidentSpot\backend"
..\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

仓库中的 `启动明亮` 文件记录了基础启动命令：

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 2. 启动前端

```powershell
cd D:\Desktop\incidentSpot\frontend
npm run dev
```

默认访问：

```text
http://127.0.0.1:5173
```

开发环境下浏览器只访问 `:5173`，`/api` 与 `/ws` 由 Vite 代理到后端 `:8000`。

## 验证与质量门禁

### 后端完整检查

从项目根目录运行：

```powershell
.\backend\check_backend.ps1
```

这个脚本会：

1. 使用根目录 `.venv\Scripts\python.exe`。
2. 设置 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`。
3. 先运行 `backend/scripts/agents_quality_gate.py`。
4. 再运行后端 pytest。
5. 把 pytest 临时目录放到 `backend/runtime/pytest-runs/<runId>`。

### 后端定向测试

```powershell
cd D:\Desktop\incidentSpot\backend
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"
..\.venv\Scripts\python.exe -m pytest tests\test_market_api.py
```

运行后端测试时建议给命令设置 60 秒超时，避免任务卡住。

### 前端构建

```powershell
cd D:\Desktop\incidentSpot\frontend
npm run build
```

### 前端轻量测试

部分前端逻辑测试是 `node:assert` 文件，可直接运行：

```powershell
cd D:\Desktop\incidentSpot\frontend
node src\components\mining\workerStatus.test.mjs
node src\components\mining\modelRunStatus.test.mjs
node src\pages\researchDashboardData.test.mjs
```

## 数据与运行时状态

- `backend/data.db`：SQLite 主数据库，保存 K 线、预测、事件、订单、结算、自动交易设置、策略槽位和缓存状态。
- `backend/models/factor_learning/*.json`：因子学习记忆、挖掘因子库、Agent 候选和相关持久化状态。
- `backend/rules/*.json`：规则回测或策略政策文件，按实际运行生成。
- `runtime/` 与 `backend/runtime/`：worker 日志、pytest 临时目录、模型搜索任务日志等运行时输出。
- `frontend/dist/`：前端构建产物。

运行时文件通常不应手工编辑。需要排查时优先读取 API 返回、后台日志、状态文件和测试输出。

## 核心运行流程

### 启动流程

1. `backend/app/main.py` 加载根目录 `.env`。
2. FastAPI 创建应用、CORS 和基础状态字段。
3. 启动时调用 `bootstrap_application()`。
4. `init_db()` 创建或迁移 SQLite schema。
5. 先注册核心路由：market、events、workbench、stream。
6. 再异步注册延迟路由：auto-trade、ensemble、factors、factor-learning、mining、models 等。
7. 启动后台循环：自动结算、自动预测、自动交易、因子排名、市场上下文、组合刷新、每日复盘等。

### 行情与预测

1. 前端请求 `/api/index-klines`、`/api/last-price`、`/api/depth` 或 WebSocket。
2. 后端从 Binance 拉取真实行情，失败时返回明确错误。
3. `/api/klines` 会优先读本地 `klines`，样本不足时回源 Binance 并写库。
4. `/api/predict` 刷新最新 1m K 线后调用规则预测。
5. 预测结果写入缓存，并通过 `/ws/predictions` 广播。

### 事件交易与结算

1. 前端创建普通事件或 quick-trade。
2. 后端校验事件周期、规则类型、方向、AI 质量字段和策略 key。
3. 模拟交易只写本地事件与订单，并在响应中标记 simulated。
4. 当前真实交易被项目策略禁用，开启请求会返回明确错误。
5. 事件到期后，自动结算循环读取 Binance premiumIndex 或本地指数价 tick。
6. 结算服务写入 settlements，更新事件状态，并记录 AI 预测是否正确。

### 因子与组合

1. 后台或接口刷新单因子排名缓存。
2. 组合刷新读取增强特征、原生因子、组合回灌因子和 Agent 单因子。
3. 达标组合写入挖掘因子库。
4. 组合信号和实盘模拟结果会进入因子学习记忆。
5. 页面通过因子库、研究驾驶舱和自动挖掘页面观察闭环状态。

### 模型候选搜索

1. `/api/models/{family}/candidate-search` 只负责入队，不在请求线程内直接训练。
2. worker 从队列中读取任务并执行候选搜索。
3. 训练状态、失败原因、日志路径和纸面实盘准入状态通过模型状态接口返回。
4. 队列状态可通过 `/api/models/search/jobs/status` 或脚本查看。

## 模型搜索命令

仓库根目录的 `全量搜索命令` 文件记录了完整搜索流程。核心步骤如下：

### 1. 入队

```powershell
cd D:\Desktop\incidentSpot
$env:PYTHONPATH="D:\Desktop\incidentSpot\backend"

D:\Desktop\incidentSpot\.venv\Scripts\python.exe backend\scripts\enqueue_model_search.py `
  --symbols BTCUSDT,ETHUSDT `
  --durations 10m 30m 60m 1d `
  --families lstm gru cnn transformer random_forest extra_trees xgboost lightgbm catboost logistic_elasticnet svm bayesian knn rl_strategy `
  --profile full
```

### 2. 启动 worker

```powershell
D:\Desktop\incidentSpot\.venv\Scripts\python.exe backend\scripts\run_model_search_worker.py `
  --run-until-empty `
  --max-running-jobs 0 `
  --internal-threads 4 `
  --parallel-workers 1 `
  --xgboost-process-workers 1 `
  --torch-jobs 1 `
  --resource-profile local_safe `
  --log-dir runtime\model-search-jobs `
  --stale-after-seconds 3600
```

### 3. 查看状态

```powershell
D:\Desktop\incidentSpot\.venv\Scripts\python.exe backend\scripts\model_search_status.py `
  --symbols BTCUSDT,ETHUSDT `
  --compact `
  --json
```

worker 失败会直接暴露错误并退出，不应改成伪成功或静默降级。

## 后端脚本

常用脚本：

- `backend/scripts/agents_quality_gate.py`：项目质量门禁。
- `backend/scripts/backfill_market_data.py`：市场数据回填。
- `backend/scripts/data_coverage_report.py`：数据覆盖报告。
- `backend/scripts/ingest_market_context.py`：市场上下文采集。
- `backend/scripts/run_factor_backtests.py`：因子回测。
- `backend/scripts/high_winrate_factor_combo_goal.py`：高胜率组合目标搜索。
- `backend/scripts/enqueue_model_search.py`：模型搜索任务入队。
- `backend/scripts/run_model_search_worker.py`：模型搜索 worker。
- `backend/scripts/model_search_status.py`：模型搜索状态。
- `backend/scripts/model_family_search_summary.py`：模型族搜索摘要。
- `backend/scripts/train_lstm.py`：LSTM 训练入口。
- `backend/scripts/run_lstm_candidate_search.py`：LSTM 候选搜索。
- `backend/scripts/run_model_family_full_search.py`：模型族全量搜索。

## 工程约束

项目遵循 debug-first 策略：

- 不新增 silent fallback。
- 不返回 mock 成功。
- 不吞掉异常。
- 不把真实接口失败改写成模拟成功。
- 不为“看起来能跑”新增隐藏边界、隐藏 caps 或模板成功路径。
- 真实交易保持关闭，除非用户明确要求改变项目策略。

代码质量门禁见 `backend/docs/agents_quality_gate.md`。当前历史债务基线在 `backend/docs/agents_quality_baseline.json`，门禁允许基线内既有问题存在，但新增或恶化的问题会失败。

## 更多文档

- `PRODUCT_TARGET.md`：产品目标、阶段边界、验收标准和错误暴露原则。
- `backend/docs/backend_process_flow.md`：后端启动、接口、行情、事件、因子、模型和自动交易流程总览。
- `backend/docs/agents_quality_gate.md`：AGENTS 质量门禁说明。
- `backend/docs/factor_learning_closed_loop.md`：因子挖掘闭环步骤、排查点和提示词。
- `backend/docs/quant_factor_research.md`：量化因子研究基线、防未来函数规则和过拟合控制。
