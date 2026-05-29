# AGENTS 质量门禁

`backend/scripts/agents_quality_gate.py` 将项目 AGENTS 约束落成可执行检查。

## 检查范围

- 默认扫描 `backend/app` 与 `backend/scripts`。
- 检查文件长度、函数长度、位置参数数量、圈复杂度、嵌套深度、空 `except: pass`。
- `backend/check_backend.ps1` 会在 pytest 前运行该门禁。

## 当前策略

当前代码库存在历史质量债务，基线记录在：

```text
backend/docs/agents_quality_baseline.json
```

门禁会允许基线内的既有问题继续存在，但新增或恶化的问题会失败。这不是静默兜底；基线文件显式列出了每一条历史债务。

## 命令

从 `backend` 目录运行：

```powershell
..\.venv\Scripts\python.exe scripts\agents_quality_gate.py --root ..
```

只在完成有意重构并确认债务变化后刷新基线：

```powershell
..\.venv\Scripts\python.exe scripts\agents_quality_gate.py --root .. --update-baseline
```
