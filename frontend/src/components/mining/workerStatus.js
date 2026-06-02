export const REQUIRED_COMMAND_FALLBACK = "python backend/scripts/run_model_search_worker.py --loop --adaptive-parallelism";

const LABELS = Object.freeze({
  queued: "已入队，worker 正在处理队列",
  running: "worker 正在执行",
  failed: "worker 执行失败",
  worker_required: "已入队，需启动 worker",
  idle: "worker idle",
});

export function miningWorkerStatusView(status) {
  const state = status?.state || "idle";
  return {
    state,
    tone: workerTone(state),
    label: LABELS[state] || `worker ${state}`,
    detail: workerDetail(status, state),
    command: status?.workerRequiredCommand || REQUIRED_COMMAND_FALLBACK,
    latestLogPath: status?.latestLogPath || null,
    failureReason: status?.latestFailureReason || null,
  };
}

function workerTone(state) {
  if (state === "failed") return "error";
  if (["queued", "running"].includes(state)) return "running";
  if (state === "worker_required") return "warn";
  return "idle";
}

function workerDetail(status, state) {
  const pending = Number(status?.pendingJobs || 0);
  const running = Number(status?.runningJobs || 0);
  const failed = Number(status?.failedJobs || 0);
  if (state === "worker_required") return `pending ${pending}，未检测到 running worker`;
  if (state === "queued") return `pending ${pending}，running ${running}`;
  if (state === "running") return `running ${running}`;
  if (state === "failed") return `failed ${failed}`;
  return "无待执行任务";
}
