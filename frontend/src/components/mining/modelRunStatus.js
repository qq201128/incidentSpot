const STATE_LABELS = Object.freeze({
  idle: "空闲",
  ready: "ready",
  queued: "已入队",
  running: "搜索中",
  worker_required: "等待 worker",
  failed: "失败",
  searching: "搜索中",
  pending_train: "待训练",
  blocked: "已阻断",
});

export function modelRunStatusView(model, runtime) {
  const state = runtime?.state || model?.cardState || "idle";
  return {
    state,
    label: STATE_LABELS[state] || state,
    ready: runtime?.ready ?? model?.cardState === "ready",
    pendingWorker: state === "worker_required" || runtime?.pendingWorker === true,
    latestFailureReason: runtime?.latestFailureReason || model?.latestFailureReason || null,
    latestLogPath: runtime?.latestLogPath || model?.latestLogPath || null,
    candidateLibraryTotal: runtime?.candidateLibraryTotal ?? model?.candidateLibraryTotal ?? 0,
    latestCandidateLabel: runtime?.latestCandidateLabel || model?.latestCandidateLabel || null,
    actionLabel: actionLabel(state),
  };
}

export function attachModelRunStatuses(models, runStatus) {
  const byFamily = new Map((runStatus?.models || []).map((row) => [row.modelFamily, row]));
  return (models || []).map((model) => ({
    ...model,
    runtimeStatus: modelRunStatusView(model, byFamily.get(model.modelFamily)),
  }));
}

function actionLabel(state) {
  if (state === "worker_required") return "等待 worker";
  if (state === "queued") return "已入队";
  if (state === "running") return "搜索中";
  return "处理中";
}
