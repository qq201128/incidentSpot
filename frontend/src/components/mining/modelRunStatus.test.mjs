import assert from "node:assert/strict";
import { attachModelRunStatuses, modelRunStatusView } from "./modelRunStatus.js";

assert.equal(modelRunStatusView({ cardState: "ready" }).state, "ready");
assert.equal(modelRunStatusView({ cardState: "ready" }).ready, true);

const required = modelRunStatusView(
  { cardState: "searching", candidateLibraryTotal: 2 },
  { state: "worker_required", pendingWorker: true, latestLogPath: "runtime/job.log" },
);
assert.equal(required.label, "等待 worker");
assert.equal(required.pendingWorker, true);
assert.equal(required.latestLogPath, "runtime/job.log");
assert.equal(required.actionLabel, "等待 worker");

const failed = modelRunStatusView(
  { cardState: "blocked" },
  { state: "failed", latestFailureReason: "training crashed" },
);
assert.equal(failed.label, "失败");
assert.equal(failed.latestFailureReason, "training crashed");

const attached = attachModelRunStatuses(
  [{ modelFamily: "knn", cardState: "pending_train" }],
  { models: [{ modelFamily: "knn", state: "queued" }] },
);
assert.equal(attached[0].runtimeStatus.state, "queued");
