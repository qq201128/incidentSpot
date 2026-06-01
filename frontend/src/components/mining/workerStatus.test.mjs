import assert from "node:assert/strict";
import { miningWorkerStatusView } from "./workerStatus.js";

const required = miningWorkerStatusView({ state: "worker_required", pendingJobs: 2 });
assert.equal(required.label, "已入队，需启动 worker");
assert.equal(required.tone, "warn");
assert.match(required.command, /run_model_search_worker\.py --loop/);
assert.equal(required.detail, "pending 2，未检测到 running worker");

const queued = miningWorkerStatusView({ state: "queued", pendingJobs: 3, runningJobs: 1 });
assert.equal(queued.label, "已入队，worker 正在处理队列");
assert.equal(queued.tone, "running");
assert.equal(queued.detail, "pending 3，running 1");

const running = miningWorkerStatusView({ state: "running", runningJobs: 1, latestLogPath: "runtime/job.log" });
assert.equal(running.label, "worker 正在执行");
assert.equal(running.latestLogPath, "runtime/job.log");

const failed = miningWorkerStatusView({ state: "failed", failedJobs: 1, latestFailureReason: "training crashed" });
assert.equal(failed.label, "worker 执行失败");
assert.equal(failed.tone, "error");
assert.equal(failed.failureReason, "training crashed");

const idle = miningWorkerStatusView(null);
assert.equal(idle.label, "worker idle");
assert.equal(idle.detail, "无待执行任务");
