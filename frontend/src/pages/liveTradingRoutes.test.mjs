import assert from "node:assert/strict";
import {
  filterSlotRows,
  liveTradingPath,
  requestFromParams,
  slotIdentity,
} from "./liveTradingRoutes.js";

const rows = [
  { duration: "30m", strategyKey: "factor_beta", symbol: "ethusdt" },
  { duration: "10m", strategyKey: "factor_alpha", symbol: "BTCUSDT" },
  { duration: "1d", strategyKey: "factor_combo", symbol: "BTCUSDT" },
];

const noParams = requestFromParams(new URLSearchParams(""));
assert.equal(noParams.hasSlotRequest, false);
assert.equal(noParams.hasSymbolParam, false);
assert.equal(noParams.hasDurationParam, false);
assert.equal(filterSlotRows(rows, noParams).length, 3);

const slotParams = requestFromParams(
  new URLSearchParams("symbol=ethusdt&duration=30m&strategyKey=factor_beta"),
);
assert.equal(slotParams.hasSlotRequest, true);
assert.equal(slotParams.symbol, "ETHUSDT");
assert.deepEqual(filterSlotRows(rows, slotParams), [rows[0]]);

const path = liveTradingPath({ duration: "30m", strategyKey: "factor beta", symbol: "ethusdt" });
const params = new URLSearchParams(path.split("?")[1]);
assert.equal(path.startsWith("/live-trading?"), true);
assert.equal(params.get("duration"), "30m");
assert.equal(params.get("strategyKey"), "factor beta");
assert.equal(params.get("symbol"), "ETHUSDT");

assert.deepEqual(slotIdentity(rows[0]), {
  duration: "30m",
  strategyKey: "factor_beta",
  symbol: "ETHUSDT",
});
assert.throws(() => liveTradingPath({ duration: "10m", symbol: "BTCUSDT" }), /missing strategyKey/);
