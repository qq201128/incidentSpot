/** 与后端 ALLOWED_INTERVALS 一致 */
export const KLINE_INTERVAL_OPTIONS = Object.freeze([
  { label: "10分", value: "10m" },
  { label: "30分", value: "30m" },
  { label: "1小时", value: "60m" },
  { label: "1天", value: "1d" },
]);

export function intervalLabel(value) {
  return KLINE_INTERVAL_OPTIONS.find((item) => item.value === value)?.label ?? value;
}
