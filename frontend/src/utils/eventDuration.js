/** 从事件起止时间推算合约时长（分钟），与持仓卡片「时长」口径一致 */
export function eventDurationMinutesFromWindow(item) {
  if (!item?.startTime || !item?.endTime) return null;
  const start = new Date(item.startTime).getTime();
  const end = new Date(item.endTime).getTime();
  if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) return null;
  return Math.round((end - start) / 60000);
}

/** 时长展示：与持仓卡片「时长」一致 */
export function contractDurationLabel(minutes) {
  if (minutes == null || !Number.isFinite(minutes)) return "未知";
  if (minutes === 1440) return "1天";
  return `${minutes}分钟`;
}
