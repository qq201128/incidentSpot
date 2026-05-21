export function chartIntervalToMinutes(interval) {
  switch (interval) {
    case "30m":
      return 30;
    case "60m":
      return 60;
    case "1d":
      return 1440;
    default:
      return 10;
  }
}

export function predictDurationKey(minutes) {
  if (minutes === 10) return "10m";
  if (minutes === 30) return "30m";
  if (minutes === 60) return "60m";
  if (minutes === 1440) return "1d";
  return "10m";
}

export function durationKeyFromChartInterval(interval) {
  return predictDurationKey(chartIntervalToMinutes(interval));
}
