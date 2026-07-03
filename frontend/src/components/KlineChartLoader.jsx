/**
 * K线图懒加载包装器
 */
import { lazy, Suspense } from 'react';
import './KlineChartLoader.css';

const KlineChartImpl = lazy(() => import('./KlineChart'));

export default function KlineChart(props) {
  return (
    <Suspense fallback={<ChartSkeleton />}>
      <KlineChartImpl {...props} />
    </Suspense>
  );
}

function ChartSkeleton() {
  return (
    <div className="chart-skeleton">
      <div className="chart-skeleton-header">
        <div className="skeleton-pill" />
        <div className="skeleton-pill" />
        <div className="skeleton-pill" />
      </div>
      <div className="chart-skeleton-body">
        <div className="skeleton-chart-area">
          {/* 模拟K线 */}
          {Array.from({ length: 20 }).map((_, i) => (
            <div
              key={i}
              className="skeleton-candle"
              style={{
                left: `${(i / 20) * 100}%`,
                height: `${30 + Math.random() * 40}%`,
                bottom: `${10 + Math.random() * 30}%`,
              }}
            />
          ))}
        </div>
      </div>
      <div className="chart-skeleton-footer">
        <div className="skeleton-text" />
      </div>
    </div>
  );
}
