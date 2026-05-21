import { useEffect, useMemo, useState } from "react";
import { formatNum, formatPct } from "./factorDisplayUtils";
import "./FactorHighWinrateCard.css";

const MEMBERS_PAGE_SIZE = 4;

export default function FactorHighWinrateCard({ combo }) {
  if (!combo?.available) {
    return (
      <aside className="factor-hwr-card card-surface">
        <header className="factor-hwr-head">
          <span className="section-kicker">高胜率目标组合</span>
          <span className="factor-hwr-tag factor-hwr-tag-muted">暂无缓存</span>
        </header>
        <p className="factor-hwr-empty">当前交易对/周期尚无高胜率组合排名缓存，请先在后台完成组合搜索。</p>
      </aside>
    );
  }

  const members = combo.members ?? [];
  const [memberPage, setMemberPage] = useState(1);
  const memberPageCount = Math.max(1, Math.ceil(members.length / MEMBERS_PAGE_SIZE));

  useEffect(() => {
    setMemberPage(1);
  }, [combo.factorName, members.length]);

  useEffect(() => {
    if (memberPage > memberPageCount) setMemberPage(memberPageCount);
  }, [memberPage, memberPageCount]);

  const memberSlice = useMemo(() => {
    const start = (memberPage - 1) * MEMBERS_PAGE_SIZE;
    return members.slice(start, start + MEMBERS_PAGE_SIZE);
  }, [memberPage, members]);

  return (
    <aside className="factor-hwr-card card-surface">
      <header className="factor-hwr-head">
        <div>
          <span className="section-kicker">高胜率目标组合</span>
          <strong className="factor-hwr-title">(组合缓存)</strong>
        </div>
        <span className="factor-hwr-tag">可用</span>
      </header>
      <dl className="factor-hwr-meta">
        <div>
          <dt>组合名称</dt>
          <dd className="factor-hwr-name">{combo.displayName}</dd>
        </div>
        <div>
          <dt>组合ID</dt>
          <dd>
            <code className="factor-hwr-code">{combo.factorName}</code>
          </dd>
        </div>
      </dl>
      {members.length ? (
        <>
          <p className="factor-hwr-members-title">成员 ({members.length})</p>
          <ul className="factor-hwr-members">
            {memberSlice.map((member) => (
              <li key={member.name || member}>
                {member.displayName || member.name || member}
              </li>
            ))}
          </ul>
          {members.length > MEMBERS_PAGE_SIZE ? (
            <MembersPagination
              page={memberPage}
              pageCount={memberPageCount}
              onPageChange={setMemberPage}
            />
          ) : null}
        </>
      ) : null}
      <dl className="factor-hwr-metrics">
        <Metric label="胜率" value={formatPct(combo.winRate, 1)} tone="positive" />
        <Metric label="日均单量" value={formatNum(combo.avgTradesPerDay, 1)} />
        <Metric label="综合评分" value={formatNum(combo.factorScore, 1)} tone="positive" />
        <Metric label="盈亏比" value={formatNum(combo.profitFactor, 2)} tone="positive" />
        <Metric label="最大回撤" value={formatPct(combo.maxDrawdown, 1)} tone="negative" />
        <Metric label="样本期数" value={combo.totalPeriods ?? "—"} />
      </dl>
      <button type="button" className="factor-hwr-detail-btn">
        查看详情
      </button>
    </aside>
  );
}

function MembersPagination({ page, pageCount, onPageChange }) {
  return (
    <nav className="factor-hwr-members-pagination" aria-label="组合成员分页">
      <button type="button" disabled={page <= 1} onClick={() => onPageChange(page - 1)} aria-label="上一页">
        ‹
      </button>
      <span>
        {page} / {pageCount}
      </span>
      <button
        type="button"
        disabled={page >= pageCount}
        onClick={() => onPageChange(page + 1)}
        aria-label="下一页"
      >
        ›
      </button>
    </nav>
  );
}

function Metric({ label, value, tone }) {
  return (
    <div className={`factor-hwr-metric${tone ? ` is-${tone}` : ""}`}>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}
