import {
  columnLabels,
  factorLabel,
  operatorTraceLabel,
} from "../utils/factorLearningLabels";

export default function FactorLearningCandidateIdeas({ memory }) {
  const ideas = memory?.llmAgent?.review?.factorMiningPlan?.candidateFactorIdeas || [];
  return (
    <div className="factor-learning-section">
      <div className="factor-learning-title">
        <h3>Agent单因子候选想法</h3>
        <span>{ideas.length} 项</span>
      </div>
      <div className="factor-learning-candidates">
        {ideas.map((idea, index) => (
          <article key={`${idea.nameHint || "idea"}-${index}`} className="factor-learning-candidate">
            <div className="factor-learning-candidate-top">
              <strong>{candidateIdeaTitle(idea, index)}</strong>
              <span>{operatorTraceLabel(idea.operatorTrace).join(" · ") || "无算子链"}</span>
            </div>
            <code>{idea.formulaHint || "—"}</code>
            <p>{idea.rationaleZh || idea.rationale || "—"}</p>
            <TagList items={columnLabels(idea.requiredColumns)} empty="无列要求" />
            <TagList items={idea.validationChecks} empty="无验证项" muted />
          </article>
        ))}
        {!ideas.length ? <EmptyCandidateState agent={memory?.llmAgent} /> : null}
      </div>
    </div>
  );
}

function EmptyCandidateState({ agent }) {
  return (
    <div className="factor-learning-empty factor-learning-candidate-empty">
      <strong>{candidateEmptyTitle(agent)}</strong>
      <span>{candidateEmptyDetail(agent)}</span>
    </div>
  );
}

function candidateIdeaTitle(idea, index) {
  if (idea.displayNameZh) return idea.displayNameZh;
  if (idea.nameHint) return factorLabel(idea.nameHint);
  return `候选因子 ${index + 1}`;
}

function candidateEmptyTitle(agent) {
  if (agent?.status === "failed") return "联网挖掘失败";
  if (agent?.status === "pending") return "联网挖掘排队中";
  if (!agent?.review) return "暂无Agent候选想法";
  return "Agent未返回候选想法";
}

function candidateEmptyDetail(agent) {
  if (agent?.error) return agent.error;
  if (agent?.status === "pending") return "等待后台写回 llmAgent.review.factorMiningPlan.candidateFactorIdeas";
  if (!agent?.review) return "这个区块只显示联网Agent提出的单因子研究想法，不显示组合因子库。";
  return "联网Agent完成了复盘，但 candidateFactorIdeas 数组为空。";
}

function TagList({ items, empty, muted = false }) {
  const rows = Array.isArray(items) ? items.filter(Boolean) : [];
  return (
    <div className={`factor-learning-tags${muted ? " muted" : ""}`}>
      {rows.length
        ? rows.map((item, index) => (
            <span key={`${index}-${item}`}>{item}</span>
          ))
        : <span>{empty}</span>}
    </div>
  );
}
