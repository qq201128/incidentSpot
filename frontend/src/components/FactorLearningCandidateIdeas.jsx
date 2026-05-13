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
        <h3>候选挖掘因子</h3>
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
        {!ideas.length ? <p className="factor-learning-empty">暂无候选因子</p> : null}
      </div>
    </div>
  );
}

function candidateIdeaTitle(idea, index) {
  if (idea.displayNameZh) return idea.displayNameZh;
  if (idea.nameHint) return factorLabel(idea.nameHint);
  return `候选因子 ${index + 1}`;
}

function TagList({ items, empty, muted = false }) {
  const rows = Array.isArray(items) ? items.filter(Boolean) : [];
  return (
    <div className={`factor-learning-tags${muted ? " muted" : ""}`}>
      {rows.length ? rows.slice(0, 6).map((item) => <span key={item}>{item}</span>) : <span>{empty}</span>}
    </div>
  );
}
