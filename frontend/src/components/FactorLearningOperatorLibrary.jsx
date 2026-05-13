import {
  categoryLabel,
  operatorLabel,
} from "../utils/factorLearningLabels";

const OPERATOR_PREVIEW_LIMIT = 64;

export default function FactorLearningOperatorLibrary({ operators }) {
  const grouped = groupOperators(operators);
  return (
    <div className="factor-learning-operators">
      <div className="factor-learning-title">
        <h3>运算符库</h3>
        <span>{operators.length} 个</span>
      </div>
      <div className="factor-learning-operator-grid">
        {grouped.slice(0, OPERATOR_PREVIEW_LIMIT).map(([category, items]) => (
          <section key={category} className="factor-learning-operator-family">
            <h4>{categoryLabel(category)}</h4>
            <div>
              {items.map((item) => (
                <span key={item.name} title={`${item.signature} · ${item.purpose}`}>
                  {operatorLabel(item.name)}
                </span>
              ))}
            </div>
          </section>
        ))}
      </div>
    </div>
  );
}

function groupOperators(operators) {
  const grouped = new Map();
  for (const operator of operators) {
    const key = operator.category || "other";
    grouped.set(key, [...(grouped.get(key) || []), operator]);
  }
  return Array.from(grouped.entries());
}
