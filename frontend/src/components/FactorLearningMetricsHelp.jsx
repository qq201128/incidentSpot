/** 因子学习面板：页内指标说明（与后端 factor_learning_* / factor_combo_monitor 语义一致） */
export default function FactorLearningMetricsHelp() {
  return (
    <details className="factor-learning-metrics-help">
      <summary>各指标含义</summary>
      <div className="factor-learning-metrics-help-body">
        <section>
          <h4>页头</h4>
          <ul>
            <li>
              <strong>结算样本</strong>：已结算、并参与「亏损模式」统计的预测条数。
            </li>
            <li>
              <strong>亏损模式</strong>：当前学习到的亏损特征模式条数（与下方「亏损特征」列表对应）。
            </li>
            <li>
              <strong>本地复盘</strong>：仅本地重算记忆与监控，不调大模型。
            </li>
            <li>
              <strong>联网挖掘</strong>：触发因子挖掘 Agent（可能排队）。
            </li>
          </ul>
        </section>
        <section>
          <h4>记忆网格（上排四格）</h4>
          <ul>
            <li>
              <strong>成功模式</strong>：历史上表现达标的因子，按「因子族 / 算子」归类。标题旁数字为列表条数；每行右侧整数为
              <strong>支持度</strong>：落入该类的因子个数（非百分比）。
            </li>
            <li>
              <strong>禁区</strong>：因子间 Spearman 相关绝对值较高的一簇（信息冗余）。右侧百分比为簇内<strong>平均绝对相关系数</strong>，越高越不宜在同一组合里堆叠。
            </li>
            <li>
              <strong>亏损特征</strong>：与亏损单显著相关的因子区间。右侧为命中该规则时的<strong>亏损占比</strong>。「暂无数据」表示未筛出足够显著的模式。
            </li>
            <li>
              <strong>自动权重</strong>：组合打分用的相对权重（约归一为 100%），会削弱命中亏损特征的因子。标题旁为有权重的因子数；列表最多展示 10 个。
            </li>
          </ul>
        </section>
        <section>
          <h4>状态区（下排四格）</h4>
          <ul>
            <li>
              <strong>挖掘因子库</strong>：已写入库的因子数量。列表右侧为因子在库中的<strong>历史胜率</strong>。
            </li>
            <li>
              <strong>实盘模拟监控</strong>：标题旁 <strong>正常 / 预警 / 样本少</strong>——样本不足 10 条为「样本少」；否则若触发告警则为「预警」。
            </li>
            <li>
              <strong>样本</strong>：用于监控的最近已结算多因子预测条数（上限 200）。
            </li>
            <li>
              <strong>成功率</strong>：上述样本中预测是否<strong>判对</strong>的比例（与系统内对错标记一致）。
            </li>
            <li>
              <strong>候选成功率</strong>：仅统计<strong>已通过质量过滤</strong>的模拟单中，预测判对的比例。
            </li>
            <li>
              <strong>连续亏损</strong>：从最新一条往历史数，连续预测错误的条数。
            </li>
            <li>
              <strong>低胜率告警</strong>：右侧 <strong>高/中/低</strong> 为严重级别。常见触发：整体成功率偏低、候选成功率偏低、连续亏损过多。
            </li>
            <li>
              <strong>解决方案</strong>：与告警关联的处理建议；右侧「执行」为操作提示，不会自动代你执行。
            </li>
          </ul>
        </section>
        <section>
          <h4>运算符库</h4>
          <ul>
            <li>按因子族分组展示可用<strong>算子</strong>（对行情列做变换的函数）。悬停在标签上可查看签名与用途（若接口提供）。</li>
            <li>
              <strong>周期</strong>（如 20 周期）：因子计算使用的回看 K 线根数，与当前所选周期一致。
            </li>
          </ul>
        </section>
      </div>
    </details>
  );
}
