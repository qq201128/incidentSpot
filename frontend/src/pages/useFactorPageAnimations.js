import { animate, createScope, createTimeline, stagger } from "animejs";
import { useEffect, useRef } from "react";

const ANIMATION_SCOPE_METHODS = {
  enterPage: "enterPage",
  refreshList: "refreshList",
  refreshRanking: "refreshRanking",
  showDetail: "showDetail",
  showBacktest: "showBacktest",
};

export function useFactorPageAnimations({ backtestKey, detailKey, listKey, pageRef, rankingKey }) {
  const animationScopeRef = useRef(null);

  useEffect(() => {
    const scope = createScope({ root: pageRef }).add((self) => {
      registerFactorPageAnimations(self);
    });
    animationScopeRef.current = scope;
    scope.methods[ANIMATION_SCOPE_METHODS.enterPage]();
    return () => {
      animationScopeRef.current = null;
      scope.revert();
    };
  }, [pageRef]);

  useEffect(() => {
    animationScopeRef.current?.methods[ANIMATION_SCOPE_METHODS.refreshList]();
  }, [listKey]);

  useEffect(() => {
    animationScopeRef.current?.methods[ANIMATION_SCOPE_METHODS.refreshRanking]();
  }, [rankingKey]);

  useEffect(() => {
    animationScopeRef.current?.methods[ANIMATION_SCOPE_METHODS.showDetail]();
  }, [detailKey]);

  useEffect(() => {
    animationScopeRef.current?.methods[ANIMATION_SCOPE_METHODS.showBacktest]();
  }, [backtestKey]);
}

function registerFactorPageAnimations(scope) {
  scope.add(ANIMATION_SCOPE_METHODS.enterPage, animatePageEntry);
  scope.add(ANIMATION_SCOPE_METHODS.refreshList, animateFactorListRefresh);
  scope.add(ANIMATION_SCOPE_METHODS.refreshRanking, animateRankingRefresh);
  scope.add(ANIMATION_SCOPE_METHODS.showDetail, animateDetailRefresh);
  scope.add(ANIMATION_SCOPE_METHODS.showBacktest, animateBacktestRefresh);
}

function animatePageEntry() {
  createTimeline({
    defaults: {
      duration: 520,
      ease: "outCubic",
      composition: "replace",
    },
  })
    .add("[data-factor-motion]", {
      opacity: [0, 1],
      y: [18, 0],
      delay: stagger(70),
    })
    .add(
      ".factors-toolbar-row > *",
      {
        opacity: [0, 1],
        x: [-10, 0],
        delay: stagger(35),
        duration: 360,
      },
      "<<+=160",
    );
}

function animateFactorListRefresh() {
  animate(".factors-list-panel .factors-table tbody tr", {
    opacity: [0, 1],
    x: [-10, 0],
    duration: 300,
    delay: stagger(18),
    ease: "outQuad",
    composition: "replace",
  });
}

function animateRankingRefresh() {
  animate(".factors-rank-hint", {
    opacity: [0.52, 1],
    color: ["#e8bf63", "#9aa9a5"],
    duration: 420,
    ease: "outQuad",
    composition: "replace",
  });
  animate(".factors-ranking-table tbody tr", {
    opacity: [0, 1],
    x: [12, 0],
    duration: 320,
    delay: stagger(14),
    ease: "outQuad",
    composition: "replace",
  });
}

function animateDetailRefresh() {
  animate(".factors-detail-panel .factors-section-head, .factors-dl > div, .factors-placeholder", {
    opacity: [0, 1],
    y: [8, 0],
    duration: 320,
    delay: stagger(34),
    ease: "outQuad",
    composition: "replace",
  });
}

function animateBacktestRefresh() {
  animate(".factors-backtest-block button, .factors-metrics > div, .factors-backtest-block .factors-error", {
    opacity: [0, 1],
    y: [6, 0],
    duration: 280,
    delay: stagger(22),
    ease: "outQuad",
    composition: "replace",
  });
}
