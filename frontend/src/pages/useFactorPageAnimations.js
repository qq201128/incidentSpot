import { animate, createScope, createTimeline, stagger } from "animejs";
import { useEffect, useRef } from "react";

const ANIMATION_SCOPE_METHODS = {
  enterPage: "enterPage",
  refreshList: "refreshList",
  refreshRanking: "refreshRanking",
};
const PAGE_ENTRY_SELECTOR = "[data-factor-motion]";
const TOOLBAR_ITEM_SELECTOR = ".factors-toolbar-row > *";
const FACTOR_LIST_ROW_SELECTOR = ".factors-list-panel .factors-table tbody tr";
const RANK_HINT_SELECTOR = ".factors-rank-hint";
const RANKING_ROW_SELECTOR = ".factors-ranking-table tbody tr";

export function useFactorPageAnimations({ listKey, pageRef, rankingKey }) {
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

}

function registerFactorPageAnimations(scope) {
  scope.add(ANIMATION_SCOPE_METHODS.enterPage, animatePageEntry);
  scope.add(ANIMATION_SCOPE_METHODS.refreshList, animateFactorListRefresh);
  scope.add(ANIMATION_SCOPE_METHODS.refreshRanking, animateRankingRefresh);
}

function animatePageEntry() {
  const timeline = createTimeline({
    defaults: {
      duration: 520,
      ease: "outCubic",
      composition: "replace",
    },
  });
  if (hasTargets(PAGE_ENTRY_SELECTOR)) {
    timeline.add(PAGE_ENTRY_SELECTOR, {
      opacity: [0, 1],
      y: [18, 0],
      delay: stagger(70),
    });
  }
  if (hasTargets(TOOLBAR_ITEM_SELECTOR)) {
    timeline.add(
      TOOLBAR_ITEM_SELECTOR,
      {
        opacity: [0, 1],
        x: [-10, 0],
        delay: stagger(35),
        duration: 360,
      },
      "<<+=160",
    );
  }
}

function animateFactorListRefresh() {
  if (!hasTargets(FACTOR_LIST_ROW_SELECTOR)) return;
  animate(FACTOR_LIST_ROW_SELECTOR, {
    opacity: [0, 1],
    x: [-10, 0],
    duration: 300,
    delay: stagger(18),
    ease: "outQuad",
    composition: "replace",
  });
}

function animateRankingRefresh() {
  if (hasTargets(RANK_HINT_SELECTOR)) animate(RANK_HINT_SELECTOR, {
    opacity: [0.52, 1],
    color: ["#e8bf63", "#9aa9a5"],
    duration: 420,
    ease: "outQuad",
    composition: "replace",
  });
  if (!hasTargets(RANKING_ROW_SELECTOR)) return;
  animate(RANKING_ROW_SELECTOR, {
    opacity: [0, 1],
    x: [12, 0],
    duration: 320,
    delay: stagger(14),
    ease: "outQuad",
    composition: "replace",
  });
}

function hasTargets(selector) {
  return document.querySelector(selector) !== null;
}
