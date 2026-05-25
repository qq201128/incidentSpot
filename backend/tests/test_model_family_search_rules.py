from __future__ import annotations

from app.services.model_family_search_rules import model_family_training_rules


def test_model_family_rules_are_successive_halving() -> None:
    rules = model_family_training_rules("xgboost")

    assert rules["searchMode"] == "successive_halving"
    assert [stage["stage"] for stage in rules["successiveHalving"]] == ["coarse", "full", "walk_forward"]
    assert rules["successiveHalving"][0]["publishesArtifacts"] is False
    assert rules["successiveHalving"][-1]["publishesArtifacts"] is True
