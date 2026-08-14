"""Execution Contract v1 — ExpAgent logical action graph.

Locks the ExpAgent side of the frozen cross-module contract
(EXECUTION_CONTRACT_V1.md): scientific intent expressed as a logical action
graph (action_id / depends_on / project_ref / workspace_intent), never as
physical execution paths.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiment_designer.controller.validator import validate_decision
from experiment_designer.models import (
    EvidenceItem,
    RecommendedAction,
    ScientificConclusion,
    ScientificDecision,
    SuggestedPlan,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "execution_contract_v1"


def _make_decision(actions: list[RecommendedAction]) -> ScientificDecision:
    return ScientificDecision(
        summary="Logical action graph decision",
        confidence="medium",
        conclusion=ScientificConclusion(status="needs_more_experiments", rationale="Test rationale."),
        evidence=[EvidenceItem(source="reasoning", description="Test evidence.")],
        recommended_actions=actions,
        risks=["Test risk"],
    )


def _coding_action(action_id: str, **kw) -> RecommendedAction:
    return RecommendedAction(
        priority="high",
        type="coding_task",
        rationale="Implement the patch",
        action_id=action_id,
        plan=SuggestedPlan(kind="coding_task", task_goal="Patch the loop"),
        **kw,
    )


def _run_action(action_id: str, **kw) -> RecommendedAction:
    return RecommendedAction(
        priority="high",
        type="run_task",
        rationale="Run after the patch",
        action_id=action_id,
        plan=SuggestedPlan(kind="run_task", command_goal="Run validation"),
        **kw,
    )


def test_fixture_roundtrip_and_validates() -> None:
    """The canonical fixture loads, validates, and round-trips intact."""
    path = FIXTURE_DIR / "logical_action_graph.json"
    data = json.loads(path.read_text(encoding="utf-8"))

    decision = ScientificDecision.model_validate(data)
    vr = validate_decision(decision)
    assert vr.status == "ok", f"Fixture should validate: {vr.issues}"

    assert decision.recommended_actions[0].action_id == "patch_training_loop"
    assert decision.recommended_actions[1].depends_on == ["patch_training_loop"]
    assert decision.recommended_actions[1].workspace_intent == "shared"
    assert decision.recommended_actions[1].project_ref == "my_research"


def test_workspace_intent_accepts_valid_values() -> None:
    """workspace_intent accepts shared / isolated / empty."""
    for value in ("shared", "isolated", ""):
        action = _run_action("run_1", workspace_intent=value)
        decision = _make_decision([action])
        assert decision.recommended_actions[0].workspace_intent == value


def test_workspace_intent_rejects_invalid_value() -> None:
    """workspace_intent must be one of shared / isolated / empty."""
    with pytest.raises(ValueError):
        _run_action("run_1", workspace_intent="bogus")


def test_depends_on_must_reference_earlier_action() -> None:
    """A dependency on a later action is rejected (acyclic by construction)."""
    # run action (index 0) depends on coding action (index 1) -> forward ref
    decision = _make_decision([
        _run_action("run_1", depends_on=["patch_1"]),
        _coding_action("patch_1"),
    ])
    vr = validate_decision(decision)
    assert vr.status == "needs_revision"
    assert any("EARLIER" in i for i in vr.issues), vr.issues


def test_depends_on_unknown_action_rejected() -> None:
    """A dependency on an undefined action_id is rejected."""
    decision = _make_decision([_run_action("run_1", depends_on=["does_not_exist"])])
    vr = validate_decision(decision)
    assert vr.status == "needs_revision"
    assert any("unknown action_id" in i for i in vr.issues), vr.issues


def test_expagent_has_no_physical_execution_fields() -> None:
    """ExpAgent models must not expose operator/CodingAgent physical fields."""
    for model in (ScientificDecision, RecommendedAction, SuggestedPlan):
        fields = set(model.model_fields)
        for physical in ("workspace_path", "copy_from", "external_repo_path", "env_name"):
            assert physical not in fields, f"{model.__name__} must not have '{physical}'"


def test_empty_action_id_rejected() -> None:
    """Every recommended action must have a non-empty action_id."""
    decision = _make_decision([
        RecommendedAction(
            priority="high",
            type="run_task",
            rationale="Missing action_id",
            plan=SuggestedPlan(kind="run_task", command_goal="Run"),
        ),
    ])
    vr = validate_decision(decision)
    assert vr.status == "needs_revision"
    assert any("empty action_id" in i for i in vr.issues), vr.issues


def test_finish_schema_has_workspace_intent() -> None:
    """The finish tool schema exposes workspace_intent so the LLM can emit it."""
    from experiment_designer.prompts import TOOLS

    finish = next(t["function"] for t in TOOLS if t["function"]["name"] == "finish")
    action_schema = finish["parameters"]["properties"]["recommended_actions"]["items"]
    assert "workspace_intent" in action_schema["properties"]
    assert action_schema["properties"]["workspace_intent"]["enum"] == ["shared", "isolated", ""]


def test_mock_output_follows_contract() -> None:
    """The deterministic mock decision carries a shared run task with workspace_intent."""
    from experiment_designer.llm import _make_mock_design_decision

    decision = _make_mock_design_decision()
    run_action = next(
        a for a in decision["recommended_actions"] if a["type"] == "run_task"
    )
    assert run_action["workspace_intent"] == "shared"
    assert run_action["depends_on"] and run_action["depends_on"][0] in {
        a["action_id"] for a in decision["recommended_actions"]
    }


def test_fan_in_analysis_fixture_roundtrip() -> None:
    """Two experiments + one result_analysis depending on both validates."""
    path = FIXTURE_DIR / "fan_in_analysis.json"
    data = json.loads(path.read_text(encoding="utf-8"))

    decision = ScientificDecision.model_validate(data)
    vr = validate_decision(decision)
    assert vr.status == "ok", f"Fan-in analysis fixture should validate: {vr.issues}"

    analysis = decision.recommended_actions[-1]
    assert analysis.type == "result_analysis"
    assert analysis.plan.kind == "result_analysis"
    assert analysis.depends_on == ["run_baseline", "run_proposed"]
    assert analysis.plan.task_goal == "Compare baseline vs proposed accuracy"


def test_required_defaults_to_true() -> None:
    """required defaults to True when not specified."""
    action = _run_action("run_1")
    assert action.required is True


def test_required_false_accepted() -> None:
    """An explicitly optional action may set required=False."""
    action = _run_action("run_1", required=False)
    decision = _make_decision([action])
    vr = validate_decision(decision)
    assert vr.status == "ok"
    assert decision.recommended_actions[0].required is False


def test_result_analysis_requires_task_goal() -> None:
    """result_analysis must carry a non-empty task_goal."""
    decision = _make_decision([
        RecommendedAction(
            priority="high",
            type="result_analysis",
            rationale="Analyze the results",
            action_id="analyze",
            plan=SuggestedPlan(kind="result_analysis", task_goal=""),
        ),
    ])
    vr = validate_decision(decision)
    assert vr.status == "needs_revision"
    assert any("task_goal" in i for i in vr.issues), vr.issues


def test_finish_schema_has_result_analysis_and_required() -> None:
    """The finish tool schema exposes result_analysis and required."""
    from experiment_designer.prompts import TOOLS

    finish = next(t["function"] for t in TOOLS if t["function"]["name"] == "finish")
    action_schema = finish["parameters"]["properties"]["recommended_actions"]["items"]
    assert "result_analysis" in action_schema["properties"]["type"]["enum"]
    assert "required" in action_schema["properties"]

    plan_schema = action_schema["properties"]["plan"]
    assert "result_analysis" in plan_schema["properties"]["kind"]["enum"]
