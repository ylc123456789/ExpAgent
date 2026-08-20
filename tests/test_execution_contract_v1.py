"""Execution Contract v2 — ExpAgent scientific action graph.

Locks the ExpAgent side of the V2 contract (SCIENTIFIC_ORCHESTRATION_MAINLINE_REDESIGN.md):
recommended_actions are a logical graph of typed ScientificActions discriminated
on capability, with no physical execution fields.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiment_designer.controller.validator import validate_decision
from experiment_designer.models import (
    AnalyzeResultsAction,
    AskUserAction,
    EvidenceItem,
    ExecuteExperimentAction,
    ModifyCodeAction,
    ReproduceExperimentAction,
    ResultAnalysis,
    ScientificAction,
    ScientificConclusion,
    ScientificDecision,
    SearchLiteratureAction,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "execution_contract_v1"

ALL_CAPABILITIES = [
    "modify_code", "reproduce_experiment", "execute_experiment",
    "analyze_results", "search_literature", "ask_user",
]


def _make_decision(actions, *, analysis_required: bool = True) -> ScientificDecision:
    return ScientificDecision(
        summary="Logical action graph decision",
        confidence="medium",
        conclusion=ScientificConclusion(status="needs_more_experiments", rationale="Test rationale."),
        evidence=[EvidenceItem(source="reasoning", description="Test evidence.")],
        recommended_actions=actions,
        analysis_required=analysis_required,
        risks=["Test risk"],
    )


def _execute_action(action_id: str, **kw) -> ExecuteExperimentAction:
    return ExecuteExperimentAction(
        capability="execute_experiment",
        action_id=action_id,
        objective="Run the experiment",
        rationale="Produce raw metrics",
        success_criteria=["metric recorded"],
        expected_metrics=["accuracy"],
        **kw,
    )


def _analyze_action(action_id: str, **kw) -> AnalyzeResultsAction:
    return AnalyzeResultsAction(
        capability="analyze_results",
        action_id=action_id,
        objective="Interpret the results",
        rationale="Judge the hypothesis",
        **kw,
    )


# ── Fixtures ──────────────────────────────────────────────────────


def test_logical_action_graph_fixture_validates() -> None:
    data = json.loads((FIXTURE_DIR / "logical_action_graph.json").read_text(encoding="utf-8"))
    decision = ScientificDecision.model_validate(data)
    vr = validate_decision(decision)
    assert vr.status == "ok", vr.issues
    assert [a.capability for a in decision.recommended_actions] == [
        "modify_code", "execute_experiment", "analyze_results",
    ]


def test_odenet_fixture_is_execute_then_analyze() -> None:
    """The ODE-Net fixture yields execute_experiment -> analyze_results."""
    data = json.loads((FIXTURE_DIR / "odenet_plan.json").read_text(encoding="utf-8"))
    decision = ScientificDecision.model_validate(data)
    vr = validate_decision(decision)
    assert vr.status == "ok", vr.issues
    caps = [a.capability for a in decision.recommended_actions]
    assert caps == ["execute_experiment", "analyze_results"]
    analysis = decision.recommended_actions[1]
    assert analysis.depends_on == ["run_odenet"]


def test_fan_in_analysis_fixture_validates() -> None:
    data = json.loads((FIXTURE_DIR / "fan_in_analysis.json").read_text(encoding="utf-8"))
    decision = ScientificDecision.model_validate(data)
    vr = validate_decision(decision)
    assert vr.status == "ok", vr.issues
    analysis = decision.recommended_actions[-1]
    assert analysis.capability == "analyze_results"
    assert analysis.depends_on == ["run_baseline", "run_proposed"]


# ── Action graph invariants ───────────────────────────────────────


def test_depends_on_must_reference_earlier_action() -> None:
    decision = _make_decision([
        _analyze_action("analyze_1", depends_on=["run_1"]),
        _execute_action("run_1"),
    ])
    vr = validate_decision(decision)
    assert vr.status == "needs_revision"
    assert any("EARLIER" in i for i in vr.issues), vr.issues


def test_depends_on_unknown_action_rejected() -> None:
    decision = _make_decision([_analyze_action("analyze_1", depends_on=["does_not_exist"])])
    vr = validate_decision(decision)
    assert vr.status == "needs_revision"
    assert any("unknown action_id" in i for i in vr.issues), vr.issues


def test_empty_action_id_rejected() -> None:
    decision = _make_decision([
        ExecuteExperimentAction(
            capability="execute_experiment",
            action_id="",
            objective="Run",
            rationale="r",
            expected_metrics=["accuracy"],
        ),
    ])
    vr = validate_decision(decision)
    assert vr.status == "needs_revision"
    assert any("empty action_id" in i for i in vr.issues), vr.issues


def test_analyze_results_requires_dependency() -> None:
    decision = _make_decision([_analyze_action("analyze_1")])
    vr = validate_decision(decision)
    assert vr.status == "needs_revision"
    assert any("must depend on at least one experiment" in i for i in vr.issues), vr.issues


def test_execute_experiment_requires_metrics_or_criteria() -> None:
    decision = _make_decision([
        ExecuteExperimentAction(
            capability="execute_experiment",
            action_id="run_1",
            objective="Run",
            rationale="r",
        ),
    ])
    vr = validate_decision(decision)
    assert vr.status == "needs_revision"
    assert any("expected_metrics or success_criteria" in i for i in vr.issues), vr.issues


def test_analysis_required_covers_experiments() -> None:
    decision = _make_decision([_execute_action("run_1")], analysis_required=True)
    vr = validate_decision(decision)
    assert vr.status == "needs_revision"
    assert any("no analyze_results coverage" in i for i in vr.issues), vr.issues


def test_analysis_required_false_allows_uncovered_experiment() -> None:
    decision = _make_decision([_execute_action("run_1")], analysis_required=False)
    vr = validate_decision(decision)
    assert vr.status == "ok", vr.issues


def test_required_action_cannot_depend_on_optional() -> None:
    """A required action's hard dependencies must all be required.

    required-analyze depending on optional-experiment is a scheduler trap:
    the "optional" experiment is forced to execute (or the required
    analysis can never run). The graph is rejected so the model re-marks
    the chain consistently.
    """
    decision = _make_decision([
        _execute_action("run_1", required=False, depends_on=[]),
        _analyze_action("analyze_1", required=True, depends_on=["run_1"]),
    ])
    vr = validate_decision(decision)
    assert vr.status == "needs_revision"
    assert any("dependencies must all be required" in i for i in vr.issues), vr.issues


def test_consistent_requirement_chains_validate() -> None:
    all_required = _make_decision([
        _execute_action("run_1", required=True, depends_on=[]),
        _analyze_action("analyze_1", required=True, depends_on=["run_1"]),
    ])
    assert validate_decision(all_required).status == "ok"
    all_optional = _make_decision([
        _execute_action("run_1", required=False, depends_on=[]),
        _analyze_action("analyze_1", required=False, depends_on=["run_1"]),
    ])
    assert validate_decision(all_optional).status == "ok"


def test_required_defaults_to_true() -> None:
    assert _execute_action("run_1").required is True


def test_required_false_accepted() -> None:
    action = _execute_action("run_1", required=False)
    decision = _make_decision([action], analysis_required=False)
    vr = validate_decision(decision)
    assert vr.status == "ok"
    assert decision.recommended_actions[0].required is False


def test_terminal_result_analysis_allows_empty_future_actions_in_chinese() -> None:
    decision = ScientificDecision(
        summary="现有证据已经足以支持该结论",
        confidence="high",
        conclusion=ScientificConclusion(
            status="supported",
            rationale="实验结果一致支持假设，无需为了重复当前分析而创建新任务。",
        ),
        evidence=[EvidenceItem(source="artifact", description="指标达到预设标准。")],
        result_analysis=ResultAnalysis(summary="结果支持假设", findings=["指标达标"]),
        recommended_actions=[],
        risks=["样本规模有限"],
    )

    assert validate_decision(decision).status == "ok"


def test_terminal_result_analysis_rejects_required_future_work() -> None:
    decision = ScientificDecision(
        summary="The supplied results support the hypothesis.",
        confidence="high",
        conclusion=ScientificConclusion(status="supported", rationale="Evidence is sufficient."),
        evidence=[EvidenceItem(source="artifact", description="Metric met the target.")],
        result_analysis=ResultAnalysis(summary="Supported", findings=["Target met"]),
        recommended_actions=[_execute_action("repeat_current_analysis", required=True)],
        analysis_required=False,
        risks=["Bounded evidence"],
    )

    vr = validate_decision(decision)
    assert vr.status == "needs_revision"
    assert any("terminal supported/not_supported conclusion" in issue for issue in vr.issues)


def test_supersedes_action_ids_are_explicit_and_disjoint() -> None:
    valid = _make_decision([])
    valid.supersedes_action_ids = ["old_run"]
    assert validate_decision(valid).status == "ok"

    invalid = _make_decision([_execute_action("same")], analysis_required=False)
    invalid.supersedes_action_ids = ["same"]
    vr = validate_decision(invalid)
    assert vr.status == "needs_revision"
    assert any("cannot also be emitted" in issue for issue in vr.issues)


# ── Physical field boundary ───────────────────────────────────────


def test_no_physical_execution_fields() -> None:
    models = [
        ScientificDecision,
        ModifyCodeAction,
        ReproduceExperimentAction,
        ExecuteExperimentAction,
        AnalyzeResultsAction,
        SearchLiteratureAction,
        AskUserAction,
    ]
    for model in models:
        fields = set(model.model_fields)
        for physical in ("workspace_path", "copy_from", "external_repo_path", "env_name"):
            assert physical not in fields, f"{model.__name__} must not have '{physical}'"


# ── Schema + mock ─────────────────────────────────────────────────


def test_finish_schema_has_v2_capabilities() -> None:
    from experiment_designer.prompts import TOOLS

    finish = next(t["function"] for t in TOOLS if t["function"]["name"] == "finish")
    params = finish["parameters"]["properties"]
    action_schema = params["recommended_actions"]["items"]
    assert action_schema["properties"]["capability"]["enum"] == ALL_CAPABILITIES
    assert "analysis_required" in params
    assert "supersedes_action_ids" in params


def test_mock_output_follows_contract() -> None:
    from experiment_designer.llm import _make_mock_design_decision

    decision = _make_mock_design_decision()
    caps = [a["capability"] for a in decision["recommended_actions"]]
    assert caps == ["modify_code", "execute_experiment", "analyze_results"]
    run_action = next(a for a in decision["recommended_actions"] if a["capability"] == "execute_experiment")
    assert run_action["expected_metrics"] or run_action["success_criteria"]
