"""Tests for the deterministic safety-net validator."""

from __future__ import annotations

import pytest

from experiment_designer.models import (
    CodingTask,
    DatasetSpec,
    ExperimentMatrix,
    ExperimentPlan,
    MethodSpec,
    MetricSpec,
    ReproTask,
    ResearchGoal,
    Risk,
    RunTask,
    TaskBundle,
)
from experiment_designer.controller.validator import validate


def _make_plan(**overrides) -> ExperimentPlan:
    """Build a minimal valid plan, then override specific fields."""
    from experiment_designer.models import AnalysisPlan

    plan = ExperimentPlan(
        version=1,
        goal=ResearchGoal(
            summary="Test experiment",
            hypothesis="Proposed method A outperforms baseline B on dataset D by metric M",
            success_criteria=["Metric M improves by >= 2%"],
        ),
        experiment_matrix=ExperimentMatrix(
            datasets=[DatasetSpec(name="CIFAR-10", split="standard", rationale="Lightweight")],
            methods=[
                MethodSpec(name="proposed_A", type="new_method", implementation_status="needs_code",
                           rationale="Core method to test"),
                MethodSpec(name="baseline_B", type="baseline", implementation_status="needs_repro",
                           rationale="Standard comparison"),
            ],
            metrics=[MetricSpec(name="accuracy", rationale="Primary metric")],
        ),
        tasks=TaskBundle(
            coding_tasks=[
                CodingTask(
                    id="code_001", workspace_path="/path/to/repo",
                    task_goal="Implement proposed_A", rationale="New code needed",
                )
            ],
        ),
        analysis_plan=AnalysisPlan(
            comparisons=["proposed_A vs baseline_B"],
            plots=["accuracy curve"],
        ),
        risks=[Risk(description="May not generalize", mitigation="Test on multiple datasets")],
    )
    for key, value in overrides.items():
        setattr(plan, key, value)
    return plan


class TestValidationPasses:
    """Cases that should pass validation."""

    def test_valid_plan_passes(self) -> None:
        plan = _make_plan()
        vr = validate(plan)
        assert vr.status == "ok", f"Expected ok but got issues: {vr.issues}"

    def test_with_repro_task(self) -> None:
        plan = _make_plan()
        plan.tasks.repro_tasks = [
            ReproTask(
                id="repro_001",
                paper_url="https://example.com/paper",
                repo_url="https://github.com/example/repo",
                experiment_goal="Reproduce baseline",
                rationale="Need baseline results",
            )
        ]
        vr = validate(plan)
        assert vr.status == "ok"

    def test_with_run_task(self) -> None:
        plan = _make_plan()
        plan.tasks.run_tasks = [
            RunTask(id="run_001", command_goal="Run experiment", rationale="Need results")
        ]
        vr = validate(plan)
        assert vr.status == "ok"

    def test_with_ablation(self) -> None:
        plan = _make_plan()
        plan.experiment_matrix.methods.append(
            MethodSpec(name="ablation_no_attention", type="ablation",
                       implementation_status="existing", rationale="Isolate attention effect")
        )
        vr = validate(plan)
        assert vr.status == "ok"


class TestValidationFails:
    """Cases that should fail validation."""

    def test_empty_hypothesis(self) -> None:
        plan = _make_plan()
        plan.goal.hypothesis = ""
        vr = validate(plan)
        assert vr.status == "needs_revision"
        assert any("hypothesis" in issue.lower() for issue in vr.issues)

    def test_short_hypothesis(self) -> None:
        plan = _make_plan()
        plan.goal.hypothesis = "Too short"
        vr = validate(plan)
        assert vr.status == "needs_revision"
        assert any("short" in issue.lower() for issue in vr.issues)

    def test_empty_success_criteria(self) -> None:
        plan = _make_plan()
        plan.goal.success_criteria = []
        vr = validate(plan)
        assert vr.status == "needs_revision"
        assert any("success_criteria" in issue.lower() for issue in vr.issues)

    def test_no_datasets(self) -> None:
        plan = _make_plan()
        plan.experiment_matrix.datasets = []
        vr = validate(plan)
        assert vr.status == "needs_revision"
        assert any("dataset" in issue.lower() for issue in vr.issues)

    def test_no_baseline(self) -> None:
        plan = _make_plan()
        plan.experiment_matrix.methods = [
            MethodSpec(name="proposed", type="new_method", implementation_status="needs_code",
                       rationale="Only method")
        ]
        vr = validate(plan)
        assert vr.status == "needs_revision"
        assert any("baseline" in issue.lower() for issue in vr.issues)

    def test_no_proposed_method(self) -> None:
        plan = _make_plan()
        plan.experiment_matrix.methods = [
            MethodSpec(name="baseline", type="baseline", implementation_status="needs_repro",
                       rationale="Only baseline")
        ]
        vr = validate(plan)
        assert vr.status == "needs_revision"
        assert any("new_method" in issue.lower() for issue in vr.issues)

    def test_no_metrics(self) -> None:
        plan = _make_plan()
        plan.experiment_matrix.metrics = []
        vr = validate(plan)
        assert vr.status == "needs_revision"
        assert any("metric" in issue.lower() for issue in vr.issues)

    def test_no_risks(self) -> None:
        plan = _make_plan()
        plan.risks = []
        vr = validate(plan)
        assert vr.status == "needs_revision"
        assert any("risk" in issue.lower() for issue in vr.issues)

    def test_no_rationale_on_method(self) -> None:
        plan = _make_plan()
        plan.experiment_matrix.methods[0].rationale = ""
        vr = validate(plan)
        assert vr.status == "needs_revision"
        assert any("rationale" in issue.lower() for issue in vr.issues)

    def test_no_rationale_on_coding_task(self) -> None:
        plan = _make_plan()
        plan.tasks.coding_tasks[0].rationale = ""
        vr = validate(plan)
        assert vr.status == "needs_revision"
        assert any("rationale" in issue.lower() for issue in vr.issues)

    def test_empty_coding_task_goal(self) -> None:
        plan = _make_plan()
        plan.tasks.coding_tasks[0].task_goal = ""
        vr = validate(plan)
        assert vr.status == "needs_revision"
        assert any("task_goal" in issue.lower() for issue in vr.issues)

    def test_empty_workspace_path(self) -> None:
        plan = _make_plan()
        plan.tasks.coding_tasks[0].workspace_path = ""
        plan.tasks.coding_tasks[0].task_goal = ""
        vr = validate(plan)
        assert vr.status == "needs_revision"
        assert any("workspace_path" in issue.lower() or "task_goal" in issue.lower() for issue in vr.issues)

    def test_empty_repro_paper_url(self) -> None:
        plan = _make_plan()
        plan.tasks.repro_tasks = [
            ReproTask(id="r1", paper_url="", repo_url="https://example.com",
                      experiment_goal="test", rationale="test")
        ]
        vr = validate(plan)
        assert vr.status == "needs_revision"
        assert any("paper_url" in issue.lower() for issue in vr.issues)

    def test_completely_empty_tasks(self) -> None:
        plan = _make_plan()
        plan.tasks = TaskBundle()  # No tasks at all
        vr = validate(plan)
        assert vr.status == "needs_revision"
        assert any("task" in issue.lower() for issue in vr.issues)

    def test_empty_risk_description(self) -> None:
        plan = _make_plan()
        plan.risks = [Risk(description="", mitigation="do something")]
        vr = validate(plan)
        assert vr.status == "needs_revision"
        assert any("description" in issue.lower() for issue in vr.issues)
