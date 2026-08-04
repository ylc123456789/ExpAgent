"""Unit tests for Pydantic data models."""

from __future__ import annotations

import pytest

from experiment_designer.models import (
    CodingTask,
    ComputeBudget,
    DatasetSpec,
    DesignInput,
    ExistingAssets,
    ExistingMethod,
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


class TestDesignInput:
    """Tests for the input model."""

    def test_minimal_input(self) -> None:
        inp = DesignInput(
            research_idea="Test idea",
            target_task="image classification",
        )
        assert inp.research_idea == "Test idea"
        assert inp.target_task == "image classification"
        assert inp.compute_budget.gpu == "CPU only"
        assert inp.constraints == []
        assert inp.existing_assets.implemented_methods == []

    def test_full_input(self) -> None:
        inp = DesignInput(
            research_idea="Test attention mechanism on CIFAR-10",
            target_task="image classification",
            compute_budget=ComputeBudget(gpu="RTX 4090", max_runtime="2 hours", max_trials=5),
            constraints=["no large dataset downloads"],
            literature_context=["ViT paper shows...", "SENet is standard baseline"],
            existing_assets=ExistingAssets(
                implemented_methods=[
                    ExistingMethod(name="my_attention", location="/path/to/models/attention.py")
                ],
                available_datasets=["CIFAR-10"],
                known_baselines=["ResNet-18 on ImageNet"],
            ),
        )
        assert inp.compute_budget.max_trials == 5
        assert len(inp.literature_context) == 2
        assert len(inp.existing_assets.implemented_methods) == 1

    def test_defaults(self) -> None:
        inp = DesignInput(
            research_idea="Test",
            target_task="classification",
        )
        assert inp.compute_budget == ComputeBudget()
        assert inp.constraints == []
        assert inp.literature_context == []


class TestExperimentPlan:
    """Tests for the output model."""

    def test_valid_plan(self) -> None:
        plan = ExperimentPlan(
            goal=ResearchGoal(
                summary="Verify attention mechanism",
                hypothesis="Proposed attention outperforms baseline by >2%",
                success_criteria=["accuracy > baseline + 2%"],
            ),
            experiment_matrix=ExperimentMatrix(
                datasets=[DatasetSpec(name="CIFAR-10", rationale="Quick validation")],
                methods=[
                    MethodSpec(name="proposed", type="new_method", implementation_status="needs_code",
                               rationale="Core method"),
                    MethodSpec(name="resnet18", type="baseline", implementation_status="needs_repro",
                               rationale="Standard baseline"),
                ],
                metrics=[MetricSpec(name="accuracy", rationale="Primary metric")],
            ),
            tasks=TaskBundle(
                coding_tasks=[
                    CodingTask(
                        id="code_001", repo_path="/path/to/repo",
                        task_goal="Implement proposed attention",
                        rationale="New code needed",
                    )
                ],
            ),
            risks=[Risk(description="May not generalize", mitigation="Test on multiple datasets")],
        )
        assert plan.goal.hypothesis
        assert len(plan.experiment_matrix.methods) == 2
        assert len(plan.tasks.coding_tasks) == 1

    def test_empty_task_lists(self) -> None:
        plan = ExperimentPlan(
            goal=ResearchGoal(summary="Test", hypothesis="H"),
            experiment_matrix=ExperimentMatrix(),
            tasks=TaskBundle(),
            risks=[],
        )
        assert plan.tasks.coding_tasks == []
        assert plan.tasks.repro_tasks == []
        assert plan.tasks.run_tasks == []

    def test_method_types(self) -> None:
        """All three method types should be accepted."""
        for mtype in ("new_method", "baseline", "ablation"):
            m = MethodSpec(name="test", type=mtype, implementation_status="needs_code")
            assert m.type == mtype

    def test_invalid_method_type(self) -> None:
        with pytest.raises(ValueError):
            MethodSpec(name="test", type="invalid_type", implementation_status="needs_code")


class TestCodingTask:
    """Tests for CodingTask alignment with CodeTaskSpec."""

    def test_minimal_coding_task(self) -> None:
        t = CodingTask(
            id="code_001",
            repo_path="/path/to/repo",
            task_goal="Implement X",
        )
        assert t.id == "code_001"
        assert t.constraints == []
        assert t.verify_commands == []
        assert t.expected_artifacts == []

    def test_full_coding_task(self) -> None:
        t = CodingTask(
            id="code_002",
            repo_path="/path/to/repo",
            task_goal="Implement proposed method",
            constraints=["Do not modify training entry"],
            verify_commands=["pytest tests/", "python -c 'import model'"],
            expected_artifacts=["patch.diff", "report.md"],
            rationale="Core method needs implementation",
        )
        assert len(t.constraints) == 1
        assert len(t.verify_commands) == 2
        assert len(t.expected_artifacts) == 2


class TestReproTask:
    """Tests for ReproTask alignment with ReproAgent's ReproTask."""

    def test_minimal_repro_task(self) -> None:
        t = ReproTask(
            id="repro_001",
            paper_url="https://arxiv.org/abs/1709.01507",
            repo_url="https://github.com/example/repo",
            experiment_goal="Reproduce results on CIFAR-10",
        )
        assert t.paper_url
        assert t.repo_url
        assert t.expected_metrics == []

    def test_with_budget(self) -> None:
        t = ReproTask(
            id="repro_001",
            paper_url="https://arxiv.org/abs/1709.01507",
            repo_url="https://github.com/example/repo",
            experiment_goal="Reproduce results",
            compute_budget=ComputeBudget(gpu="RTX 4090", max_runtime="30 min", max_trials=3),
        )
        assert t.compute_budget is not None
        assert t.compute_budget.gpu == "RTX 4090"


class TestRunTask:
    """Tests for RunTask."""

    def test_gpu_flag(self) -> None:
        t = RunTask(id="run_001", command_goal="Run training", requires_gpu=True)
        assert t.requires_gpu is True

    def test_cpu_only(self) -> None:
        t = RunTask(id="run_001", command_goal="Compute stats")
        assert t.requires_gpu is False


class TestValidationResult:
    """Tests for ValidationResult."""

    def test_ok(self) -> None:
        from experiment_designer.models import ValidationResult
        vr = ValidationResult(status="ok", issues=[])
        assert vr.status == "ok"

    def test_needs_revision(self) -> None:
        from experiment_designer.models import ValidationResult
        vr = ValidationResult(status="needs_revision", issues=["missing baseline", "no risk"])
        assert len(vr.issues) == 2
