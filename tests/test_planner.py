"""Integration tests for the planner (mock LLM and real LLM)."""

from __future__ import annotations

from pathlib import Path

import pytest

from experiment_designer.models import DesignInput, ComputeBudget
from experiment_designer.planner import plan, revise
from experiment_designer.validator import validate


class TestPlannerMock:
    """Tests using deterministic mock LLM."""

    def test_plan_mock_generates_valid_plan(self) -> None:
        """Mock LLM should return a valid ExperimentPlan."""
        inp = DesignInput(
            research_idea="Test channel attention on CIFAR-10 image classification",
            target_task="image classification",
            compute_budget=ComputeBudget(gpu="RTX 4090", max_runtime="2 hours", max_trials=5),
        )
        result, diags = plan(inp, mock=True)
        assert result is not None
        assert result.version == 1
        assert result.goal.hypothesis
        assert len(result.experiment_matrix.datasets) > 0
        assert len(result.experiment_matrix.methods) > 0
        assert len(result.experiment_matrix.metrics) > 0
        assert len(result.risks) > 0

    def test_plan_mock_passes_validation(self) -> None:
        """Mock-generated plan should pass deterministic validation."""
        inp = DesignInput(
            research_idea="Test idea",
            target_task="classification",
        )
        result, diags = plan(inp, mock=True)
        vr = validate(result)
        assert vr.status == "ok", f"Mock plan failed validation: {vr.issues}"

    def test_plan_mock_has_task_rationale(self) -> None:
        """Mock plan tasks should have rationale."""
        inp = DesignInput(
            research_idea="Test idea",
            target_task="classification",
        )
        result, diags = plan(inp, mock=True)
        for t in result.tasks.coding_tasks:
            assert t.rationale, f"Task {t.id} missing rationale"
        for t in result.tasks.repro_tasks:
            assert t.rationale, f"Task {t.id} missing rationale"
        for t in result.tasks.run_tasks:
            assert t.rationale, f"Task {t.id} missing rationale"

    def test_plan_mock_has_method_rationale(self) -> None:
        """Mock plan methods should have rationale."""
        inp = DesignInput(
            research_idea="Test idea",
            target_task="classification",
        )
        result, diags = plan(inp, mock=True)
        for m in result.experiment_matrix.methods:
            assert m.rationale, f"Method {m.name} missing rationale"

    def test_plan_mock_has_baseline_and_proposed(self) -> None:
        """Mock plan should include both baseline and proposed methods."""
        inp = DesignInput(
            research_idea="Test idea",
            target_task="classification",
        )
        result, diags = plan(inp, mock=True)
        types = {m.type for m in result.experiment_matrix.methods}
        assert "baseline" in types, "No baseline in mock plan"
        assert "new_method" in types, "No new_method in mock plan"


class TestPlannerReviseMock:
    """Tests for the revision workflow with mock LLM."""

    def test_revise_mock_returns_plan(self) -> None:
        """Revise should return a plan when called with mock."""
        from experiment_designer.planner import revise
        from experiment_designer.models import (
            ExperimentPlan, ResearchGoal, ExperimentMatrix, TaskBundle,
        )

        # Generate initial plan
        inp = DesignInput(
            research_idea="Test idea",
            target_task="classification",
        )
        current, _ = plan(inp, mock=True)

        # Revise it
        revised, diags = revise(current, "Add ViT baseline", mock=True)
        assert revised is not None
        assert revised.version == 1


class TestPlannerExtractYaml:
    """Tests for the YAML extraction logic."""

    def test_extract_fenced_yaml(self) -> None:
        from experiment_designer.planner import _extract_yaml
        text = """Some text
```yaml
version: 1
goal:
  summary: test
```
More text"""
        result = _extract_yaml(text)
        assert "version: 1" in result
        assert "Some text" not in result
        assert "More text" not in result

    def test_extract_generic_fence(self) -> None:
        from experiment_designer.planner import _extract_yaml
        text = """```
version: 1
goal:
  summary: test
```"""
        result = _extract_yaml(text)
        assert "version: 1" in result

    def test_extract_raw_yaml(self) -> None:
        from experiment_designer.planner import _extract_yaml
        text = "version: 1\ngoal:\n  summary: test"
        result = _extract_yaml(text)
        assert result.strip() == text.strip()


class TestPlannerTraceDir:
    """Tests for trace file writing."""

    def test_trace_dir_written(self) -> None:
        """Trace files should be written when trace_dir is provided."""
        inp = DesignInput(
            research_idea="Test idea",
            target_task="classification",
        )
        trace_dir = _runs_dir() / "test_traces"
        if trace_dir.exists():
            import shutil
            shutil.rmtree(trace_dir)
        result, diags = plan(inp, mock=True, trace_dir=trace_dir)
        files = list(trace_dir.glob("*.prompt.txt"))
        assert len(files) > 0, "No prompt trace files found"
        resp_files = list(trace_dir.glob("*.response.txt"))
        assert len(resp_files) > 0, "No response trace files found"


def _runs_dir() -> Path:
    """Return the project-local runs directory for test artifacts."""
    project_root = Path(__file__).resolve().parent.parent
    return project_root / "runs" / "tests"
