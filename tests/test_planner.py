"""Integration tests for the planner (mock LLM and real LLM)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import experiment_designer.advisor as advisor_mod
import experiment_designer.llm as llm_mod
from experiment_designer.advisor import advise
from experiment_designer.models import AdvisorContext, ComputeBudget, DesignInput
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


class TestMultiToolCalls:
    """Regression tests for DeepSeek returning multiple parallel tool_calls."""

    def test_call_llm_returns_all_parallel_tool_calls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        payload = {
            "choices": [{
                "message": {
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "search_papers", "arguments": json.dumps({"query": "alpha"})},
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {"name": "note_finding", "arguments": json.dumps({"topic": "t", "finding": "f"})},
                        },
                    ]
                }
            }]
        }

        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps(payload).encode("utf-8")

        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
        monkeypatch.setattr(llm_mod.urllib.request, "urlopen", lambda req, timeout: _Resp())

        result = llm_mod.call_llm(
            model="deepseek-chat",
            api_base="https://api.deepseek.com/v1",
            api_key_env="DEEPSEEK_API_KEY",
            system="system",
            messages=[{"role": "user", "content": "hi"}],
        )

        assert result["type"] == "tool_calls"
        assert [c["name"] for c in result["calls"]] == ["search_papers", "note_finding"]
        assert result["calls"][0]["arguments"] == {"query": "alpha"}
        assert result["calls"][1]["arguments"] == {"topic": "t", "finding": "f"}

    def test_advisor_executes_all_parallel_tool_calls(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        responses = iter([
            {
                "type": "tool_calls",
                "calls": [
                    {"name": "note_finding", "arguments": {"topic": "first topic", "finding": "first finding", "source": "test"}},
                    {"name": "note_finding", "arguments": {"topic": "second topic", "finding": "second finding", "source": "test"}},
                ],
            },
            {
                "type": "tool_calls",
                "calls": [{"name": "finish", "arguments": {
                    "summary": "Multi-tool regression decision",
                    "confidence": "medium",
                    "conclusion_status": "needs_more_experiments",
                    "conclusion_rationale": "Regression test rationale.",
                    "evidence": [{"source": "reasoning", "description": "Both parallel calls executed."}],
                    "recommended_actions": [{"priority": "high", "type": "coding_task",
                        "rationale": "Keep test executable",
                        "plan": {"kind": "coding_task", "task_goal": "No-op regression task"}}],
                    "risks": ["Regression test only"],
                    "needs_user_input": [],
                }}],
            },
        ])

        monkeypatch.setattr(advisor_mod, "call_llm", lambda **kwargs: next(responses))

        decision, trace = advise(
            AdvisorContext(situation="multi tool regression"),
            run_dir=tmp_path,
            trace_dir=tmp_path / "logs",
        )

        assert decision.summary == "Multi-tool regression decision"
        assert [t["summary"] for t in trace if t["action"] == "note_finding"] == ["first topic", "second topic"]
        state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
        assert [f["topic"] for f in state["findings"]] == ["first topic", "second topic"]


class TestFinishSchema:
    """Regression tests for the structured finish contract."""

    def test_finish_tool_schema_constrains_nested_plan(self) -> None:
        from experiment_designer.prompts import TOOLS

        finish = next(t["function"] for t in TOOLS if t["function"]["name"] == "finish")
        params = finish["parameters"]
        plan_schema = params["properties"]["experiment_plan"]

        method = plan_schema["properties"]["experiment_matrix"]["properties"]["methods"]["items"]
        assert method["properties"]["type"]["enum"] == ["new_method", "baseline", "ablation"]
        assert method["properties"]["implementation_status"]["enum"] == ["needs_code", "needs_repro", "existing"]
        assert "rationale" in method["required"]

        run_task = plan_schema["properties"]["tasks"]["properties"]["run_tasks"]["items"]
        assert "command_goal" in run_task["required"]
        assert "rationale" in run_task["required"]

    def test_system_prompt_matches_structured_finish(self) -> None:
        from experiment_designer.prompts import SYSTEM_PROMPT

        assert "decision_json" not in SYSTEM_PROMPT
        assert "JSON-encoded string" not in SYSTEM_PROMPT
        assert "conclusion_status" in SYSTEM_PROMPT


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
        trace_dir = _runs_dir("test_traces")
        if trace_dir.exists():
            import shutil
            shutil.rmtree(trace_dir)
        result, diags = plan(inp, mock=True, trace_dir=trace_dir)
        files = list(trace_dir.glob("*.prompt.txt"))
        assert len(files) > 0, "No prompt trace files found"
        resp_files = list(trace_dir.glob("*.response.txt"))
        assert len(resp_files) > 0, "No response trace files found"


def _runs_dir(*subdirs: str) -> Path:
    """Return runs/tests/<subdirs>/<timestamp>/ for isolated test artifacts."""
    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    project_root = Path(__file__).resolve().parent.parent
    return project_root / "runs" / "tests" / Path(*subdirs) / stamp
