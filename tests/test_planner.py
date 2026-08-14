"""Integration tests for the planner (mock LLM and real LLM)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import experiment_designer.controller.loop as loop_mod
import experiment_designer.llm as llm_mod
from experiment_designer.agent import advise
from experiment_designer.models import AdvisorContext, ComputeBudget, DesignInput
from experiment_designer.controller.planner import plan, revise
from experiment_designer.controller.validator import validate


class TestPlannerMock:
    """Tests using deterministic mock LLM."""

    def test_plan_mock_generates_valid_plan(self) -> None:
        """Mock LLM should return a valid ExperimentPlan."""
        inp = DesignInput(
            research_idea="Test channel attention on CIFAR-10 image classification",
            target_task="image classification",
            compute_budget=ComputeBudget(gpu="RTX 4090", max_runtime="2 hours", max_trials=5),
        )
        result, diags = plan(inp, mock=True, run_dir=_runs_dir("mock"))
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
        result, diags = plan(inp, mock=True, run_dir=_runs_dir("mock"))
        vr = validate(result)
        assert vr.status == "ok", f"Mock plan failed validation: {vr.issues}"

    def test_plan_mock_has_task_rationale(self) -> None:
        """Mock plan tasks should have rationale."""
        inp = DesignInput(
            research_idea="Test idea",
            target_task="classification",
        )
        result, diags = plan(inp, mock=True, run_dir=_runs_dir("mock"))
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
        result, diags = plan(inp, mock=True, run_dir=_runs_dir("mock"))
        for m in result.experiment_matrix.methods:
            assert m.rationale, f"Method {m.name} missing rationale"

    def test_plan_mock_has_baseline_and_proposed(self) -> None:
        """Mock plan should include both baseline and proposed methods."""
        inp = DesignInput(
            research_idea="Test idea",
            target_task="classification",
        )
        result, diags = plan(inp, mock=True, run_dir=_runs_dir("mock"))
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
                    "recommended_actions": [{"capability": "modify_code",
                        "rationale": "Keep test executable",
                        "action_id": "noop_regression",
                        "objective": "No-op regression task",
                        "success_criteria": ["test stays executable"],
                        "constraints": [], "verify_commands": [], "expected_artifacts": []}],
                    "risks": ["Regression test only"],
                    "needs_user_input": [],
                }}],
            },
        ])

        monkeypatch.setattr(loop_mod, "call_llm", lambda **kwargs: next(responses))

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


class TestActionDependencyMetadata:
    """Tests for the V2 action graph (capability, action_id, depends_on, project_ref)."""

    def test_modify_then_run_then_analyze_roundtrip(self) -> None:
        """modify_code -> execute_experiment -> analyze_results round-trips."""
        import yaml
        from experiment_designer.models import (
            ScientificDecision, ScientificConclusion, EvidenceItem,
            ModifyCodeAction, ExecuteExperimentAction, AnalyzeResultsAction,
        )
        from experiment_designer.controller.validator import validate_decision

        sd = ScientificDecision(
            summary="Decision with dependency chain",
            confidence="medium",
            conclusion=ScientificConclusion(status="needs_more_experiments", rationale="Test rationale."),
            evidence=[EvidenceItem(source="reasoning", description="Test evidence.")],
            recommended_actions=[
                ModifyCodeAction(
                    action_id="patch_training_loop",
                    objective="Add loss logging",
                    rationale="Patch training loop",
                    project_ref="current_project",
                ),
                ExecuteExperimentAction(
                    action_id="run_with_patch",
                    objective="Run validation",
                    rationale="Run after code patch",
                    depends_on=["patch_training_loop"],
                    project_ref="current_project",
                    success_criteria=["validation completes"],
                    expected_metrics=["accuracy"],
                ),
                AnalyzeResultsAction(
                    action_id="analyze_after_run",
                    objective="Interpret validation",
                    rationale="Judge the result",
                    depends_on=["run_with_patch"],
                    project_ref="current_project",
                ),
            ],
            risks=["Dependency chain untested"],
        )

        vr = validate_decision(sd)
        assert vr.status == "ok", f"Dependency decision should validate: {vr.issues}"

        dumped = yaml.dump(sd.model_dump(), allow_unicode=True, sort_keys=False)
        loaded = yaml.safe_load(dumped)
        revalidated = ScientificDecision.model_validate(loaded)
        assert len(revalidated.recommended_actions) == 3
        assert revalidated.recommended_actions[1].depends_on == ["patch_training_loop"]
        assert revalidated.recommended_actions[2].depends_on == ["run_with_patch"]
        assert revalidated.recommended_actions[2].project_ref == "current_project"

    def test_duplicate_action_id_rejected(self) -> None:
        """Two actions with the same action_id should fail validation."""
        from experiment_designer.models import (
            ScientificDecision, ScientificConclusion, EvidenceItem,
            ModifyCodeAction, ExecuteExperimentAction,
        )
        from experiment_designer.controller.validator import validate_decision

        sd = ScientificDecision(
            summary="Duplicate action_id test",
            confidence="medium",
            conclusion=ScientificConclusion(status="needs_more_experiments", rationale="Test."),
            evidence=[EvidenceItem(source="reasoning", description="Test.")],
            recommended_actions=[
                ModifyCodeAction(action_id="same_id", objective="Task 1", rationale="First"),
                ExecuteExperimentAction(action_id="same_id", objective="Task 2", rationale="Second", expected_metrics=["m"]),
            ],
            risks=["Test risk"],
        )
        vr = validate_decision(sd)
        assert vr.status == "needs_revision", "Duplicate action_id should be rejected"
        assert any("same_id" in i for i in vr.issues), f"Expected issue about 'same_id': {vr.issues}"

    def test_unknown_dependency_rejected(self) -> None:
        """depends_on referencing a non-existent action_id should fail validation."""
        from experiment_designer.models import (
            ScientificDecision, ScientificConclusion, EvidenceItem,
            AnalyzeResultsAction,
        )
        from experiment_designer.controller.validator import validate_decision

        sd = ScientificDecision(
            summary="Unknown dependency test",
            confidence="medium",
            conclusion=ScientificConclusion(status="needs_more_experiments", rationale="Test."),
            evidence=[EvidenceItem(source="reasoning", description="Test.")],
            recommended_actions=[
                AnalyzeResultsAction(
                    action_id="analyze",
                    objective="Interpret",
                    rationale="why",
                    depends_on=["nonexistent_action"],
                ),
            ],
            risks=["Test risk"],
        )
        vr = validate_decision(sd)
        assert vr.status == "needs_revision", "Unknown dependency should be rejected"
        assert any("nonexistent_action" in i for i in vr.issues), (
            f"Expected issue about 'nonexistent_action': {vr.issues}"
        )

    def test_mock_output_has_dependency_metadata(self) -> None:
        """Mock LLM output should include action_id/depends_on/project_ref in recommended actions."""
        from experiment_designer.llm import _make_mock_design_decision

        decision = _make_mock_design_decision()
        actions = decision["recommended_actions"]
        assert len(actions) >= 3, "Mock should have at least 3 recommended actions"
        assert actions[0].get("action_id"), "First action should have action_id"
        assert actions[0].get("project_ref") == "current_project"
        assert actions[1].get("depends_on"), "Second action should have depends_on"
        assert actions[0]["action_id"] in actions[1]["depends_on"]


class TestPlannerReviseMock:
    """Tests for the revision workflow with mock LLM."""

    def test_revise_mock_returns_plan(self) -> None:
        """Revise should return a plan when called with mock."""
        from experiment_designer.controller.planner import revise
        from experiment_designer.models import (
            ExperimentPlan, ResearchGoal, ExperimentMatrix, TaskBundle,
        )

        # Generate initial plan
        inp = DesignInput(
            research_idea="Test idea",
            target_task="classification",
        )
        current, _ = plan(inp, mock=True, run_dir=_runs_dir("mock"))

        # Revise it
        revised, diags = revise(current, "Add ViT baseline", mock=True, run_dir=_runs_dir("mock_revise"))
        assert revised is not None
        assert revised.version == 1


class TestPlannerExtractYaml:
    """Tests for the YAML extraction logic."""

    def test_extract_fenced_yaml(self) -> None:
        from experiment_designer.controller.planner import _extract_yaml
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
        from experiment_designer.controller.planner import _extract_yaml
        text = """```
version: 1
goal:
  summary: test
```"""
        result = _extract_yaml(text)
        assert "version: 1" in result

    def test_extract_raw_yaml(self) -> None:
        from experiment_designer.controller.planner import _extract_yaml
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
        result, diags = plan(inp, mock=True, trace_dir=trace_dir, run_dir=_runs_dir("mock_trace"))
        files = list(trace_dir.glob("*.prompt.txt"))
        assert len(files) > 0, "No prompt trace files found"
        resp_files = list(trace_dir.glob("*.response.txt"))
        assert len(resp_files) > 0, "No response trace files found"


class TestSessionCard:
    """Tests for E1: session.yaml index card."""

    def test_session_card_written_on_success(self) -> None:
        """plan() mock run should produce session.yaml in run_dir."""
        run_dir = _runs_dir("session_success")
        run_dir.mkdir(parents=True, exist_ok=True)
        inp = DesignInput(
            research_idea="Test idea",
            target_task="classification",
        )
        result, diags = plan(inp, mock=True, run_dir=run_dir)

        card_path = run_dir / "session.yaml"
        assert card_path.exists(), "session.yaml missing"

        import yaml
        card = yaml.safe_load(card_path.read_text(encoding="utf-8"))
        assert card["schema_version"] == 1
        assert card["module"] == "expagent"
        assert card["kind"] == "advisory_session"
        assert card["status"] == "completed"
        assert card["session_id"].startswith("exp-")
        assert card["project_path"]  # at minimum has the run dir path

    def test_session_card_written_on_failure(self) -> None:
        """Even on step exhaustion, session.yaml should be written."""
        run_dir = _runs_dir("session_fail")
        run_dir.mkdir(parents=True, exist_ok=True)

        # max_steps=0 → loop never enters → exhausted immediately
        from experiment_designer.agent import advise
        from experiment_designer.models import AdvisorContext

        ctx = AdvisorContext(situation="TASK: Some task.")
        decision, trace = advise(ctx, mock=True, run_dir=run_dir, max_steps=0)

        card_path = run_dir / "session.yaml"
        assert card_path.exists(), "session.yaml should exist even on failure"

        import yaml
        card = yaml.safe_load(card_path.read_text(encoding="utf-8"))
        assert card["status"] == "failed"


class TestAdvisoryThread:
    """Tests for E2: advisory thread support."""

    def test_thread_injects_prior_summaries(self) -> None:
        """When thread_dir has prior entries, they appear in the prompt."""
        run_dir = _runs_dir("thread1")
        thread_dir = _runs_dir("thread_store")
        thread_dir.mkdir(parents=True, exist_ok=True)

        # Write a prior thread entry
        import yaml
        thread_file = thread_dir / "thread.yaml"
        thread_file.write_text(yaml.dump({
            "entries": [{"ts": "2026-08-10T00:00:00", "summary": "Prior finding: SE-block is the standard baseline."}]
        }), encoding="utf-8")

        from experiment_designer.agent import advise
        from experiment_designer.models import AdvisorContext

        ctx = AdvisorContext(
            situation="New question: what other baselines should we consider?",
            thread_dir=str(thread_dir),
        )
        decision, trace = advise(ctx, mock=True, run_dir=run_dir, max_steps=3)

        # Verify thread was appended
        updated = yaml.safe_load(thread_file.read_text(encoding="utf-8"))
        assert len(updated["entries"]) == 2, "Should have original + new entry"
        assert decision.summary in [e["summary"] for e in updated["entries"]]


def _runs_dir(*subdirs: str) -> Path:
    """Return a timestamped directory for test artifacts.

    Uses temp dir by default. Set EXPAGENT_KEEP_TEST_TRACES=1 to persist.
    """
    import os as _os
    import tempfile as _tempfile
    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    if _os.environ.get("EXPAGENT_KEEP_TEST_TRACES") == "1":
        project_root = Path(__file__).resolve().parent.parent
        return project_root / "runs" / "tests" / Path(*subdirs) / stamp

    return Path(_tempfile.mkdtemp(prefix="expagent_test_")) / Path(*subdirs) / stamp
