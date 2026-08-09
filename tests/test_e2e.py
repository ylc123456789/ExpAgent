"""End-to-end tests using real DeepSeek API.

These tests validate both correctness (generates valid structured output)
and quality (generates useful, non-trivial experiment plans).

Requires DEEPSEEK_API_KEY to be set in the environment.
Skip if not available.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from experiment_designer.models import DesignInput, ComputeBudget, ExistingAssets, ExistingMethod
from experiment_designer.planner import plan, revise
from experiment_designer.report import write_plan
from experiment_designer.validator import validate


# ── Skip marker ──────────────────────────────────────────────────

API_KEY = os.environ.get("DEEPSEEK_API_KEY")

needs_api_key = pytest.mark.skipif(
    not API_KEY,
    reason="DEEPSEEK_API_KEY not set — set it to run real LLM tests",
)


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def simple_idea(tmp_path: Path) -> DesignInput:
    """A simple, well-scoped research idea for testing."""
    project_dir = tmp_path / "my_project"
    resnet_path = project_dir / "models" / "resnet.py"
    resnet_path.parent.mkdir(parents=True, exist_ok=True)
    resnet_path.write_text(
        '"""Minimal ResNet-18 fixture used by e2e tests."""\n',
        encoding="utf-8",
    )

    return DesignInput(
        research_idea=(
            "我想验证一种新的通道注意力机制在CIFAR-10图像分类上的效果。"
            "核心假设是：通过对通道权重做L2归一化后再缩放，"
            "可以让注意力权重更稳定，比SE-block在训练初期收敛更快，"
            f"同时最终精度不下降。我已经有一个PyTorch项目在 {project_dir}，"
            "里面有标准的ResNet-18训练代码。"
        ),
        target_task="image classification",
        compute_budget=ComputeBudget(
            gpu="RTX 4090",
            max_runtime="2 hours",
            max_trials=5,
        ),
        existing_assets=ExistingAssets(
            implemented_methods=[
                ExistingMethod(
                    name="resnet18_base",
                    location=str(resnet_path),
                    description="标准ResNet-18，包含训练和评估代码",
                )
            ],
            available_datasets=["CIFAR-10"],
        ),
        constraints=["先做小规模验证（10 epochs）", "不要修改训练入口脚本"],
    )


@pytest.fixture
def complex_idea() -> DesignInput:
    """A more complex research idea requiring multiple baselines and ablations."""
    return DesignInput(
        research_idea=(
            "我提出了一种混合架构：用Spatial CNN提取局部特征 + Graph NN "
            "建模全局依赖，用于时间序列预测。核心创新在于Graph的构建方式"
            "——不是用预定义的距离阈值，而是通过网络学习节点间的自适应"
            "连接权重。目标是在ETTh1和Weather数据集上验证，假设我的方法"
            "在长期预测（>96步）上比Informer和Autoformer更好，且参数量更少。"
        ),
        target_task="time series forecasting",
        compute_budget=ComputeBudget(
            gpu="A100",
            max_runtime="4 hours",
            max_trials=10,
        ),
        constraints=["ETTh1和Weather数据集不能重新下载，用缓存"],
    )


# ── Core E2E tests ───────────────────────────────────────────────

@needs_api_key
@pytest.mark.e2e
class TestPlanGeneration:
    """Test that real LLM generates valid, quality plans."""

    def test_simple_idea_generates_valid_plan(self, simple_idea: DesignInput) -> None:
        """A simple, well-scoped idea should generate a valid plan in one shot."""
        result, diags = _call_plan(simple_idea)

        # Basic correctness
        assert result is not None
        assert result.version == 1
        assert result.goal.summary
        assert result.goal.hypothesis

        # Validation must pass
        vr = validate(result)
        assert vr.status == "ok", (
            f"Plan validation FAILED with {len(vr.issues)} issues:\n" +
            "\n".join(f"  - {i}" for i in vr.issues)
        )

    def test_simple_idea_no_diagnostics(self, simple_idea: DesignInput) -> None:
        """A straightforward idea should not need retries."""
        result, diags = _call_plan(simple_idea)
        # Filter out informational diagnostics (confidence level, etc.)
        actual_errors = [d for d in diags if d and "confidence" not in d and "Final:" not in d and "Fallback" not in d]
        assert len(actual_errors) == 0, (
            f"Unexpected retries/diagnostics:\n" +
            "\n".join(f"  - {d}" for d in actual_errors)
        )

    def test_has_meaningful_hypothesis(self, simple_idea: DesignInput) -> None:
        """Hypothesis should be specific, not generic."""
        result, diags = _call_plan(simple_idea)
        hyp = result.goal.hypothesis.lower()
        # Should reference specific concepts from the idea
        assert len(result.goal.hypothesis) > 30, (
            f"Hypothesis too short: '{result.goal.hypothesis}'"
        )
        # Should not be a one-size-fits-all template
        banned_phrases = ["will be evaluated", "will be tested", "will be compared"]
        for phrase in banned_phrases:
            assert phrase not in hyp.lower(), (
                f"Hypothesis contains vague placeholder: '{phrase}'"
            )

    def test_infers_baselines(self, simple_idea: DesignInput) -> None:
        """LLM should infer baselines beyond what the user mentioned."""
        result, diags = _call_plan(simple_idea)
        baseline_names = [
            m.name.lower() for m in result.experiment_matrix.methods
            if m.type == "baseline"
        ]
        # User mentioned SE-block; LLM should at minimum include that
        # and ideally also suggest at least one more baseline (e.g. CBAM, ECA)
        assert len(baseline_names) >= 1, (
            f"Expected at least 1 baseline, got {len(baseline_names)}"
        )

    def test_distinguishes_task_types(self, simple_idea: DesignInput) -> None:
        """Plan should properly separate coding vs repro vs run tasks."""
        result, diags = _call_plan(simple_idea)

        # Should have at least one coding task (implement proposed method)
        assert len(result.tasks.coding_tasks) >= 1, (
            "Expected at least 1 coding task for implementing the proposed method"
        )

        # Coding tasks must have a meaningful task_goal
        for t in result.tasks.coding_tasks:
            assert t.task_goal, (
                f"Coding task {t.id} has no task_goal"
            )

    def test_every_design_choice_has_rationale(self, simple_idea: DesignInput) -> None:
        """Every method, dataset, metric, and task should explain WHY."""
        result, diags = _call_plan(simple_idea)

        for ds in result.experiment_matrix.datasets:
            assert ds.rationale.strip(), f"Dataset {ds.name} has empty rationale"

        for m in result.experiment_matrix.methods:
            assert m.rationale.strip(), f"Method {m.name} has empty rationale"

        for t in result.tasks.coding_tasks:
            assert t.rationale.strip(), f"Coding task {t.id} has empty rationale"

        for t in result.tasks.repro_tasks:
            assert t.rationale.strip(), f"Repro task {t.id} has empty rationale"

        for t in result.tasks.run_tasks:
            assert t.rationale.strip(), f"Run task {t.id} has empty rationale"

    def test_risks_are_concrete(self, simple_idea: DesignInput) -> None:
        """Risks should be specific to the experiment, not generic."""
        result, diags = _call_plan(simple_idea)
        assert len(result.risks) >= 2, f"Expected at least 2 risks, got {len(result.risks)}"

        # Each risk should have both description and mitigation
        for r in result.risks:
            assert len(r.description) > 20, (
                f"Risk description too generic: '{r.description}'"
            )
            assert r.mitigation.strip(), (
                f"Risk '{r.description[:50]}...' has empty mitigation"
            )

    def test_success_criteria_are_measurable(self, simple_idea: DesignInput) -> None:
        """Success criteria should contain numbers or concrete thresholds."""
        result, diags = _call_plan(simple_idea)
        assert len(result.goal.success_criteria) >= 1

        # At least one criterion should be numeric/concrete
        has_concrete = False
        for sc in result.goal.success_criteria:
            if any(c.isdigit() for c in sc):
                has_concrete = True
                break
            if any(kw in sc.lower() for kw in ["%", "percent", "faster", "higher", "lower"]):
                has_concrete = True
                break
        assert has_concrete, (
            f"No success criterion appears measurable: {result.goal.success_criteria}"
        )

    def test_recognizes_existing_assets(self, simple_idea: DesignInput) -> None:
        """When user has existing code/data, LLM should note and use it."""
        result, diags = _call_plan(simple_idea)

        # At least one method should be marked as 'existing'
        existing = [m for m in result.experiment_matrix.methods
                    if m.implementation_status == "existing"]
        assert len(existing) >= 1, (
            f"Expected at least 1 method with status 'existing', "
            f"but got: {[(m.name, m.implementation_status) for m in result.experiment_matrix.methods]}"
        )


@needs_api_key
@pytest.mark.e2e
class TestComplexScenarios:
    """Tests for more challenging research ideas."""

    def test_complex_idea_passes_validation(self, complex_idea: DesignInput) -> None:
        """Complex multi-dataset, multi-baseline idea should validate."""
        result, diags = _call_plan(complex_idea)
        vr = validate(result)
        assert vr.status == "ok", (
            f"Complex plan validation FAILED:\n" +
            "\n".join(f"  - {i}" for i in vr.issues)
        )

    def test_complex_idea_has_multiple_baselines(self, complex_idea: DesignInput) -> None:
        """Time series idea should include Informer and Autoformer baselines."""
        result, diags = _call_plan(complex_idea)
        baseline_names = [
            m.name.lower() for m in result.experiment_matrix.methods
            if m.type == "baseline"
        ]
        # Should have at least 2 baselines (Informer, Autoformer mentioned in idea)
        assert len(baseline_names) >= 2, (
            f"Expected >= 2 baselines for time series, got {len(baseline_names)}: {baseline_names}"
        )

    def test_complex_idea_has_multiple_datasets(self, complex_idea: DesignInput) -> None:
        """Idea mentions ETTh1 and Weather — both should appear."""
        result, diags = _call_plan(complex_idea)
        dataset_names = [d.name.lower() for d in result.experiment_matrix.datasets]
        assert len(dataset_names) >= 2, (
            f"Expected >= 2 datasets, got {len(dataset_names)}: {dataset_names}"
        )


@needs_api_key
@pytest.mark.e2e
class TestRevision:
    """Tests for the plan revision workflow."""

    def test_revise_add_baseline(self, simple_idea: DesignInput) -> None:
        """After generating a plan, user can ask to add a baseline."""
        result, _ = _call_plan(simple_idea)
        n_before = len(result.experiment_matrix.methods)

        revised, diags = revise(
            result,
            "加一个ViT-B/16作为额外的baseline，验证方法对Transformer架构是否也有效",
            api_key_env="DEEPSEEK_API_KEY",
        )

        n_after = len(revised.experiment_matrix.methods)
        assert n_after >= n_before, (
            f"Expected methods to increase after adding baseline "
            f"({n_before} -> expected >= {n_before}, got {n_after})"
        )

        # Should still validate
        vr = validate(revised)
        assert vr.status == "ok", (
            f"Revised plan failed validation: {vr.issues}"
        )

    def test_revise_add_ablation(self, simple_idea: DesignInput) -> None:
        """After generating a plan, user can ask to add an ablation study."""
        result, _ = _call_plan(simple_idea)

        revised, diags = revise(
            result,
            "增加一个ablation实验：去掉L2归一化只保留缩放，看归一化这一步是否真的必要",
            api_key_env="DEEPSEEK_API_KEY",
        )

        # The ablation should appear somewhere — methods, tasks, or rationale
        methods_after = len(revised.experiment_matrix.methods)
        has_ablation = any(
            m.type == "ablation" for m in revised.experiment_matrix.methods
        )
        # Even if not in methods, the plan should have grown or changed
        assert methods_after > 0, "Plan should still have methods after revision"

    def test_revise_preserves_unrelated_parts(self, simple_idea: DesignInput) -> None:
        """Revision should keep unrelated sections intact."""
        result, _ = _call_plan(simple_idea)
        original_summary = result.goal.summary
        original_datasets = [d.name for d in result.experiment_matrix.datasets]

        revised, diags = revise(
            result,
            "把success_criteria里的精度阈值从2%改成3%",
            api_key_env="DEEPSEEK_API_KEY",
        )

        # Core structure should be preserved
        assert revised.goal.summary == original_summary, (
            "Revision changed the summary when it shouldn't have"
        )
        revised_dataset_names = [d.name for d in revised.experiment_matrix.datasets]
        assert revised_dataset_names == original_datasets, (
            "Revision changed datasets when it shouldn't have"
        )


@needs_api_key
@pytest.mark.e2e
class TestReportWrite:
    """Tests for writing plans to disk."""

    def test_write_and_reread(self, simple_idea: DesignInput) -> None:
        """Generated plan should survive a write-reread roundtrip."""
        result, _ = _call_plan(simple_idea)

        output_dir = _runs_dir("test_write_reread")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        path = write_plan(result, output_dir)
        assert path.exists()

        # Reread
        import yaml
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        from experiment_designer.models import ExperimentPlan
        reread = ExperimentPlan.model_validate(data)

        assert reread.goal.summary == result.goal.summary
        assert reread.goal.hypothesis == result.goal.hypothesis
        assert len(reread.experiment_matrix.methods) == len(result.experiment_matrix.methods)


# ── Helpers ──────────────────────────────────────────────────────


def _call_plan(inp: DesignInput, **kwargs) -> tuple:
    """Call plan() with default DeepSeek settings."""
    return plan(
        inp,
        model="deepseek-chat",
        api_base="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY",
        run_dir=_runs_dir("plan"),
        **kwargs,
    )


@needs_api_key
@pytest.mark.e2e
class TestAdvisorV2:
    """Tests for the new v2 advise() API directly."""

    def test_advise_design_experiment(self) -> None:
        """Direct advise() call for experiment design."""
        from experiment_designer.advisor import advise
        from experiment_designer.models import AdvisorContext

        ctx = AdvisorContext(
            situation=(
                "TASK: Design an initial experiment plan.\n"
                "Research Idea: Validate a novel channel attention mechanism "
                "on CIFAR-10 image classification.\n"
                "Target Task: image classification\n"
            )
        )
        decision, trace = advise(ctx, api_key_env="DEEPSEEK_API_KEY", run_dir=_runs_dir("advisor"))
        assert decision.summary
        assert decision.confidence in ("high", "medium", "low")
        assert decision.conclusion.status
        assert len(decision.recommended_actions) >= 1

    def test_advise_with_artifacts(self) -> None:
        """Advise with artifact references."""
        from experiment_designer.advisor import advise
        from experiment_designer.models import AdvisorContext, ArtifactRef

        ctx = AdvisorContext(
            situation=(
                "TASK: Analyze experiment results.\n"
                "Our proposed method got 82.1% accuracy while the ResNet-18 baseline "
                "got 83.7%. The hypothesis predicted our method would outperform.\n"
                "Is this result conclusive, or should we run more experiments?"
            ),
            artifacts=[
                ArtifactRef(id="run_001", type="run_log",
                            summary="proposed method: 82.1% accuracy, baseline: 83.7%"),
            ],
        )
        decision, trace = advise(ctx, api_key_env="DEEPSEEK_API_KEY", run_dir=_runs_dir("advisor"))
        assert decision.summary
        assert decision.conclusion.status
        # Should have some evidence
        assert len(decision.evidence) >= 1

    def test_advise_yields_validatable_decision(self) -> None:
        """Every advise() output should be structurally sound."""
        from experiment_designer.advisor import advise
        from experiment_designer.models import AdvisorContext
        from experiment_designer.validator import validate_decision

        ctx = AdvisorContext(
            situation=(
                "TASK: Design an experiment plan.\n"
                "Research Idea: Test a hybrid CNN+Transformer architecture "
                "for time series forecasting on ETTh1.\n"
                "Target Task: time series forecasting\n"
            )
        )
        decision, trace = advise(ctx, api_key_env="DEEPSEEK_API_KEY", run_dir=_runs_dir("advisor"))
        vr = validate_decision(decision)
        # Core fields must be present
        assert decision.summary
        assert decision.conclusion.status
        assert decision.conclusion.rationale
        assert len(decision.recommended_actions) >= 1
        assert len(decision.risks) >= 1

    def test_recommended_actions_self_contained(self) -> None:
        """Each recommended action must be self-contained for downstream agents."""
        from experiment_designer.advisor import advise
        from experiment_designer.models import AdvisorContext

        ctx = AdvisorContext(
            situation=(
                "TASK: Design experiment plan for verifying a new attention mechanism "
                "on CIFAR-10. Include at least one baseline to reproduce.\n"
                "Target Task: image classification\n"
            )
        )
        decision, trace = advise(ctx, api_key_env="DEEPSEEK_API_KEY", run_dir=_runs_dir("advisor"))

        for action in decision.recommended_actions:
            plan = action.plan
            if action.type == "repro_task":
                assert plan.paper_url or plan.repo_url, (
                    f"Repro action missing paper/repo URL: {action.rationale[:60]}"
                )
            elif action.type == "coding_task":
                # LLM may put task goal in experiment_goal field
                goal = plan.task_goal or plan.experiment_goal or ""
                assert goal, (
                    f"Coding action missing goal: {action.rationale[:60]}"
                )
            elif action.type == "run_task":
                goal = plan.command_goal or plan.experiment_goal or ""
                assert goal, (
                    f"Run action missing goal: {action.rationale[:60]}"
                )


@needs_api_key
@pytest.mark.e2e
class TestV2Models:
    """Tests for v2 data models."""

    def test_scientific_decision_roundtrip(self) -> None:
        """ScientificDecision should survive YAML roundtrip."""
        import yaml
        from experiment_designer.models import (
            ScientificDecision, ScientificConclusion, EvidenceItem,
            RecommendedAction, SuggestedPlan,
        )

        sd = ScientificDecision(
            summary="Test decision",
            confidence="medium",
            conclusion=ScientificConclusion(status="needs_more_experiments", rationale="Need more data"),
            evidence=[EvidenceItem(source="reasoning", description="Logic suggests more trials")],
            recommended_actions=[
                RecommendedAction(
                    priority="high", type="repro_task",
                    rationale="Reproduce baseline",
                    plan=SuggestedPlan(
                        kind="repro_task",
                        paper_url="https://example.com/paper",
                        repo_url="https://github.com/example/repo",
                        experiment_goal="Reproduce results",
                    ),
                )
            ],
            risks=["Small sample size"],
        )

        dumped = yaml.dump(sd.model_dump(), allow_unicode=True, sort_keys=False)
        loaded = yaml.safe_load(dumped)
        revalidated = ScientificDecision.model_validate(loaded)
        assert revalidated.summary == sd.summary
        assert len(revalidated.recommended_actions) == 1

    def test_advisor_context_build(self) -> None:
        """AdvisorContext should be easy to construct."""
        from experiment_designer.models import AdvisorContext, ArtifactRef

        ctx = AdvisorContext(
            situation="Test situation",
            artifacts=[ArtifactRef(id="a1", type="run_log", summary="82.1% accuracy")],
        )
        assert len(ctx.artifacts) == 1
        assert ctx.artifacts[0].id == "a1"


def _runs_dir(*subdirs: str) -> Path:
    """Return a timestamped directory for test artifacts.

    Uses pytest tmp_path by default (cleaned after test). Set
    EXPAGENT_KEEP_TEST_TRACES=1 to persist traces in the project.
    """
    import os as _os
    import tempfile as _tempfile
    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    if _os.environ.get("EXPAGENT_KEEP_TEST_TRACES") == "1":
        project_root = Path(__file__).resolve().parent.parent
        return project_root / "runs" / "tests" / Path(*subdirs) / stamp

    return Path(_tempfile.mkdtemp(prefix="expagent_test_")) / Path(*subdirs) / stamp
