"""Phase 0 compatibility locks for behavior-preserving refactors."""

from __future__ import annotations

import hashlib
from contextlib import redirect_stdout
from io import StringIO

import experiment_designer
from experiment_designer.main import _parse_args
from experiment_designer.models import (
    AdvisorContext,
    RecommendedAction,
    ScientificDecision,
    SuggestedPlan,
)
from experiment_designer.prompts import SYSTEM_PROMPT, build_initial_prompt


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_public_package_contract() -> None:
    assert experiment_designer.__all__ == [
        "AdvisorContext", "AnalysisPlan", "ArtifactRef", "CodingTask",
        "ComputeBudget", "ContextPolicy", "DatasetSpec", "DesignInput",
        "EvidenceItem", "ExistingAssets", "ExperimentMatrix", "ExperimentPlan",
        "FailureDiagnosis", "MethodSpec", "MetricSpec", "RecommendedAction",
        "ReproTask", "ResearchGoal", "ResultAnalysis", "Risk", "RunTask",
        "ScientificConclusion", "ScientificDecision", "SuggestedPlan",
        "TaskBundle", "ValidationResult",
    ]


def test_cli_help_contract() -> None:
    output = StringIO()
    with redirect_stdout(output):
        try:
            _parse_args(["--help"])
        except SystemExit as exc:
            assert exc.code == 0
    assert _sha256(output.getvalue()) == "bbce4c5c23b9109ebd12a3bc8ee6b3a5cc7112edc99be7548c27614c3655374b"


def test_prompt_contracts() -> None:
    rendered = build_initial_prompt(AdvisorContext(situation="phase0 situation"))
    assert _sha256(SYSTEM_PROMPT) == "0d8d74aa0f575e64312f829b7cdf0357d3fc973f7cbd1a00e5b5d194b5143569"
    assert _sha256(rendered) == "84128b7d5013e99f22f18348ffe53722f13d0660f93e1c22ad8bdf627f551cbc"


def test_cross_module_model_field_contracts() -> None:
    assert list(AdvisorContext.model_fields) == [
        "situation", "artifacts", "existing_plan", "thread_dir", "parent_run",
    ]
    assert list(ScientificDecision.model_fields) == [
        "summary", "confidence", "conclusion", "evidence", "experiment_plan",
        "result_analysis", "failure_diagnosis", "recommended_actions", "risks",
        "needs_user_input",
    ]
    assert list(RecommendedAction.model_fields) == [
        "priority", "type", "rationale", "plan", "action_id", "depends_on",
        "project_ref", "workspace_intent",
    ]
    assert list(SuggestedPlan.model_fields) == [
        "kind", "code_availability", "workspace_path", "task_goal", "constraints",
        "verify_commands", "expected_artifacts", "paper_url", "repo_url",
        "experiment_goal", "compute_budget", "expected_metrics", "command_goal",
        "expected_runtime", "requires_gpu", "search_query", "question",
    ]
