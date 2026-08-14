"""Phase 0 compatibility locks for behavior-preserving refactors."""

from __future__ import annotations

import hashlib
from contextlib import redirect_stdout
from io import StringIO

import experiment_designer
from experiment_designer.main import _parse_args
from experiment_designer.models import (
    AdvisorContext,
    AnalyzeResultsAction,
    ExecuteExperimentAction,
    ScientificDecision,
)
from experiment_designer.prompts import SYSTEM_PROMPT, build_initial_prompt


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_public_package_contract() -> None:
    assert experiment_designer.__all__ == [
        "AdvisorContext", "AnalysisPlan", "AnalyzeResultsAction", "ArtifactRef",
        "AskUserAction", "CodingTask", "ComputeBudget", "ContextPolicy",
        "DatasetSpec", "DesignInput", "EvidenceItem", "ExecuteExperimentAction",
        "ExistingAssets", "ExperimentMatrix", "ExperimentPlan", "FailureDiagnosis",
        "MethodSpec", "MetricSpec", "ModifyCodeAction", "ReproduceExperimentAction",
        "ReproTask", "ResearchGoal", "ResultAnalysis", "Risk", "RunTask",
        "ScientificAction", "ScientificConclusion", "ScientificDecision",
        "SearchLiteratureAction", "TaskBundle", "ValidationResult",
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
    assert _sha256(SYSTEM_PROMPT) == "8d1025dc13ccd7f2800c513a4b1702f31ce7490326a43fd77e8a14afecd29640"
    assert _sha256(rendered) == "84128b7d5013e99f22f18348ffe53722f13d0660f93e1c22ad8bdf627f551cbc"


def test_cross_module_model_field_contracts() -> None:
    assert list(AdvisorContext.model_fields) == [
        "situation", "artifacts", "existing_plan", "thread_dir", "parent_run",
    ]
    assert list(ScientificDecision.model_fields) == [
        "summary", "confidence", "conclusion", "evidence", "experiment_plan",
        "result_analysis", "failure_diagnosis", "recommended_actions",
        "analysis_required", "risks", "needs_user_input",
    ]
    # V2 scientific action contract — a representative capability submodel
    assert list(ExecuteExperimentAction.model_fields) == [
        "action_id", "capability", "objective", "rationale", "depends_on",
        "project_ref", "required", "success_criteria", "requires_gpu",
        "expected_metrics",
    ]
    assert list(AnalyzeResultsAction.model_fields) == [
        "action_id", "capability", "objective", "rationale", "depends_on",
        "project_ref", "required", "success_criteria",
    ]
