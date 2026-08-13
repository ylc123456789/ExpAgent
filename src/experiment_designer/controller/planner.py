"""Backward-compatible wrappers around the advisor agentic loop.

plan() and revise() now delegate to advise() internally.
Existing callers (tests, REPL) don't need to change.
"""

from __future__ import annotations

from pathlib import Path

from ..agent import advise
from ..models import (
    AdvisorContext,
    DesignInput,
    ExperimentPlan,
    ScientificDecision,
)
from ..prompts import build_plan_prompt, build_revise_prompt


def plan(
    inp: DesignInput,
    *,
    model: str = "deepseek-chat",
    api_base: str = "https://api.deepseek.com/v1",
    api_key_env: str = "DEEPSEEK_API_KEY",
    mock: bool = False,
    trace_dir: Path | None = None,
    run_dir: Path | None = None,
) -> tuple[ExperimentPlan, list[str]]:
    """Generate a new experiment plan from a research idea.

    Delegates to advise() internally.
    """
    situation = build_plan_prompt(inp)
    ctx = AdvisorContext(situation=situation)

    decision, trace = advise(
        ctx,
        model=model,
        api_base=api_base,
        api_key_env=api_key_env,
        mock=mock,
        trace_dir=trace_dir,
        run_dir=run_dir,
    )

    if decision.experiment_plan:
        return _populate_tasks_from_actions(decision.experiment_plan, decision.recommended_actions), [f"confidence: {decision.confidence}"]

    # Fallback: build a minimal plan from the decision
    from ..models import AnalysisPlan, ExperimentMatrix, ResearchGoal, Risk, TaskBundle, CodingTask, ReproTask, RunTask
    tasks = TaskBundle(
        coding_tasks=_extract_coding_tasks(decision.recommended_actions),
        repro_tasks=_extract_repro_tasks(decision.recommended_actions),
        run_tasks=_extract_run_tasks(decision.recommended_actions),
    )
    hypothesis = (decision.conclusion.rationale[:200] if decision.conclusion and decision.conclusion.rationale
                  else decision.summary[:200])
    fallback = ExperimentPlan(
        goal=ResearchGoal(summary=decision.summary, hypothesis=hypothesis),
        experiment_matrix=ExperimentMatrix(),
        tasks=tasks,
        analysis_plan=AnalysisPlan(),
        risks=[Risk(description=r) for r in decision.risks],
    )
    return fallback, [f"Fallback plan from decision (confidence: {decision.confidence})"]


def _populate_tasks_from_actions(plan: ExperimentPlan, actions: list) -> ExperimentPlan:
    """Ensure experiment_plan.tasks is populated from recommended_actions if empty."""
    from ..models import RecommendedAction
    if not plan.tasks.coding_tasks and not plan.tasks.repro_tasks and not plan.tasks.run_tasks:
        plan.tasks.coding_tasks = _extract_coding_tasks(actions)
        plan.tasks.repro_tasks = _extract_repro_tasks(actions)
        plan.tasks.run_tasks = _extract_run_tasks(actions)
    return plan


def _extract_coding_tasks(actions: list) -> list:
    from ..models import CodingTask
    tasks = []
    for i, a in enumerate(actions):
        if a.type == "coding_task":
            p = a.plan
            tasks.append(CodingTask(
                id=f"code_{i+1:03d}",
                workspace_path="",  # ResAgent resolves at dispatch (ExpAgent emits no physical paths)
                task_goal=p.task_goal or p.experiment_goal or a.rationale[:100],
                constraints=p.constraints,
                verify_commands=p.verify_commands,
                expected_artifacts=p.expected_artifacts,
                rationale=a.rationale,
            ))
    return tasks


def _extract_repro_tasks(actions: list) -> list:
    from ..models import ReproTask, ComputeBudget
    tasks = []
    for i, a in enumerate(actions):
        if a.type == "repro_task":
            p = a.plan
            tasks.append(ReproTask(
                id=f"repro_{i+1:03d}",
                paper_url=p.paper_url,
                repo_url=p.repo_url,
                experiment_goal=p.experiment_goal or a.rationale,
                compute_budget=p.compute_budget,
                expected_metrics=p.expected_metrics,
                rationale=a.rationale,
            ))
    return tasks


def _extract_run_tasks(actions: list) -> list:
    from ..models import RunTask
    tasks = []
    for i, a in enumerate(actions):
        if a.type == "run_task":
            p = a.plan
            tasks.append(RunTask(
                id=f"run_{i+1:03d}",
                command_goal=p.command_goal or p.experiment_goal or a.rationale[:100],
                expected_runtime=p.expected_runtime,
                requires_gpu=p.requires_gpu,
                rationale=a.rationale,
            ))
    return tasks


def revise(
    current_plan: ExperimentPlan,
    feedback: str,
    *,
    model: str = "deepseek-chat",
    api_base: str = "https://api.deepseek.com/v1",
    api_key_env: str = "DEEPSEEK_API_KEY",
    mock: bool = False,
    trace_dir: Path | None = None,
    run_dir: Path | None = None,
) -> tuple[ExperimentPlan, list[str]]:
    """Revise an existing experiment plan based on user feedback.

    Delegates to advise() internally.
    """
    situation = build_revise_prompt(current_plan, feedback)
    ctx = AdvisorContext(situation=situation, existing_plan=current_plan)

    decision, trace = advise(
        ctx,
        model=model,
        api_base=api_base,
        api_key_env=api_key_env,
        mock=mock,
        trace_dir=trace_dir,
        run_dir=run_dir,
    )

    if decision.experiment_plan:
        return _populate_tasks_from_actions(decision.experiment_plan, decision.recommended_actions), [f"confidence: {decision.confidence}"]

    return current_plan, ["Revision did not produce a new experiment plan"]


def _extract_yaml(text: str) -> str:
    """Extract YAML content from an LLM response (backward-compat re-export)."""
    if "```yaml" in text:
        start = text.index("```yaml") + len("```yaml")
        try:
            end = text.index("```", start)
            return text[start:end].strip()
        except ValueError:
            return text[start:].strip()
    if "```" in text:
        start = text.index("```") + 3
        remaining = text[start:]
        nl = remaining.find("\n")
        if nl != -1:
            start = start + nl + 1
        try:
            end = text.index("```", start)
            return text[start:end].strip()
        except ValueError:
            return text[start:].strip()
    return text.strip()
