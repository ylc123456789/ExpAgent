"""Deterministic safety-net validation for experiment plans and scientific decisions.

Checks that the LLM didn't skip essential fields. These are structural
completeness checks — not semantic review (the LLM handles that).
"""

from __future__ import annotations

from ..models import ExperimentPlan, ScientificDecision, ValidationResult


def validate(plan: ExperimentPlan) -> ValidationResult:
    """Run all deterministic checks on an experiment plan.

    Returns ValidationResult with status="ok" if all checks pass,
    or status="needs_revision" with a list of specific issues.
    """
    issues: list[str] = []

    # ── Goal checks ─────────────────────────────────────────────
    if not plan.goal.summary.strip():
        issues.append("goal.summary is empty")
    if not plan.goal.hypothesis.strip():
        issues.append("goal.hypothesis is empty — must state what is being tested")
    if len(plan.goal.hypothesis.strip()) < 20:
        issues.append("goal.hypothesis is too short (< 20 chars) — needs a specific claim")
    if not plan.goal.success_criteria:
        issues.append("goal.success_criteria is empty — must define how to measure success")
    else:
        for i, sc in enumerate(plan.goal.success_criteria):
            if not sc.strip():
                issues.append(f"goal.success_criteria[{i}] is empty")

    # ── Experiment matrix checks ────────────────────────────────
    if not plan.experiment_matrix.datasets:
        issues.append("experiment_matrix.datasets is empty — must specify at least one dataset")
    else:
        for i, ds in enumerate(plan.experiment_matrix.datasets):
            if not ds.name.strip():
                issues.append(f"experiment_matrix.datasets[{i}].name is empty")
            if not ds.rationale.strip():
                issues.append(f"experiment_matrix.datasets[{i}] ({ds.name}) has no rationale")

    if not plan.experiment_matrix.methods:
        issues.append("experiment_matrix.methods is empty — must include at least one method")
    else:
        has_baseline = any(m.type == "baseline" for m in plan.experiment_matrix.methods)
        if not has_baseline:
            issues.append("experiment_matrix.methods has no baseline — must include at least one baseline method for comparison")
        has_proposed = any(m.type == "new_method" for m in plan.experiment_matrix.methods)
        if not has_proposed:
            issues.append("experiment_matrix.methods has no new_method — must include the proposed method being tested")
        for i, m in enumerate(plan.experiment_matrix.methods):
            if not m.name.strip():
                issues.append(f"experiment_matrix.methods[{i}].name is empty")
            if not m.rationale.strip():
                issues.append(f"experiment_matrix.methods[{i}] ({m.name}) has no rationale")
            if m.type not in ("new_method", "baseline", "ablation"):
                issues.append(f"experiment_matrix.methods[{i}] ({m.name}) has invalid type: {m.type}")

    if not plan.experiment_matrix.metrics:
        issues.append("experiment_matrix.metrics is empty — must specify at least one evaluation metric")
    else:
        for i, m in enumerate(plan.experiment_matrix.metrics):
            if not m.name.strip():
                issues.append(f"experiment_matrix.metrics[{i}].name is empty")

    # ── Task checks ─────────────────────────────────────────────
    coding = plan.tasks.coding_tasks
    repro = plan.tasks.repro_tasks
    run = plan.tasks.run_tasks

    if not coding and not repro and not run:
        issues.append("tasks is completely empty — must have at least one coding, repro, or run task")

    for i, t in enumerate(coding):
        if not t.id.strip():
            issues.append(f"coding_tasks[{i}].id is empty")
        if not t.task_goal.strip():
            issues.append(f"coding_tasks[{i}] ({t.id}): task_goal is empty")
        if not t.rationale.strip():
            issues.append(f"coding_tasks[{i}] ({t.id}): rationale is empty")
        if not t.workspace_path.strip() and not t.task_goal.strip():
            issues.append(f"coding_tasks[{i}] ({t.id}): both workspace_path and task_goal are empty")

    for i, t in enumerate(repro):
        if not t.id.strip():
            issues.append(f"repro_tasks[{i}].id is empty")
        if not t.paper_url.strip():
            issues.append(f"repro_tasks[{i}] ({t.id}): paper_url is empty")
        if not t.experiment_goal.strip():
            issues.append(f"repro_tasks[{i}] ({t.id}): experiment_goal is empty")
        if not t.rationale.strip():
            issues.append(f"repro_tasks[{i}] ({t.id}): rationale is empty")

    for i, t in enumerate(run):
        if not t.id.strip():
            issues.append(f"run_tasks[{i}].id is empty")
        if not t.command_goal.strip():
            issues.append(f"run_tasks[{i}] ({t.id}): command_goal is empty")
        if not t.rationale.strip():
            issues.append(f"run_tasks[{i}] ({t.id}): rationale is empty")

    # ── Risk checks ─────────────────────────────────────────────
    if not plan.risks:
        issues.append("risks is empty — must identify at least one risk with mitigation")
    else:
        for i, r in enumerate(plan.risks):
            if not r.description.strip():
                issues.append(f"risks[{i}].description is empty")

    # ── Analysis plan checks ────────────────────────────────────
    if not plan.analysis_plan.comparisons and not plan.analysis_plan.plots:
        issues.append("analysis_plan has no comparisons or plots — should describe what to compare and visualize")

    status = "needs_revision" if issues else "ok"
    return ValidationResult(status=status, issues=issues)


def _validate_action_dependencies(decision: ScientificDecision) -> list[str]:
    """Validate action_id uniqueness and depends_on references within a decision.

    Rules:
    - Every action must have a non-empty action_id.
    - action_id values must be unique across all actions.
    - Every id in depends_on must reference a valid action_id in the same decision.
    - A dependency must point to an EARLIER action, keeping the graph acyclic.
    """
    issues: list[str] = []
    actions = decision.recommended_actions
    if not actions:
        return issues

    # Non-empty action_id check
    for i, a in enumerate(actions):
        if not a.action_id.strip():
            issues.append(
                f"recommended_actions[{i}] has an empty action_id — every action "
                f"must have a unique, non-empty action_id"
            )

    # Collect all non-empty action_ids
    all_ids = [a.action_id for a in actions if a.action_id.strip()]

    # Duplicate check
    seen: set[str] = set()
    for aid in all_ids:
        if aid in seen:
            issues.append(f"Duplicate action_id: '{aid}' — each action_id must be unique within a decision")
        seen.add(aid)

    # Position of each action_id (first occurrence) for the ordering check
    id_position: dict[str, int] = {}
    for i, a in enumerate(actions):
        if a.action_id.strip() and a.action_id not in id_position:
            id_position[a.action_id] = i

    # Reference + ordering (acyclic) check
    for i, action in enumerate(actions):
        for dep_id in action.depends_on:
            if dep_id not in id_position:
                issues.append(
                    f"recommended_actions[{i}] depends_on unknown action_id '{dep_id}' — "
                    f"must reference an action_id defined in the same decision"
                )
            elif id_position[dep_id] >= i:
                issues.append(
                    f"recommended_actions[{i}] depends_on '{dep_id}' must reference an "
                    f"EARLIER action (found at index {id_position[dep_id]}) to keep the graph acyclic"
                )

    return issues


def _validate_analysis_coverage(decision: ScientificDecision) -> list[str]:
    """Ensure terminal experiments are covered by analyze_results when required.

    When analysis_required is True (the default), every experiment action
    (execute_experiment / reproduce_experiment) must be depended on by at
    least one analyze_results action.
    """
    issues: list[str] = []
    if not decision.analysis_required:
        return issues

    experiment_ids = [
        a.action_id for a in decision.recommended_actions
        if a.capability in ("execute_experiment", "reproduce_experiment")
    ]
    if not experiment_ids:
        return issues

    covered: set[str] = set()
    for a in decision.recommended_actions:
        if a.capability == "analyze_results":
            covered.update(a.depends_on)

    for eid in experiment_ids:
        if eid not in covered:
            issues.append(
                f"experiment action '{eid}' has no analyze_results coverage — "
                f"analysis_required is true, so every experiment must be analyzed"
            )
    return issues


def validate_decision(decision: ScientificDecision) -> ValidationResult:
    """Validate a ScientificDecision for structural completeness.

    Checks differ from validate() because ScientificDecision is a different
    kind of output — it's advice, not an experiment plan.
    """

    issues: list[str] = []

    # Summary
    if not decision.summary.strip():
        issues.append("summary is empty")

    # Conclusion: optional for pure explanation/discussion requests
    if decision.conclusion is not None:
        if not decision.conclusion.rationale.strip():
            issues.append("conclusion.rationale is empty — must explain the scientific reasoning")
        if not decision.conclusion.status:
            issues.append("conclusion.status is empty")
    else:
        if len(decision.summary.strip()) < 50:
            issues.append("summary too short for conclusion=None — must be at least 50 chars")

    # Evidence
    if not decision.evidence:
        issues.append("evidence is empty — must cite at least one piece of evidence supporting the conclusion")

    # Recommended actions — can be empty but should explain why
    if not decision.recommended_actions and decision.conclusion is not None:
        if "no action" not in decision.conclusion.rationale.lower():
            issues.append("recommended_actions is empty — if no actions are needed, explain why in the conclusion")

    # Each action's capability-specific fields
    for i, action in enumerate(decision.recommended_actions):
        if not action.rationale.strip():
            issues.append(f"recommended_actions[{i}]: rationale is empty")
        if not action.objective.strip():
            issues.append(f"recommended_actions[{i}] ({action.capability}): objective is empty")
        cap = action.capability
        if cap == "reproduce_experiment":
            if not action.paper_url.strip():
                issues.append(f"recommended_actions[{i}] (reproduce_experiment): paper_url is empty")
            if not action.repo_url.strip():
                issues.append(f"recommended_actions[{i}] (reproduce_experiment): repo_url is empty")
        elif cap == "execute_experiment":
            if not action.expected_metrics and not action.success_criteria:
                issues.append(f"recommended_actions[{i}] (execute_experiment): must include expected_metrics or success_criteria")
        elif cap == "search_literature":
            if not action.search_query.strip():
                issues.append(f"recommended_actions[{i}] (search_literature): search_query is empty")
        elif cap == "analyze_results":
            if not action.depends_on:
                issues.append(f"recommended_actions[{i}] (analyze_results): must depend on at least one experiment action")
        elif cap == "ask_user":
            if not action.question.strip():
                issues.append(f"recommended_actions[{i}] (ask_user): question is empty")

    # Risks
    if not decision.risks:
        issues.append("risks is empty — must identify at least one scientific risk")

    # Action dependency metadata + analysis coverage
    issues.extend(_validate_action_dependencies(decision))
    issues.extend(_validate_analysis_coverage(decision))

    # Experiment plan (when present, must be complete)
    if decision.experiment_plan is not None:
        ep = decision.experiment_plan
        if not ep.goal.success_criteria:
            issues.append("experiment_plan.goal.success_criteria is empty")
        if not ep.experiment_matrix.datasets:
            issues.append("experiment_plan.experiment_matrix.datasets is empty")
        if not ep.experiment_matrix.methods:
            issues.append("experiment_plan.experiment_matrix.methods is empty")
        if not ep.experiment_matrix.metrics:
            issues.append("experiment_plan.experiment_matrix.metrics is empty")
        if not ep.tasks.coding_tasks and not ep.tasks.repro_tasks and not ep.tasks.run_tasks:
            issues.append("experiment_plan.tasks is empty — must have at least one coding, repro, or run task")

    status = "needs_revision" if issues else "ok"
    return ValidationResult(status=status, issues=issues)
