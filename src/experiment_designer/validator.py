"""Deterministic safety-net validation for experiment plans.

Checks that the LLM didn't skip essential fields. These are structural
completeness checks — not semantic review (the LLM handles that).
"""

from __future__ import annotations

from .models import ExperimentPlan, ValidationResult


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
        if not t.repo_path.strip():
            issues.append(f"coding_tasks[{i}] ({t.id}): repo_path is empty")

    for i, t in enumerate(repro):
        if not t.id.strip():
            issues.append(f"repro_tasks[{i}].id is empty")
        if not t.paper_url.strip():
            issues.append(f"repro_tasks[{i}] ({t.id}): paper_url is empty")
        if not t.repo_url.strip():
            issues.append(f"repro_tasks[{i}] ({t.id}): repo_url is empty")
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
