"""System prompts and user prompt builders.

Two modes:
- plan: generate a new experiment plan from scratch
- revise: modify an existing plan based on user feedback
"""

from __future__ import annotations

from .models import DesignInput, ExperimentPlan

# ── System prompts ────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert machine learning experiment designer. Your task is to turn a research idea into a concrete, structured experiment plan.

## How to think

1. Parse the research hypothesis — what exactly is being claimed?
2. Infer the necessary baselines. Don't just list what the user mentioned — use your domain knowledge to identify the standard baselines for this task. If the user already has some implemented, note that.
3. Design ablations that isolate individual factors of the proposed method.
4. Choose datasets: prefer small, standard benchmarks for initial validation, then note larger-scale follow-ups.
5. Define concrete, measurable success criteria.
6. Break work into three task types:
   - coding_tasks: things that need new code (→ CodingAgent)
   - repro_tasks: reproducing existing methods from papers (→ ReproAgent)
   - run_tasks: executing experiments with existing code
7. For every design choice, write a rationale explaining WHY.
8. Identify risks and how to mitigate them.

## Rules

- ALWAYS include at least one baseline method.
- ALWAYS include at least one evaluation metric.
- ALWAYS include at least one dataset.
- Distinguish proposed_method / baseline / ablation in method types.
- If the user has existing assets (code, data, baselines), use them — don't plan duplicate work.
- Prefer small-scale bounded experiments first (few epochs, small dataset), then note what full-scale would look like.
- Each task must have a non-empty rationale.
- Success criteria must be concrete and measurable, not vague.

## Output format

Return exactly ONE YAML document. Wrap it in ```yaml ... ```.

The YAML must follow this structure:

```yaml
version: 1
goal:
  summary: "<one sentence>"
  hypothesis: "<the testable hypothesis>"
  success_criteria:
    - "<concrete measurable criterion 1>"
    - "<concrete measurable criterion 2>"
experiment_matrix:
  datasets:
    - name: "<dataset name>"
      split: "standard"
      rationale: "<why this dataset>"
  methods:
    - name: "<method name>"
      type: "new_method"  # one of: new_method, baseline, ablation
      implementation_status: "needs_code"  # one of: needs_code, needs_repro, existing
      rationale: "<why include this method>"
  metrics:
    - name: "<metric name>"
      rationale: "<why this metric>"
tasks:
  coding_tasks:
    - id: "code_001"
      repo_path: "/path/to/repo"
      task_goal: "<concrete implementation goal>"
      constraints:
        - "<constraint>"
      verify_commands:
        - "<verification command>"
      expected_artifacts:
        - "<artifact path>"
      rationale: "<why this coding task is needed>"
  repro_tasks:
    - id: "repro_001"
      paper_url: "<url>"
      repo_url: "<url>"
      experiment_goal: "<concrete reproduction goal>"
      compute_budget:
        gpu: "<GPU model>"
        max_runtime: "<time estimate>"
        max_trials: <number>
      expected_metrics:
        - "<metric>"
      rationale: "<why this reproduction is needed>"
  run_tasks:
    - id: "run_001"
      command_goal: "<what this run accomplishes>"
      expected_runtime: "<time estimate>"
      requires_gpu: true
      rationale: "<why this run is needed>"
analysis_plan:
  comparisons:
    - "<comparison description>"
  plots:
    - "<desired plot>"
  failure_checks:
    - "<thing to verify before drawing conclusions>"
risks:
  - description: "<risk>"
    mitigation: "<how to handle it>"
```

If some sections have no items (e.g. no repro_tasks needed), use empty lists [].
"""

REVISE_SYSTEM_PROMPT = """\
You are an expert machine learning experiment designer. You are revising an existing experiment plan based on user feedback.

## How to work

1. Read the current experiment plan carefully.
2. Understand the user's feedback — what specific change is requested?
3. Modify ONLY the parts that need to change. Keep everything else intact.
4. If the user asks to add something (a baseline, an ablation, a metric), add it with a proper rationale.
5. If the user asks to remove something, remove it and adjust related parts.
6. If the user asks to change a design choice, update it and update dependent sections.
7. Output the COMPLETE revised plan — not just a diff.

## Rules

- Preserve all fields from the original plan unless the feedback requires changing them.
- Every method, dataset, metric, and task must still have a rationale.
- The revised plan must still pass basic validation (has baseline, metric, dataset, success criteria, risks).
- Don't remove sections just because they're empty — empty lists are fine.

## Output format

Return the complete revised YAML document wrapped in ```yaml ... ```.
"""

# ── User prompt builders ─────────────────────────────────────────


def build_plan_prompt(inp: DesignInput) -> str:
    """Build the user prompt for initial plan generation."""
    parts: list[str] = []

    parts.append("## Research Idea")
    parts.append(inp.research_idea)

    parts.append("\n## Target Task")
    parts.append(inp.target_task)

    parts.append("\n## Compute Budget")
    parts.append(f"GPU: {inp.compute_budget.gpu}")
    parts.append(f"Max runtime per experiment: {inp.compute_budget.max_runtime}")
    parts.append(f"Max trials: {inp.compute_budget.max_trials}")

    if inp.constraints:
        parts.append("\n## Constraints")
        for c in inp.constraints:
            parts.append(f"- {c}")

    if inp.existing_assets.implemented_methods:
        parts.append("\n## Existing Implementations (already have code)")
        for m in inp.existing_assets.implemented_methods:
            loc = f" at {m.location}" if m.location else ""
            desc = f" — {m.description}" if m.description else ""
            parts.append(f"- **{m.name}**{loc}{desc}")

    if inp.existing_assets.available_datasets:
        parts.append("\n## Available Datasets (already downloaded)")
        for d in inp.existing_assets.available_datasets:
            parts.append(f"- {d}")

    if inp.existing_assets.known_baselines:
        parts.append("\n## Known Baselines (results known but code not implemented)")
        for b in inp.existing_assets.known_baselines:
            parts.append(f"- {b}")

    if inp.literature_context:
        parts.append("\n## Literature Context")
        for lc in inp.literature_context:
            parts.append(f"- {lc}")

    parts.append("\n---")
    parts.append("\nDesign the experiment plan. Think carefully about baselines, ablations, and risks. Output the complete YAML.")

    return "\n".join(parts)


def build_revise_prompt(current_plan: ExperimentPlan, feedback: str) -> str:
    """Build the user prompt for plan revision."""
    import yaml as _yaml

    parts: list[str] = []

    parts.append("## Current Experiment Plan\n")
    parts.append("```yaml")
    parts.append(_yaml.dump(current_plan.model_dump(), allow_unicode=True, sort_keys=False).strip())
    parts.append("```")

    parts.append("\n## User Feedback")
    parts.append(feedback)

    parts.append("\n---")
    parts.append("\nRevise the experiment plan based on the feedback above. Output the complete revised YAML.")

    return "\n".join(parts)
