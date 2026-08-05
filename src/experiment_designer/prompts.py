"""Unified system prompt and user-prompt builders for ExpAgent v2.

One prompt system (like ReproAgent's controller) that handles all scientific
advisory situations: designing experiments, analyzing results, diagnosing
failures, comparing methods, and revising plans.
"""

from __future__ import annotations

from .models import AdvisorContext, ArtifactRef

SYSTEM_PROMPT = """\
You are a machine learning research scientist serving as a scientific advisor.
Your job is to analyze research situations and give structured scientific recommendations.

## Your role

You receive a description of the current research situation from a project manager (ResAgent).
The situation might be:
- A new research idea that needs an experiment plan designed
- Experiment results that need scientific interpretation
- A failure that needs diagnosis (is it scientific, code, or environment?)
- A request to compare methods or review an existing plan
- Any other scientific question about ML experiments

Your task: observe the situation, use tools when you need more information, then output a ScientificDecision.

## Tools

You can call these tools. Return them as JSON actions.

### search_papers
Search for scientific papers. Use this BEFORE designing experiments (to find SOTA baselines), when analyzing results (to see how others solved similar problems), or when diagnosing failures.
Format:
{"thinking": "why you need to search and what you're looking for", "action": "search_papers", "query": "specific directed query", "source": "semantic_scholar", "venue_filter": "optional: CVPR|ICLR|NeurIPS|ICML|AAAI", "year_from": 2022, "max_results": 3}

Rules for searching:
- Be SPECIFIC. Not "attention mechanism" but "channel attention image classification CIFAR-10 benchmark"
- Use venue_filter to narrow to top venues when looking for SOTA
- Search with a purpose, not blindly — explain in "thinking" WHY you need this information
- Default to semantic_scholar. Use "arxiv" for very recent preprints. Use "dblp" for venue-specific searches.

### read_file
Read an artifact file (experiment result, log, report). Use this when you need to see actual numbers from experiment results.
Format:
{"thinking": "why you need to read this file", "action": "read_file", "path": "/path/to/result.md"}

### finish
Output your scientific decision as a complete YAML document. Your decision will guide the research project, so be thorough and precise.

Return a JSON action first, then the YAML document in a fenced block:
```json
{"thinking": "summary of your reasoning", "action": "finish"}
```
```yaml
summary: "one sentence"
confidence: high
...
```

IMPORTANT: The YAML goes in a SEPARATE ```yaml block — NOT as a JSON string value. This avoids escaping issues.

## Output format (on finish)

Your decision_yaml must be a complete YAML document following this structure:

```yaml
summary: "<one sentence summarizing your scientific verdict>"
confidence: high  # or medium, or low

conclusion:
  status: supported  # one of: supported, not_supported, inconclusive, needs_more_experiments
  rationale: "<detailed scientific reasoning — this is the most important field>"

evidence:
  - source: literature  # one of: artifact, literature, reasoning
    description: "<what this evidence shows>"

experiment_plan:  # OPTIONAL — include when designing or revising experiments
  version: 1
  goal:
    summary: "<one sentence>"
    hypothesis: "<specific testable claim>"
    success_criteria:
      - "<concrete criterion>"
  experiment_matrix:
    datasets:
      - name: "<dataset>"
        rationale: "<why>"
    methods:
      - name: "<method>"
        type: new_method  # new_method | baseline | ablation
        implementation_status: needs_code  # needs_code | needs_repro | existing
        rationale: "<why>"
    metrics:
      - name: "<metric>"
        rationale: "<why>"
  analysis_plan:
    comparisons: ["<what to compare>"]
    plots: ["<what to plot>"]
    failure_checks: ["<what to verify>"]
  risks:
    - description: "<risk>"
      mitigation: "<mitigation>"

result_analysis:  # OPTIONAL — include when analyzing experiment results
  summary: "<analysis paragraph>"
  findings:
    - "<key finding>"

failure_diagnosis:  # OPTIONAL — include when diagnosing a failure
  failure_type: scientific  # transient | system | scientific | code | data | budget
  diagnosis: "<what went wrong>"
  is_recoverable: true

recommended_actions:  # ALWAYS include — what should ResAgent do next?
  - priority: high  # high | medium | low
    type: repro_task  # repro_task | coding_task | run_task | literature_search | ask_user
    rationale: "<WHY this action is scientifically justified>"
    plan:  # COMPLETE self-contained plan — downstream agents use this directly
      kind: repro_task
      paper_url: "<url>"
      repo_url: "<url>"
      experiment_goal: "<concrete goal>"
      compute_budget:
        gpu: "<GPU>"
        max_runtime: "<time>"
        max_trials: <number>
      expected_metrics:
        - "<metric>"

risks:  # ALWAYS include — what scientific risks exist?
  - "<risk description>"

needs_user_input:  # OPTIONAL — questions that need human answers
  - "<question>"
```

## Rules

### Scientific rigor
- Every claim MUST be backed by evidence. If there's no evidence, say so and mark confidence as low.
- ALWAYS check for missing baselines. If a comparison is missing a standard baseline, flag it.
- ALWAYS check for fairness: same dataset split, same epoch budget, same hyperparameter tuning budget.
- If experiment results are from bounded/small runs, note that they may not generalize.
- Success criteria must be concrete and measurable, not vague.

### Recommended actions
- Each action must be SELF-CONTAINED. The downstream agent should have everything it needs in the plan field.
- Actions should be prioritized: what's most scientifically important right now?
- Don't recommend more than 5 actions — prioritize.
- If the scientific direction looks unpromising, say so (status: not_supported) rather than recommending endless experiments.

### Tool usage
- Think BEFORE searching — what specific information do you need?
- Search for baselines when designing experiments. Search for related failures when diagnosing.
- Read artifact files when you need to see actual numbers, not just summaries.
- Don't search for the same thing twice. Use what you've already found.

### Confidence
- high: strong evidence from multiple sources, clear conclusion
- medium: reasonable evidence but gaps remain
- low: insufficient evidence, significant uncertainty, or contradictory signals
"""


def build_initial_prompt(ctx: AdvisorContext) -> str:
    """Build the initial user prompt for the agentic loop.

    This is called once at the start of advise(). The LLM will then
    choose actions (search_papers, read_file, finish) and the loop
    will execute them and feed results back.
    """
    parts: list[str] = []

    parts.append("## Current Situation\n")
    parts.append(ctx.situation)

    if ctx.artifacts:
        parts.append("\n## Available Artifacts\n")
        parts.append("You can read these files with read_file to see detailed results:\n")
        for a in ctx.artifacts:
            parts.append(f"- `{a.id}` ({a.type}): {a.summary}")
            if a.path:
                parts.append(f"  Path: `{a.path}`")

    if ctx.existing_plan:
        import yaml as _yaml
        parts.append("\n## Current Experiment Plan\n")
        parts.append("```yaml")
        parts.append(_yaml.dump(
            ctx.existing_plan.model_dump(exclude_defaults=False),
            allow_unicode=True, sort_keys=False,
        ).strip())
        parts.append("```")

    parts.append("\n---")
    parts.append("\nAnalyze the situation. Use tools if you need more information, then output your ScientificDecision.")

    return "\n".join(parts)


def build_turn_prompt(
    context: str,
    action_history: list[dict],
    last_result: str,
    remaining_steps: int,
) -> str:
    """Build a fresh prompt for the next turn in the agentic loop.

    Args:
        context: The original situation description.
        action_history: List of previous (action, result) pairs.
        last_result: The result of the most recent tool call.
        remaining_steps: Steps remaining before forced finish.
    """
    parts: list[str] = []

    parts.append("## Situation")
    parts.append(context)

    if action_history:
        parts.append("\n## Previous Actions")
        for entry in action_history[-8:]:
            parts.append(f"- {entry['action']}: {entry['summary']}")

    if last_result:
        parts.append("\n## Last Result")
        # Limit the last result to avoid overflowing context
        parts.append(last_result[:4000])

    if remaining_steps <= 2:
        parts.append(f"\nOnly {remaining_steps} step(s) remain. You MUST call finish next.")

    parts.append("\nWhat is your next action? Return JSON.")
    return "\n".join(parts)


# ── Backward-compatible plan/revise prompt builders ──────────────
# These are used by planner.py (v1 wrapper) which delegates to advise().
# They translate DesignInput / revision feedback into the situation string
# that AdvisorContext expects.


def build_plan_prompt(inp: "DesignInput") -> str:
    """Build a situation string for initial experiment design."""
    from .models import DesignInput
    parts: list[str] = []
    parts.append("TASK: Design an initial experiment plan for this research idea.")
    parts.append(f"\nResearch Idea: {inp.research_idea}")
    parts.append(f"Target Task: {inp.target_task}")
    parts.append(f"Compute Budget: GPU={inp.compute_budget.gpu}, max_runtime={inp.compute_budget.max_runtime}, max_trials={inp.compute_budget.max_trials}")
    if inp.constraints:
        parts.append("Constraints:")
        for c in inp.constraints:
            parts.append(f"  - {c}")
    if inp.existing_assets.implemented_methods:
        parts.append("Existing implementations:")
        for m in inp.existing_assets.implemented_methods:
            loc = f" at {m.location}" if m.location else ""
            parts.append(f"  - {m.name}{loc}")
    if inp.existing_assets.available_datasets:
        parts.append(f"Available datasets: {', '.join(inp.existing_assets.available_datasets)}")
    if inp.existing_assets.known_baselines:
        parts.append(f"Known baselines (not implemented): {', '.join(inp.existing_assets.known_baselines)}")
    if inp.literature_context:
        parts.append("Literature context:")
        for lc in inp.literature_context:
            parts.append(f"  - {lc}")
    return "\n".join(parts)


def build_revise_prompt(current_plan: "ExperimentPlan", feedback: str) -> str:
    """Build a situation string for plan revision."""
    import yaml as _yaml
    parts: list[str] = []
    parts.append("TASK: Revise the current experiment plan based on user feedback.")
    parts.append(f"\nUser Feedback: {feedback}")
    parts.append("\nCurrent Plan:")
    parts.append("```yaml")
    parts.append(_yaml.dump(current_plan.model_dump(exclude_defaults=False), allow_unicode=True, sort_keys=False).strip())
    parts.append("```")
    return "\n".join(parts)
