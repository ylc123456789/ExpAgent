"""Turn-prompt builders that render structured state into prompt text.

CodingAgent style: the full prompt is rebuilt from LoopState each turn rather
than appended to a growing messages array.
"""

from __future__ import annotations

from ..context import ContextPolicy, LoopState
from ..models import AdvisorContext


def build_initial_prompt(ctx: AdvisorContext, disable_search: bool = False) -> str:
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
    if disable_search:
        parts.append("\nNOTE: Literature search is disabled for this request. Answer based on your knowledge and the provided artifacts only.")
    else:
        parts.append("\nAnalyze the situation. Use tools if you need more information, then output your ScientificDecision.")

    return "\n".join(parts)


# ── Plan/revise prompt builders ──────────────────────────────────
# Used by controller/planner.py to translate DesignInput → situation string
# for advise().
# They translate DesignInput / revision feedback into the situation string
# that AdvisorContext expects.


def build_plan_prompt(inp: "DesignInput") -> str:
    """Build a situation string for initial experiment design."""
    from ..models import DesignInput
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


def build_turn_prompt(state: LoopState, policy: ContextPolicy) -> str:
    """Build a fresh user prompt from structured state each turn.

    CodingAgent style: the full prompt is rebuilt from state, not appended
    to a growing messages array. FC tool_pairs are handled separately in controller/loop.py.
    """
    parts: list[str] = [state.situation]

    # Findings from reading (LLM's conclusions, not raw text)
    if state.findings:
        parts.append("\n## Findings from Reading")
        shown = state.findings[-policy.paper_index_entries:]
        for i, f in enumerate(shown, 1):
            parts.append(f"[{i}] {f['topic']}")
            parts.append(f"    {f['finding'][:300]}")
            if f.get('source'):
                parts.append(f"    source: {f['source']}")

    # Paper index (always in context — lightweight entries)
    if state.paper_index:
        parts.append("\n## Saved Papers")
        shown = state.paper_index[-policy.paper_index_entries:]
        for i, p in enumerate(shown, 1):
            yr = f" ({p['year']})" if p.get('year') else ""
            parts.append(f"[{i}] {p['title']}{yr} · {p.get('first_author', '?')} et al.")
            parts.append(f"    {p.get('one_liner', '')[:policy.observation_tail]}")
            parts.append(f"    paper: {p.get('paper_id', '')}  file: papers/{p.get('slug', '')}.md")

    # Compressed step history (search queries + key results preserved)
    if state.compressed:
        parts.append("\n## Step History")
        shown = state.compressed[-policy.step_history:]
        for line in shown:
            parts.append(line)

    # File cache (recently read files — tail only)
    if state.file_cache:
        entries = list(state.file_cache.items())[-policy.file_cache_count:]
        parts.append("\n## Recent File Reads")
        for key, text in entries:
            tail = text[-policy.file_cache_chars:]
            parts.append(f"[{key}] ({len(text)} chars, tail {len(tail)}):\n{tail}")

    parts.append("\n---\nWhat is your next action?")
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
