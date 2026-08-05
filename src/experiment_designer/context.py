"""Structured state and prompt building for the ExpAgent loop.

Style-aligned with CodingAgent's context.py:
- AdvisorState holds all loop state (situation, papers, steps, file cache)
- Each turn rebuilds the user prompt FROM SCRATCH from state
- Old steps are compressed to one-line summaries (keeping paper references)
- PaperIndex stays in context; full papers on disk → read_file on demand
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .context_policy import ContextPolicy


# ── Paper index (always in context) ──────────────────────────────


@dataclass
class PaperEntry:
    """One paper in the index — lightweight, always in context."""

    title: str
    first_author: str
    year: int | None
    one_liner: str             # one-sentence summary by the LLM
    paper_id: str              # arxiv id, DOI, or semantic scholar id
    code_url: str = ""         # known code repository, if any
    slug: str = ""             # file-safe name for paper library


class PaperIndex:
    """Growing index of saved papers. Injected into every turn's context."""

    def __init__(self) -> None:
        self.entries: list[PaperEntry] = []

    def add(self, entry: PaperEntry) -> None:
        # Avoid duplicates by paper_id
        if not any(e.paper_id == entry.paper_id for e in self.entries):
            self.entries.append(entry)

    def to_context(self, policy: ContextPolicy) -> str:
        """Format the index for injection into the user prompt."""
        if not self.entries:
            return ""
        shown = self.entries[-policy.paper_index_entries:]
        lines = ["## Paper Index", ""]
        for i, e in enumerate(shown, 1):
            year_str = f" ({e.year})" if e.year else ""
            code_str = f"  code: {e.code_url}" if e.code_url else ""
            lines.append(
                f"[{i}] {e.title}{year_str} · {e.first_author} et al."
                f"\n    {e.one_liner[:policy.paper_index_summary_chars]}"
                f"\n    paper: {e.paper_id}{code_str}"
                f"\n    full metadata: papers/{e.slug}.md"
            )
        return "\n".join(lines)


# ── Step records ────────────────────────────────────────────────


@dataclass
class AdvisorStep:
    """Record of one loop step — action + observation."""

    step: int
    action: str                         # search_papers | read_file | finish | api_error
    observation: str                    # result text (may be long)
    papers_saved: list[str] = field(default_factory=list)  # paper_ids saved this step


# ── State ───────────────────────────────────────────────────────


@dataclass
class AdvisorState:
    """Structured state rebuilt into a fresh prompt each turn.

    Mirrors CodingAgent's AgentState pattern.
    """

    situation: str
    artifacts_summary: str = ""         # brief summary of available artifacts
    existing_plan_yaml: str = ""        # serialized ExperimentPlan, if any
    paper_index: PaperIndex = field(default_factory=PaperIndex)
    file_cache: dict[str, str] = field(default_factory=dict)   # path → content tail
    steps: list[AdvisorStep] = field(default_factory=list)

    @classmethod
    def from_advisor_context(cls, ctx: "AdvisorContext") -> "AdvisorState":
        """Build initial state from the input AdvisorContext."""
        import yaml as _yaml

        artifacts_lines: list[str] = []
        for a in ctx.artifacts:
            artifacts_lines.append(f"- {a.id} ({a.type}): {a.summary}")
            if a.path:
                artifacts_lines.append(f"  path: {a.path}")

        plan_yaml = ""
        if ctx.existing_plan:
            plan_yaml = _yaml.dump(
                ctx.existing_plan.model_dump(exclude_defaults=False),
                allow_unicode=True, sort_keys=False,
            )

        return cls(
            situation=ctx.situation,
            artifacts_summary="\n".join(artifacts_lines),
            existing_plan_yaml=plan_yaml,
        )


# ── Prompt builders (CodingAgent style: build fresh from state) ─


def build_initial_context(state: AdvisorState) -> str:
    """Build the very first user prompt — called once at loop start."""
    parts: list[str] = []

    parts.append("## Situation")
    parts.append(state.situation)

    if state.artifacts_summary:
        parts.append("\n## Available Artifacts")
        parts.append("Use read_file to see detailed results.")
        parts.append(state.artifacts_summary)

    if state.existing_plan_yaml:
        parts.append("\n## Current Experiment Plan")
        parts.append("```yaml")
        parts.append(state.existing_plan_yaml)
        parts.append("```")

    parts.append("\n---")
    parts.append("Analyze the situation. Use search_papers to find baselines or related work, read_file to inspect artifacts, then finish with your ScientificDecision.")

    return "\n".join(parts)


def build_turn_context(state: AdvisorState, policy: ContextPolicy) -> str:
    """Build a fresh user prompt from structured state.

    Equivalent to CodingAgent's build_turn_prompt() and ReproAgent's
    build_turn_prompt(): the state is rebuilt into a single user message
    from scratch each turn.  No message-array mutation.
    """
    parts: list[str] = []

    # 1. Situation
    parts.append("## Situation")
    parts.append(state.situation)

    # 2. Paper Index (always in context — lightweight entries)
    paper_text = state.paper_index.to_context(policy)
    if paper_text:
        parts.append("\n" + paper_text)

    # 3. File cache (recently read files — tail only)
    if state.file_cache:
        entries = list(state.file_cache.items())[-policy.file_cache_count:]
        parts.append("\n## Recently Read Files")
        for path, text in entries:
            tail = text[-policy.file_cache_chars:]
            parts.append(f"\n### {path} ({len(text)} chars total, showing last {len(tail)})")
            parts.append(tail)

    # 4. Previous steps (compressed to one-line summaries)
    if len(state.steps) > 1:
        compacted = [_compact_step(s, policy) for s in state.steps[:-1]]
        shown = compacted[-policy.step_history:]
        parts.append("\n## Previous Steps")
        for line in shown:
            parts.append(line)

    # 5. Last result (full, not compressed — what just happened)
    if state.steps:
        parts.append("\n## Last Result")
        parts.append(_format_step_full(state.steps[-1], policy))

    parts.append("\nWhat is your next action?")
    return "\n".join(parts)


def build_final_pressure(policy: ContextPolicy) -> str:
    """Return a message to append when the step budget is nearly gone."""
    return (
        "[GRACE STOP: Very few steps remain. You MUST call finish now "
        "with your best scientific judgment. Prioritize the conclusion "
        "and recommended_actions. You may skip optional searches.]"
    )


# ── Step compression (keeps paper references) ────────────────────


def _compact_step(step: AdvisorStep, policy: ContextPolicy) -> str:
    """Compress one step to a single line, preserving paper references."""
    obs_tail = step.observation.strip()[-policy.observation_chars:]

    if step.action == "search_papers":
        # Include what was searched — prevents infinite search loops
        import re
        m = re.search(r"Found (\d+) papers?", obs_tail)
        count = m.group(1) if m else "?"
        # Extract first paper title as hint
        title_m = re.search(r"\d+\. (.+?)(?:\s*\(|\s*\[|$)", obs_tail)
        hint = title_m.group(1)[:80] if title_m else "?"
        saved = f", saved: {', '.join(step.papers_saved)}" if step.papers_saved else ""
        return f"- Step {step.step} search: found {count}p on '{hint}'{saved}"

    if step.action == "read_file":
        # Show the file path and first meaningful content
        snippet = obs_tail[:200].replace("\n", " ")
        return f"- Step {step.step} read_file: {snippet}"

    if step.action == "finish":
        return f"- Step {step.step} finish: {obs_tail[:200]}"

    if step.action == "api_error":
        return f"- Step {step.step} api_error: {obs_tail[:200]}"

    return f"- Step {step.step} {step.action}: {obs_tail[:200]}"


def _format_step_full(step: AdvisorStep, policy: ContextPolicy) -> str:
    """Format the most recent step in full (not compressed)."""
    return step.observation[-policy.last_result_chars:]
