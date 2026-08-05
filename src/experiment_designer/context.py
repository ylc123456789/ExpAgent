"""Structured state and prompt building for the ExpAgent loop.

Style-aligned with CodingAgent's context.py:
- LoopState holds all mutable loop state
- Each turn rebuilds the user prompt FROM SCRATCH from state
- Compressed history preserves search queries and key results
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .context_policy import ContextPolicy


@dataclass
class LoopState:
    """Mutable state for the agentic loop, rebuilt into a fresh prompt each turn."""

    situation: str                                    # original task description
    compressed: list[str] = field(default_factory=list)   # step history (one-line each)
    paper_index: list[dict] = field(default_factory=list)  # saved paper entries
    file_cache: dict[str, str] = field(default_factory=dict)  # path → content


def build_initial_prompt(state: LoopState) -> str:
    """Build the very first user prompt — called once at loop start."""
    parts = [state.situation]
    parts.append("\n---\nAnalyze the situation. Use search_papers to find baselines, read_file to inspect artifacts, then finish with your ScientificDecision.")
    return "\n".join(parts)


def build_turn_prompt(state: LoopState, policy: ContextPolicy) -> str:
    """Build a fresh user prompt from structured state each turn.

    CodingAgent style: the full prompt is rebuilt from state, not appended
    to a growing messages array. FC tool_pairs are handled separately in advisor.py.
    """
    parts: list[str] = [state.situation]

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
