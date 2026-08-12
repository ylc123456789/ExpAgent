"""Loop state for the ExpAgent agentic loop.

Style-aligned with CodingAgent's context.py:
- LoopState holds all mutable loop state
- State is rebuilt into a fresh prompt each turn (prompt logic in prompts/rendering.py)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LoopState:
    """Mutable state for the agentic loop, rebuilt into a fresh prompt each turn."""

    situation: str                                    # original task description
    compressed: list[str] = field(default_factory=list)   # step history (one-line each)
    paper_index: list[dict] = field(default_factory=list)  # saved paper entries
    findings: list[dict] = field(default_factory=list)     # note_finding records
    file_cache: dict[str, str] = field(default_factory=dict)  # path → content
