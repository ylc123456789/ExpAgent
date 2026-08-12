"""System prompt, tool schemas, and turn-prompt builders.

Split across three modules by content:
- system.py: the system-level instruction string
- schemas.py: Function Calling tool schemas (JSON-Schema-like dicts)
- rendering.py: prompt builders that render structured state into text
"""

from .schemas import TOOLS
from .system import SYSTEM_PROMPT
from .rendering import (
    build_initial_prompt,
    build_plan_prompt,
    build_revise_prompt,
    build_turn_prompt,
)

__all__ = [
    "TOOLS",
    "SYSTEM_PROMPT",
    "build_initial_prompt",
    "build_plan_prompt",
    "build_revise_prompt",
    "build_turn_prompt",
]
