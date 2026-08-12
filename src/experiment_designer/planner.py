"""Backward-compatible re-export of plan/revise.

Implementation moved to controller/planner.py.
"""

from .controller.planner import _extract_yaml, plan, revise

__all__ = ["plan", "revise", "_extract_yaml"]
