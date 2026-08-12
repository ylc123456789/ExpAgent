"""Backward-compatible re-export of the public advise() API.

The agentic loop moved to agent.py (advise) and controller/loop.py (_run_loop).
ResAgent's adapter imports advise from this path, so it is kept as a thin
forwarder rather than removed.
"""

from .agent import advise

__all__ = ["advise"]
