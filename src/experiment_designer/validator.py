"""Backward-compatible re-export of validate/validate_decision.

Implementation moved to controller/validator.py.
"""

from .controller.validator import validate, validate_decision

__all__ = ["validate", "validate_decision"]
