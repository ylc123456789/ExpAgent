"""Core planning engine — LLM-driven experiment plan generation and revision."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .llm import call_llm
from .models import DesignInput, ExperimentPlan, ValidationResult
from .prompts import (
    REVISE_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_plan_prompt,
    build_revise_prompt,
)
from .validator import validate

# Maximum retries for LLM calls (parse failure or validation failure).
MAX_RETRIES = 2


def plan(
    inp: DesignInput,
    *,
    model: str = "deepseek-chat",
    api_base: str = "https://api.deepseek.com/v1",
    api_key_env: str = "DEEPSEEK_API_KEY",
    mock: bool = False,
    trace_dir: Path | None = None,
) -> tuple[ExperimentPlan, list[str]]:
    """Generate a new experiment plan from a research idea.

    Args:
        inp: The design input containing research idea, constraints, etc.
        model: LLM model name.
        api_base: API base URL.
        api_key_env: Env var holding the API key.
        mock: If True, use deterministic mock instead of real LLM.
        trace_dir: Optional directory for LLM trace files.

    Returns:
        Tuple of (experiment plan, list of diagnostic messages from retries).
    """
    system = SYSTEM_PROMPT
    user = build_plan_prompt(inp)
    return _generate(system, user, model=model, api_base=api_base,
                     api_key_env=api_key_env, mock=mock, trace_dir=trace_dir,
                     trace_label="plan")


def revise(
    current_plan: ExperimentPlan,
    feedback: str,
    *,
    model: str = "deepseek-chat",
    api_base: str = "https://api.deepseek.com/v1",
    api_key_env: str = "DEEPSEEK_API_KEY",
    mock: bool = False,
    trace_dir: Path | None = None,
) -> tuple[ExperimentPlan, list[str]]:
    """Revise an existing experiment plan based on user feedback.

    Args:
        current_plan: The current experiment plan to modify.
        feedback: User's natural language feedback on what to change.
        model: LLM model name.
        api_base: API base URL.
        api_key_env: Env var holding the API key.
        mock: If True, use deterministic mock instead of real LLM.
        trace_dir: Optional directory for LLM trace files.

    Returns:
        Tuple of (revised experiment plan, list of diagnostic messages).
    """
    system = REVISE_SYSTEM_PROMPT
    user = build_revise_prompt(current_plan, feedback)
    return _generate(system, user, model=model, api_base=api_base,
                     api_key_env=api_key_env, mock=mock, trace_dir=trace_dir,
                     trace_label="revise")


# ── Internal ─────────────────────────────────────────────────────


def _generate(
    system: str,
    user: str,
    *,
    model: str,
    api_base: str,
    api_key_env: str,
    mock: bool,
    trace_dir: Path | None,
    trace_label: str,
) -> tuple[ExperimentPlan, list[str]]:
    """Shared generation loop with retry logic."""
    diags: list[str] = []

    for attempt in range(1, MAX_RETRIES + 2):
        raw = call_llm(
            model=model,
            api_base=api_base,
            api_key_env=api_key_env,
            system=system,
            user=user,
            mock=mock,
            trace_dir=trace_dir,
            trace_label=f"{trace_label}_attempt{attempt}",
        )

        # Extract YAML from the LLM response
        yaml_text = _extract_yaml(raw)

        # Try to parse
        try:
            data = yaml.safe_load(yaml_text)
        except yaml.YAMLError as e:
            msg = f"Attempt {attempt}: YAML parse error — {e}"
            diags.append(msg)
            if attempt <= MAX_RETRIES:
                user = _retry_prompt(user, f"YAML parse error: {e}\nPlease output valid YAML only.")
                continue
            raise RuntimeError(f"Failed to parse LLM output after {attempt} attempts.\n{msg}\n\nRaw output:\n{raw}")

        if not isinstance(data, dict):
            msg = f"Attempt {attempt}: LLM output is not a YAML dictionary (got {type(data).__name__})"
            diags.append(msg)
            if attempt <= MAX_RETRIES:
                user = _retry_prompt(user, "Output was not a valid YAML document. Please output a complete YAML document.")
                continue
            raise RuntimeError(f"Failed to get valid YAML from LLM after {attempt} attempts.\n{msg}")

        # Try Pydantic parsing
        try:
            plan_obj = ExperimentPlan.model_validate(data)
        except Exception as e:
            msg = f"Attempt {attempt}: Pydantic validation error — {e}"
            diags.append(msg)
            if attempt <= MAX_RETRIES:
                user = _retry_prompt(user, f"Schema validation error: {e}\nPlease fix the YAML to match the required schema.")
                continue
            raise RuntimeError(f"Failed to validate plan schema after {attempt} attempts.\n{msg}")

        # Deterministic safety-net validation
        vr: ValidationResult = validate(plan_obj)
        if vr.status == "needs_revision" and attempt <= MAX_RETRIES:
            issue_list = "\n".join(f"- {i}" for i in vr.issues)
            msg = f"Attempt {attempt}: validation issues — {vr.issues}"
            diags.append(msg)
            user = _retry_prompt(user, f"Plan validation found these issues:\n{issue_list}\nPlease fix them and output the complete corrected YAML.")
            continue

        # If validation still fails on last attempt, return the plan anyway
        # with the issues recorded — caller can decide what to do.
        if vr.status == "needs_revision":
            diags.append(f"Final: validation issues remain — {vr.issues}")

        return plan_obj, diags

    # Should be unreachable
    raise RuntimeError("Unexpected: retry loop exhausted")


def _extract_yaml(text: str) -> str:
    """Extract YAML content from an LLM response.

    Handles:
    - ```yaml ... ``` blocks
    - ``` ... ``` blocks
    - ```yaml ... (no closing fence — LLM truncated)
    - Raw YAML (no code fence)
    """
    # Try fenced yaml block first
    if "```yaml" in text:
        start = text.index("```yaml") + len("```yaml")
        try:
            end = text.index("```", start)
            return text[start:end].strip()
        except ValueError:
            # No closing fence — take everything after the opening fence
            return text[start:].strip()

    # Try generic fenced block
    if "```" in text:
        start = text.index("```") + 3
        # Skip optional language tag on the opening fence
        remaining = text[start:]
        nl = remaining.find("\n")
        if nl != -1:
            start = start + nl + 1
        try:
            end = text.index("```", start)
            return text[start:end].strip()
        except ValueError:
            return text[start:].strip()

    # Assume raw YAML
    return text.strip()


def _retry_prompt(original_user: str, error_detail: str) -> str:
    """Append error feedback to the user prompt for retry."""
    return f"{original_user}\n\n[SYSTEM NOTE: Previous attempt failed.\n{error_detail}]"
