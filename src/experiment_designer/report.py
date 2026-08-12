"""Final output writers: experiment_plan.yaml, scientific_decision.json,
and validation_report.md.

Session tracking (session.yaml, state.json) lives in session.py.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .models import ExperimentPlan, ScientificDecision, ValidationResult


def write_plan(plan: ExperimentPlan, output_dir: Path) -> Path:
    """Write experiment_plan.yaml to output_dir. Creates directory if needed.

    Returns the path to the written file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "experiment_plan.yaml"

    # Use model_dump with exclude_defaults to keep output clean
    data = plan.model_dump(exclude_defaults=False)

    output_path.write_text(
        _represent_yaml(data),
        encoding="utf-8",
    )
    return output_path


def write_validation_report(vr: ValidationResult, output_dir: Path) -> Path:
    """Write validation_report.md to output_dir.

    Returns the path to the written file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "validation_report.md"

    lines = [
        "# Validation Report",
        "",
        f"Status: **{vr.status}**",
        "",
    ]
    if vr.issues:
        lines.append("## Issues")
        for issue in vr.issues:
            lines.append(f"- [ ] {issue}")
    else:
        lines.append("No issues found. ✓")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def write_decision(decision: ScientificDecision, output_dir: Path) -> Path:
    """Write scientific_decision.json to output_dir. Creates directory if needed.

    Returns the path to the written file.
    """
    import json as _json

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "scientific_decision.json"

    data = decision.model_dump(exclude_defaults=False)
    output_path.write_text(
        _json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return output_path


# ── Internal ─────────────────────────────────────────────────────


def _represent_yaml(data: dict) -> str:
    """Serialize a dict to readable YAML with consistent formatting.

    Uses a custom representer to avoid Python-specific tags (!!python/...)
    and produce clean, human-readable output.
    """

    class _CleanDumper(yaml.Dumper):
        """Dumper that avoids Python-specific tags."""

    def _str_representer(dumper, value):
        """Represent multi-line strings as block literals."""
        if "\n" in value:
            return dumper.represent_scalar("tag:yaml.org,2002:str", value, style="|")
        return dumper.represent_scalar("tag:yaml.org,2002:str", value)

    def _list_representer(dumper, data):
        """Represent empty lists as [] on one line."""
        if not data:
            return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)
        return dumper.represent_sequence("tag:yaml.org,2002:seq", data)

    _CleanDumper.add_representer(str, _str_representer)
    _CleanDumper.add_representer(list, _list_representer)

    return yaml.dump(
        data,
        Dumper=_CleanDumper,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    )
