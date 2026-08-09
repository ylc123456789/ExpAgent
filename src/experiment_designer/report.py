"""Write experiment plans to disk."""

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


def list_run_files(run_dir: Path) -> list[str]:
    """Return relative paths of all files ExpAgent wrote in a run directory.

    Useful for orchestrators (ResAgent) to discover produced artifacts
    without knowing ExpAgent's internal file layout.
    """
    if not run_dir.exists():
        return []
    result: list[str] = []
    for p in sorted(run_dir.rglob("*")):
        if p.is_file():
            result.append(str(p.relative_to(run_dir)))
    return result


def write_state(
    run_dir: Path,
    situation: str,
    model: str,
    trace: list[dict],
    decision: dict | None,
    paper_index: list[dict],
    findings: list[dict],
) -> Path:
    """Write state.json — full run record with every step visible.

    Matches ReproAgent's state.json pattern: one file tells the whole story.
    """
    import json as _json
    from datetime import datetime, timezone

    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "state.json"

    data = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "task": {"situation": situation, "model": model},
        "steps": [],
    }

    for i, t in enumerate(trace, 1):
        step = {"step": i, "action": t.get("action", "?"), "summary": t.get("summary", "")}
        data["steps"].append(step)

    data["paper_index"] = paper_index
    data["findings"] = findings
    if decision:
        data["result"] = decision

    path.write_text(_json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path


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
