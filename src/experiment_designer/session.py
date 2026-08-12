"""Session card, run state, and artifact discovery.

These are the tracking outputs of an ExpAgent run: session.yaml (the
cross-module contract) and state.json (the full step-by-step record).
"""

from __future__ import annotations

from pathlib import Path


def write_session_card(
    run_dir: Path,
    *,
    session_id: str = "",
    status: str = "completed",
    summary: str = "",
    parent: dict | None = None,
) -> Path:
    """Write session.yaml in run_dir — the only cross-module contract.

    All fields follow §3 of the session/project model spec.
    """
    import uuid as _uuid
    from datetime import datetime, timezone

    run_dir.mkdir(parents=True, exist_ok=True)

    if not session_id:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        session_id = f"exp-{stamp}-{_uuid.uuid4().hex[:6]}"

    now = datetime.now(timezone.utc).isoformat()
    card = {
        "schema_version": 1,
        "session_id": session_id,
        "module": "expagent",
        "kind": "advisory_session",
        "status": status,
        "created_at": now,
        "updated_at": now,
        "parent": parent,
        "project_path": str(run_dir.resolve()),
        "summary": summary[:500],
        "key_artifacts": _key_artifacts(run_dir),
        "bindings": {},
    }

    import yaml as _yaml
    path = run_dir / "session.yaml"
    path.write_text(_yaml.dump(card, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def _key_artifacts(run_dir: Path) -> list[dict]:
    """Discover key artifacts in the run directory for the session card."""
    artifacts: list[dict] = []
    decision_json = run_dir / "scientific_decision.json"
    if decision_json.exists():
        artifacts.append({"type": "scientific_decision", "path": "scientific_decision.json",
                          "summary": "ExpAgent scientific decision output"})
    papers_dir = run_dir / "papers"
    if papers_dir.exists():
        paper_files = list(papers_dir.glob("*.md"))
        if paper_files:
            artifacts.append({"type": "paper_library", "path": "papers/",
                              "summary": f"{len(paper_files)} saved paper(s)"})
    return artifacts


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
