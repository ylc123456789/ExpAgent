"""CLI entry point — argument parsing and dependency assembly.

Interactive REPL logic lives in repl.py; terminal rendering lives in
presentation.py. This module only parses arguments, converts CLI input, and
calls the public API (advise / plan), then exits with the right code.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .agent import advise
from .config import resolve_llm_config
from .controller.planner import plan
from .controller.validator import validate, validate_decision
from .models import (
    AdvisorContext,
    ArtifactRef,
    ComputeBudget,
    DesignInput,
    ExperimentPlan,
)
from .repl import run_repl
from .report import write_decision, write_plan, write_validation_report


def main(argv: list[str] | None = None) -> None:
    """Entry point for ExpAgent CLI."""
    args = _parse_args(argv)

    if args.command == "advise":
        _run_advise(args)
        return

    # Resolve LLM config
    llm = resolve_llm_config(
        config_path=args.config,
        cli_model=args.model,
        cli_api_base=args.api_base,
        cli_api_key_env=args.api_key_env,
    )

    # Build initial DesignInput if --idea provided
    initial_input: DesignInput | None = None
    if args.idea:
        idea_text = _read_idea(args.idea)
        initial_input = DesignInput(
            research_idea=idea_text,
            target_task=args.target_task or "unspecified",
            compute_budget=ComputeBudget(
                gpu=args.gpu,
                max_runtime=args.max_runtime,
                max_trials=args.max_trials,
            ),
            constraints=args.constraint or [],
        )

    # Determine mode
    if args.no_interactive and initial_input:
        _run_direct(
            inp=initial_input,
            output_dir=args.output or _default_output_dir(),
            llm=llm,
            mock=args.mock_llm,
        )
    else:
        run_repl(
            initial_input=initial_input,
            llm=llm,
            mock=args.mock_llm,
            config_path=args.config,
            default_output_dir=_default_output_dir,
        )


# ── Direct mode ───────────────────────────────────────────────────


def _run_direct(
    inp: DesignInput,
    output_dir: Path,
    llm: dict[str, str],
    mock: bool,
) -> None:
    """Non-interactive: generate plan and save to disk."""
    print("Generating experiment plan...")
    plan_obj, diags = plan(
        inp,
        model=llm["model"],
        api_base=llm["api_base"],
        api_key_env=llm["api_key_env"],
        mock=mock,
    )

    filepath = write_plan(plan_obj, output_dir)
    vr = validate(plan_obj)
    write_validation_report(vr, output_dir)

    print(f"Saved: {filepath}")
    if diags:
        for d in diags:
            if d:
                print(f"  [{d}]")
    if vr.status == "needs_revision":
        print(f"Validation issues ({len(vr.issues)}):")
        for issue in vr.issues:
            print(f"  - {issue}")


# ── CLI argument parsing ──────────────────────────────────────────


def _run_advise(args: argparse.Namespace) -> None:
    """Handle the 'expagent advise' subcommand."""
    import yaml as _yaml

    llm = resolve_llm_config(
        config_path=args.config,
        cli_model=args.model,
        cli_api_base=args.api_base,
        cli_api_key_env=args.api_key_env,
    )

    situation = _read_idea(args.context) if args.context else ""
    artifacts: list[ArtifactRef] = []
    if args.artifacts:
        for art_path in args.artifacts:
            p = Path(art_path).expanduser()
            if p.exists():
                summary = p.read_text(encoding="utf-8")[:500] if p.is_file() else str(p)
                artifacts.append(ArtifactRef(id=p.stem, type="other", path=str(p.resolve()), summary=summary))

    existing_plan = None
    if getattr(args, "existing_plan", None):
        plan_path = Path(args.existing_plan).expanduser()
        if plan_path.exists():
            data = _yaml.safe_load(plan_path.read_text(encoding="utf-8"))
            existing_plan = ExperimentPlan.model_validate(data)

    ctx = AdvisorContext(situation=situation, artifacts=artifacts,
                         existing_plan=existing_plan,
                         thread_dir=getattr(args, "thread_dir", "") or "")
    output_dir = args.output or _default_output_dir()
    mock = getattr(args, "mock_llm", False)

    print("ExpAgent analyzing...")
    decision, trace = advise(ctx, model=llm["model"], api_base=llm["api_base"],
                             api_key_env=llm["api_key_env"], mock=mock, run_dir=output_dir)

    plan_path = write_decision(decision, output_dir)
    vr = validate_decision(decision)
    write_validation_report(vr, output_dir)
    if decision.experiment_plan:
        write_plan(decision.experiment_plan, output_dir)

    print(f"Saved: {plan_path}")
    print(f"Confidence: {decision.confidence}")
    conclusion_status = decision.conclusion.status if decision.conclusion else "N/A"
    print(f"Conclusion: {conclusion_status}")
    print(f"Actions: {len(decision.recommended_actions)}")
    if vr.status == "needs_revision":
        print(f"Validation issues ({len(vr.issues)}):")
        for issue in vr.issues:
            print(f"  - {issue}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="expagent",
        description="ExpAgent — LLM-first scientific advisor for ML research",
    )

    sub = parser.add_subparsers(dest="command")

    # ---- advise subcommand ----
    advise_parser = sub.add_parser("advise", help="Run ExpAgent as scientific advisor")
    advise_parser.add_argument("--context", type=str, default=None, help="Situation description (file path or inline string)")
    advise_parser.add_argument("--artifacts", type=str, nargs="*", default=None, help="Artifact file paths")
    advise_parser.add_argument("--existing-plan", type=str, default=None, help="Path to experiment_plan.yaml")
    advise_parser.add_argument("--output", "-o", type=Path, default=None, help="Output directory")
    advise_parser.add_argument("--model", type=str, default=None)
    advise_parser.add_argument("--api-base", type=str, default=None)
    advise_parser.add_argument("--api-key-env", type=str, default=None)
    advise_parser.add_argument("--mock-llm", action="store_true", default=False)
    advise_parser.add_argument("--thread-dir", type=str, default=None,
                                help="Thread directory for continuous advisory sessions")
    advise_parser.add_argument("--config", "-c", type=str, default=None)

    # ---- Default / REPL mode (no subcommand) ----

    # Idea input
    parser.add_argument(
        "--idea", type=str, default=None,
        help="Research idea (file path or inline string)",
    )
    parser.add_argument(
        "--target-task", type=str, default=None,
        help="Task domain (e.g. 'image classification')",
    )

    # Output
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output directory for experiment_plan.yaml",
    )

    # Compute budget
    parser.add_argument("--gpu", type=str, default="RTX 4090")
    parser.add_argument("--max-runtime", type=str, default="2 hours")
    parser.add_argument("--max-trials", type=int, default=10)
    parser.add_argument("--constraint", type=str, action="append", default=None)

    # LLM config overrides
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--api-base", type=str, default=None)
    parser.add_argument("--api-key-env", type=str, default=None)
    parser.add_argument("--mock-llm", action="store_true", default=False)

    # Config
    parser.add_argument("--config", "-c", type=str, default=None,
                        help="Path to config.yaml")

    # Mode
    parser.add_argument("--no-interactive", action="store_true", default=False,
                        help="Generate plan and exit (no REPL)")

    return parser.parse_args(argv)


def _read_idea(spec: str) -> str:
    """Read idea from file or return directly if not a file path."""
    path = Path(spec).expanduser()
    if path.exists() and path.is_file():
        return path.read_text(encoding="utf-8").strip()
    return spec


def _default_output_dir() -> Path:
    """Return the default output directory.

    Priority: RESAGENT_WORKSPACE env > project runs/ > cwd runs/.
    """
    import os as _os
    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    root = _os.environ.get("RESAGENT_WORKSPACE")
    if root:
        return Path(root) / stamp
    project_root = Path(__file__).resolve().parent.parent.parent
    if project_root.joinpath("pyproject.toml").exists():
        return project_root / "runs" / stamp
    return Path.cwd() / "runs" / stamp


if __name__ == "__main__":
    main()
