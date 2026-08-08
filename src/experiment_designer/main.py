"""CLI entry point — REPL, direct, and advise modes."""

from __future__ import annotations

import argparse
import readline
import sys
from pathlib import Path

from .advisor import advise
from .config import resolve_llm_config
from .models import (
    AdvisorContext,
    ArtifactRef,
    ComputeBudget,
    DesignInput,
    ExistingAssets,
    ExperimentPlan,
)
from .planner import plan, revise
from .report import write_decision, write_plan, write_validation_report
from .validator import validate, validate_decision


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
        _run_repl(
            initial_input=initial_input,
            llm=llm,
            mock=args.mock_llm,
            config_path=args.config,
        )


# ── REPL ──────────────────────────────────────────────────────────


def _run_repl(
    *,
    initial_input: DesignInput | None,
    llm: dict[str, str],
    mock: bool,
    config_path: str | None,
) -> None:
    """Run the interactive REPL loop."""
    state: dict = {
        "current_plan": None,  # ExperimentPlan | None
        "current_idea": initial_input.research_idea if initial_input else None,
        "plan_path": None,  # Path | None (where it was saved)
    }

    _print_welcome()

    # If launched with --idea, auto-generate first plan
    if initial_input:
        _print_info("Generating initial experiment plan...")
        try:
            plan_obj, diags = _generate_plan(initial_input, llm, mock)
            state["current_plan"] = plan_obj
            state["current_idea"] = initial_input.research_idea
            _print_plan_preview(plan_obj)
            for d in diags:
                if d:
                    _print_dim(f"  [{d}]")
        except Exception as e:
            _print_error(f"Failed to generate plan: {e}")
    else:
        print()
        print("  Describe your research idea to get started.")
        print()

    # REPL loop
    while True:
        try:
            user_input = input("▸ ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue

        # Slash commands
        if user_input.startswith("/"):
            _handle_command(user_input, state, llm, mock, config_path)
            continue

        # Natural language input
        _handle_natural_input(user_input, state, llm, mock)


def _handle_command(
    cmd: str,
    state: dict,
    llm: dict[str, str],
    mock: bool,
    config_path: str | None,
) -> None:
    """Process a /slash command."""
    parts = cmd.split(maxsplit=1)
    name = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if name in ("/quit", "/exit", "/q"):
        if state["current_plan"] and not state["plan_path"]:
            _print_warn("You have unsaved changes. Use /save first or /quit again to discard.")
            # Check if they really want to quit
            try:
                again = input("▸ ").strip()
                if again.lower() in ("/quit", "/exit", "/q"):
                    pass
                else:
                    return
            except (EOFError, KeyboardInterrupt):
                pass
        print("  Goodbye!")
        sys.exit(0)

    elif name == "/help":
        _print_help()

    elif name == "/view":
        if state["current_plan"] is None:
            _print_warn("No plan yet. Describe your research idea first.")
            return
        section = arg.strip().lower() if arg else ""
        _print_plan_full(state["current_plan"], section)

    elif name == "/save":
        if state["current_plan"] is None:
            _print_warn("No plan to save.")
            return
        output_dir = Path(arg) if arg else _default_output_dir()
        try:
            filepath = write_plan(state["current_plan"], output_dir)
            state["plan_path"] = filepath
            vr = validate(state["current_plan"])
            write_validation_report(vr, output_dir)
            _print_info(f"Saved to {filepath}")
            if vr.status == "needs_revision":
                _print_warn(f"Validation issues ({len(vr.issues)}):")
                for issue in vr.issues:
                    _print_dim(f"  - {issue}")
        except OSError as e:
            _print_error(f"Failed to save: {e}")

    elif name == "/load":
        if not arg:
            _print_warn("Usage: /load <path/to/experiment_plan.yaml>")
            return
        try:
            import yaml
            path = Path(arg).expanduser().resolve()
            if not path.exists():
                _print_error(f"File not found: {path}")
                return
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            plan_obj = ExperimentPlan.model_validate(data)
            state["current_plan"] = plan_obj
            state["plan_path"] = path
            _print_info(f"Loaded plan from {path}")
            _print_plan_preview(plan_obj)
        except Exception as e:
            _print_error(f"Failed to load: {e}")

    elif name == "/idea":
        if state["current_idea"]:
            _print_info("Current research idea:")
            print(f"  {state['current_idea']}")
        else:
            _print_warn("No research idea stored.")

    elif name == "/new":
        state["current_plan"] = None
        state["current_idea"] = None
        state["plan_path"] = None
        _print_info("Cleared current plan. Describe your new research idea.")

    elif name == "/config":
        _print_config(llm, mock)

    else:
        _print_warn(f"Unknown command: {name}. Type /help for available commands.")


def _handle_natural_input(
    user_input: str,
    state: dict,
    llm: dict[str, str],
    mock: bool,
) -> None:
    """Process natural language input."""
    if state["current_plan"] is not None:
        # Revision mode
        _print_info("Revising plan based on your feedback...")
        try:
            plan_obj, diags = revise(
                state["current_plan"],
                user_input,
                model=llm["model"],
                api_base=llm["api_base"],
                api_key_env=llm["api_key_env"],
                mock=mock,
            )
            state["current_plan"] = plan_obj
            state["plan_path"] = None  # Modified, not saved
            _print_plan_preview(plan_obj)
            for d in diags:
                if d:
                    _print_dim(f"  [{d}]")
        except Exception as e:
            _print_error(f"Failed to revise plan: {e}")
    else:
        # New plan mode
        inp = DesignInput(
            research_idea=user_input,
            target_task="unspecified",  # LLM will infer from the idea
            compute_budget=ComputeBudget(),
        )
        state["current_idea"] = user_input
        _print_info("Generating experiment plan...")
        try:
            plan_obj, diags = _generate_plan(inp, llm, mock)
            state["current_plan"] = plan_obj
            _print_plan_preview(plan_obj)
            for d in diags:
                if d:
                    _print_dim(f"  [{d}]")
        except Exception as e:
            _print_error(f"Failed to generate plan: {e}")


# ── Direct mode ───────────────────────────────────────────────────


def _run_direct(
    inp: DesignInput,
    output_dir: Path,
    llm: dict[str, str],
    mock: bool,
) -> None:
    """Non-interactive: generate plan and save to disk."""
    print("Generating experiment plan...")
    plan_obj, diags = _generate_plan(inp, llm, mock)

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


# ── Shared helpers ────────────────────────────────────────────────


def _generate_plan(
    inp: DesignInput,
    llm: dict[str, str],
    mock: bool,
) -> tuple[ExperimentPlan, list[str]]:
    """Generate a plan from input. Thin wrapper around planner.plan()."""
    return plan(
        inp,
        model=llm["model"],
        api_base=llm["api_base"],
        api_key_env=llm["api_key_env"],
        mock=mock,
    )


# ── Display helpers ───────────────────────────────────────────────


def _print_welcome() -> None:
    """Print the REPL welcome banner."""
    print()
    print("  ╔══════════════════════════════════════════╗")
    print("  ║  ExpAgent — Scientific Advisor  ║")
    print("  ║  Type /help for commands                 ║")
    print("  ╚══════════════════════════════════════════╝")
    print()


def _print_help() -> None:
    """Print help text."""
    print()
    print("  Commands:")
    print("  /help              Show this message")
    print("  /view [section]    View current plan (section: goal|matrix|tasks|risks|analysis)")
    print("  /save [path]       Save plan to directory (default: runs/<timestamp>/)")
    print("  /load <path>       Load a saved plan")
    print("  /idea              Show the original research idea")
    print("  /new               Clear current plan and start over")
    print("  /config            Show current configuration")
    print("  /quit, /exit       Exit")
    print()
    print("  Just type naturally to:")
    print("  - Describe a new research idea (when no plan exists)")
    print("  - Request changes to the current plan (when a plan exists)")
    print()


def _print_plan_preview(plan: ExperimentPlan) -> None:
    """Print a compact preview of the experiment plan."""
    n_baselines = sum(1 for m in plan.experiment_matrix.methods if m.type == "baseline")
    n_ablations = sum(1 for m in plan.experiment_matrix.methods if m.type == "ablation")
    n_coding = len(plan.tasks.coding_tasks)
    n_repro = len(plan.tasks.repro_tasks)
    n_run = len(plan.tasks.run_tasks)
    n_risks = len(plan.risks)

    vr = validate(plan)

    print()
    _print_kv("Hypothesis", plan.goal.hypothesis[:120] + ("..." if len(plan.goal.hypothesis) > 120 else ""))
    _print_kv("Dataset", ", ".join(d.name for d in plan.experiment_matrix.datasets))
    _print_kv("Methods", f"{len(plan.experiment_matrix.methods)} total ({n_baselines} baselines, {n_ablations} ablations)")
    _print_kv("Metrics", ", ".join(m.name for m in plan.experiment_matrix.metrics))
    _print_kv("Tasks", f"{n_coding} coding, {n_repro} repro, {n_run} run")
    _print_kv("Risks", str(n_risks))
    if vr.status == "needs_revision":
        _print_warn(f"Validation: {len(vr.issues)} issue(s)")
        for issue in vr.issues[:3]:
            _print_dim(f"  - {issue}")
        if len(vr.issues) > 3:
            _print_dim(f"  ... and {len(vr.issues) - 3} more")
    else:
        _print_info("Validation: passed")
    print()
    print("  Type /view to see the full plan, or describe changes.")
    print()


def _print_plan_full(plan: ExperimentPlan, section: str = "") -> None:
    """Print the full experiment plan as YAML."""
    import yaml

    data = plan.model_dump(exclude_defaults=False)

    if section and section != "all":
        section_map = {
            "goal": ["goal"],
            "matrix": ["experiment_matrix"],
            "tasks": ["tasks"],
            "risks": ["risks"],
            "analysis": ["analysis_plan"],
        }
        keys = section_map.get(section, [section])
        data = {k: v for k, v in data.items() if k in keys}
        if not data:
            _print_warn(f"No section matching '{section}'. Available: goal, matrix, tasks, risks, analysis")
            return

    lines = yaml.dump(data, allow_unicode=True, sort_keys=False).strip()
    line_count = lines.count("\n") + 1

    print()
    print("  " + "─" * 60)
    print(lines)
    print("  " + "─" * 60)
    print(f"  {line_count} lines — /view <section> for specific parts")
    print()


def _print_config(
    llm: dict[str, str],
    mock: bool,
) -> None:
    """Print current configuration."""
    print()
    _print_kv("Model", llm["model"])
    _print_kv("API Base", llm["api_base"])
    _print_kv("API Key Env", llm["api_key_env"])
    _print_kv("Mock Mode", "ON" if mock else "OFF")
    print()


def _print_kv(key: str, value: str) -> None:
    """Print a key-value pair aligned."""
    print(f"  {key:<14} {value}")


def _print_info(msg: str) -> None:
    """Print an informational message."""
    print(f"\n  \033[34mⓘ\033[0m {msg}")


def _print_warn(msg: str) -> None:
    """Print a warning message."""
    print(f"\n  \033[33m⚠\033[0m {msg}")


def _print_error(msg: str) -> None:
    """Print an error message."""
    print(f"\n  \033[31m✗\033[0m {msg}")


def _print_dim(msg: str) -> None:
    """Print a dimmed message."""
    print(f"\033[90m{msg}\033[0m")


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

    ctx = AdvisorContext(situation=situation, artifacts=artifacts, existing_plan=existing_plan)
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
    """Return the default output directory (project-local runs/<timestamp>)."""
    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    # Find the project root (where pyproject.toml lives)
    project_root = Path(__file__).resolve().parent.parent.parent
    # If installed in editable mode, __file__ points to src/experiment_designer/
    # so parent.parent.parent is the project root.
    # As a fallback, use cwd
    candidate = project_root / "runs" / stamp
    if not project_root.joinpath("pyproject.toml").exists():
        candidate = Path.cwd() / "runs" / stamp
    return candidate


if __name__ == "__main__":
    main()
