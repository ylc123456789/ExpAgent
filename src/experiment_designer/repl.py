"""Interactive REPL for ExpAgent.

Session state and slash-command / natural-language handling live here so that
main.py stays limited to argument parsing and dependency assembly.
"""

from __future__ import annotations

import readline  # noqa: F401 — enables line editing on input()
import sys
from pathlib import Path

from .controller.planner import plan, revise
from .controller.validator import validate
from .models import ComputeBudget, DesignInput, ExperimentPlan
from .presentation import (
    print_config,
    print_dim,
    print_error,
    print_help,
    print_info,
    print_plan_full,
    print_plan_preview,
    print_warn,
    print_welcome,
)
from .report import write_plan, write_validation_report


def run_repl(
    *,
    initial_input: DesignInput | None,
    llm: dict[str, str],
    mock: bool,
    config_path: str | None,
    default_output_dir,
) -> None:
    """Run the interactive REPL loop."""
    state: dict = {
        "current_plan": None,  # ExperimentPlan | None
        "current_idea": initial_input.research_idea if initial_input else None,
        "plan_path": None,  # Path | None (where it was saved)
    }

    print_welcome()

    # If launched with --idea, auto-generate first plan
    if initial_input:
        print_info("Generating initial experiment plan...")
        try:
            plan_obj, diags = _generate_plan(initial_input, llm, mock)
            state["current_plan"] = plan_obj
            state["current_idea"] = initial_input.research_idea
            print_plan_preview(plan_obj)
            for d in diags:
                if d:
                    print_dim(f"  [{d}]")
        except Exception as e:
            print_error(f"Failed to generate plan: {e}")
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
            _handle_command(user_input, state, llm, mock, config_path, default_output_dir)
            continue

        # Natural language input
        _handle_natural_input(user_input, state, llm, mock)


def _handle_command(
    cmd: str,
    state: dict,
    llm: dict[str, str],
    mock: bool,
    config_path: str | None,
    default_output_dir,
) -> None:
    """Process a /slash command."""
    parts = cmd.split(maxsplit=1)
    name = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if name in ("/quit", "/exit", "/q"):
        if state["current_plan"] and not state["plan_path"]:
            print_warn("You have unsaved changes. Use /save first or /quit again to discard.")
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
        print_help()

    elif name == "/view":
        if state["current_plan"] is None:
            print_warn("No plan yet. Describe your research idea first.")
            return
        section = arg.strip().lower() if arg else ""
        print_plan_full(state["current_plan"], section)

    elif name == "/save":
        if state["current_plan"] is None:
            print_warn("No plan to save.")
            return
        output_dir = Path(arg) if arg else default_output_dir()
        try:
            filepath = write_plan(state["current_plan"], output_dir)
            state["plan_path"] = filepath
            vr = validate(state["current_plan"])
            write_validation_report(vr, output_dir)
            print_info(f"Saved to {filepath}")
            if vr.status == "needs_revision":
                print_warn(f"Validation issues ({len(vr.issues)}):")
                for issue in vr.issues:
                    print_dim(f"  - {issue}")
        except OSError as e:
            print_error(f"Failed to save: {e}")

    elif name == "/load":
        if not arg:
            print_warn("Usage: /load <path/to/experiment_plan.yaml>")
            return
        try:
            import yaml
            path = Path(arg).expanduser().resolve()
            if not path.exists():
                print_error(f"File not found: {path}")
                return
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            plan_obj = ExperimentPlan.model_validate(data)
            state["current_plan"] = plan_obj
            state["plan_path"] = path
            print_info(f"Loaded plan from {path}")
            print_plan_preview(plan_obj)
        except Exception as e:
            print_error(f"Failed to load: {e}")

    elif name == "/idea":
        if state["current_idea"]:
            print_info("Current research idea:")
            print(f"  {state['current_idea']}")
        else:
            print_warn("No research idea stored.")

    elif name == "/new":
        state["current_plan"] = None
        state["current_idea"] = None
        state["plan_path"] = None
        print_info("Cleared current plan. Describe your new research idea.")

    elif name == "/config":
        print_config(llm, mock)

    else:
        print_warn(f"Unknown command: {name}. Type /help for available commands.")


def _handle_natural_input(
    user_input: str,
    state: dict,
    llm: dict[str, str],
    mock: bool,
) -> None:
    """Process natural language input."""
    if state["current_plan"] is not None:
        # Revision mode
        print_info("Revising plan based on your feedback...")
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
            print_plan_preview(plan_obj)
            for d in diags:
                if d:
                    print_dim(f"  [{d}]")
        except Exception as e:
            print_error(f"Failed to revise plan: {e}")
    else:
        # New plan mode
        inp = DesignInput(
            research_idea=user_input,
            target_task="unspecified",  # LLM will infer from the idea
            compute_budget=ComputeBudget(),
        )
        state["current_idea"] = user_input
        print_info("Generating experiment plan...")
        try:
            plan_obj, diags = _generate_plan(inp, llm, mock)
            state["current_plan"] = plan_obj
            print_plan_preview(plan_obj)
            for d in diags:
                if d:
                    print_dim(f"  [{d}]")
        except Exception as e:
            print_error(f"Failed to generate plan: {e}")


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
