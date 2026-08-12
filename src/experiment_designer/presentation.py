"""Terminal output rendering for the ExpAgent CLI/REPL.

Kept separate from the interaction loop (repl.py) so the display primitives
can be reused and tested independently.
"""

from __future__ import annotations

from .controller.validator import validate
from .models import ExperimentPlan


def print_welcome() -> None:
    """Print the REPL welcome banner."""
    print()
    print("  ╔══════════════════════════════════════════╗")
    print("  ║  ExpAgent — Scientific Advisor  ║")
    print("  ║  Type /help for commands                 ║")
    print("  ╚══════════════════════════════════════════╝")
    print()


def print_help() -> None:
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


def print_plan_preview(plan: ExperimentPlan) -> None:
    """Print a compact preview of the experiment plan."""
    n_baselines = sum(1 for m in plan.experiment_matrix.methods if m.type == "baseline")
    n_ablations = sum(1 for m in plan.experiment_matrix.methods if m.type == "ablation")
    n_coding = len(plan.tasks.coding_tasks)
    n_repro = len(plan.tasks.repro_tasks)
    n_run = len(plan.tasks.run_tasks)
    n_risks = len(plan.risks)

    vr = validate(plan)

    print()
    print_kv("Hypothesis", plan.goal.hypothesis[:120] + ("..." if len(plan.goal.hypothesis) > 120 else ""))
    print_kv("Dataset", ", ".join(d.name for d in plan.experiment_matrix.datasets))
    print_kv("Methods", f"{len(plan.experiment_matrix.methods)} total ({n_baselines} baselines, {n_ablations} ablations)")
    print_kv("Metrics", ", ".join(m.name for m in plan.experiment_matrix.metrics))
    print_kv("Tasks", f"{n_coding} coding, {n_repro} repro, {n_run} run")
    print_kv("Risks", str(n_risks))
    if vr.status == "needs_revision":
        print_warn(f"Validation: {len(vr.issues)} issue(s)")
        for issue in vr.issues[:3]:
            print_dim(f"  - {issue}")
        if len(vr.issues) > 3:
            print_dim(f"  ... and {len(vr.issues) - 3} more")
    else:
        print_info("Validation: passed")
    print()
    print("  Type /view to see the full plan, or describe changes.")
    print()


def print_plan_full(plan: ExperimentPlan, section: str = "") -> None:
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
            print_warn(f"No section matching '{section}'. Available: goal, matrix, tasks, risks, analysis")
            return

    lines = yaml.dump(data, allow_unicode=True, sort_keys=False).strip()
    line_count = lines.count("\n") + 1

    print()
    print("  " + "─" * 60)
    print(lines)
    print("  " + "─" * 60)
    print(f"  {line_count} lines — /view <section> for specific parts")
    print()


def print_config(
    llm: dict[str, str],
    mock: bool,
) -> None:
    """Print current configuration."""
    print()
    print_kv("Model", llm["model"])
    print_kv("API Base", llm["api_base"])
    print_kv("API Key Env", llm["api_key_env"])
    print_kv("Mock Mode", "ON" if mock else "OFF")
    print()


def print_kv(key: str, value: str) -> None:
    """Print a key-value pair aligned."""
    print(f"  {key:<14} {value}")


def print_info(msg: str) -> None:
    """Print an informational message."""
    print(f"\n  \033[34mⓘ\033[0m {msg}")


def print_warn(msg: str) -> None:
    """Print a warning message."""
    print(f"\n  \033[33m⚠\033[0m {msg}")


def print_error(msg: str) -> None:
    """Print an error message."""
    print(f"\n  \033[31m✗\033[0m {msg}")


def print_dim(msg: str) -> None:
    """Print a dimmed message."""
    print(f"\033[90m{msg}\033[0m")
