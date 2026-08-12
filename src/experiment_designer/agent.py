"""Top-level public run API for ExpAgent.

advise() is the entry point into the scientific advisor loop. It resolves the
run directory and context policy, then delegates to the loop in controller/loop.py.
"""

from __future__ import annotations

from pathlib import Path

from .context.policy import ContextPolicy
from .controller.loop import _run_loop
from .models import AdvisorContext, ScientificDecision


def advise(
    ctx: AdvisorContext,
    *,
    model: str = "deepseek-chat",
    api_base: str = "https://api.deepseek.com/v1",
    api_key_env: str = "DEEPSEEK_API_KEY",
    mock: bool = False,
    trace_dir: Path | None = None,
    run_dir: Path | None = None,
    max_steps: int | None = None,
    enable_paper_search: bool = True,
) -> tuple[ScientificDecision, list[dict]]:
    """Run the ExpAgent agentic loop and return a ScientificDecision.

    Args:
        run_dir: ExpAgent-owned directory. Caller must pass a unique path
                 per invocation. ExpAgent may create any files/subdirs here.
                 Callers should treat everything except state.json as
                 implementation detail.
        max_steps: Override MAX_STEPS (default 20). For advisory/QA, use 8.
        enable_paper_search: If False, remove search_papers/save_paper from tools.
    """
    if run_dir is None:
        import os as _os
        from datetime import datetime, timezone
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        root = _os.environ.get("RESAGENT_WORKSPACE")
        run_dir = (Path(root) if root else Path.cwd() / "runs") / stamp
    if trace_dir is None:
        trace_dir = run_dir / "logs"
    policy = ContextPolicy.for_model(model)
    return _run_loop(ctx, model, api_base, api_key_env, mock, trace_dir, policy, run_dir,
                     max_steps, enable_paper_search)
