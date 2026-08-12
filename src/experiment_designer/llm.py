"""LLM client — OpenAI-compatible API via urllib.

Matches ReproAgent's llm.py and CodingAgent's llm.py style:
- No openai package, no chat history.
- API layer: 3 attempts (2 retries) on transient errors (network, 5xx).
- Loop layer: controller/loop.py catches failures and injects as api_error steps.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .prompts import TOOLS


def call_llm(
    *,
    model: str,
    api_base: str,
    api_key_env: str,
    system: str,
    messages: list[dict],
    mock: bool = False,
    trace_dir: Path | None = None,
    trace_label: str = "llm",
    tools: list[dict] | None = None,
) -> dict:
    """Call LLM with tool definitions, returning parsed tool calls or message.

    Args:
        messages: List of message dicts [{"role": "system", "content": ...}, ...].
                  The system message should be included as the first message.
        Others: Same as call_llm.

    Returns:
        dict with keys:
        - "type": "tool_calls" | "message" | "error"
        - "calls": list of {"name": str, "arguments": dict} (if tool_calls)
        - "content": str (if message)
        - "error": str (if error)
    """
    if mock:
        result = _mock_tool_response(messages)
        if trace_dir:
            _write_trace(trace_dir, trace_label, "",
                         json.dumps(messages[-1], ensure_ascii=False) if messages else "",
                         json.dumps(result, ensure_ascii=False))
        return result

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} is not set.")

    _tools = tools if tools is not None else TOOLS
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "tools": _tools,
        "tool_choice": "auto",
        "temperature": 0.3,
    }

    url = _chat_completions_url(api_base)
    last_error: Exception | None = None

    for attempt in range(3):
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(body).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(min(2 ** attempt * 2, 30))
            continue
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code >= 500 and attempt < 2:
                time.sleep(min(2 ** attempt * 2, 30))
                continue
            body_text = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"LLM request failed: {exc.code} {body_text}") from exc

        choice = data["choices"][0]
        msg = choice.get("message", {})

        if trace_dir:
            _write_trace(trace_dir, trace_label, "", json.dumps(messages, ensure_ascii=False),
                         json.dumps(msg, ensure_ascii=False))

        # Check for tool calls — DeepSeek may return multiple parallel calls
        tool_calls = msg.get("tool_calls", [])
        if tool_calls:
            calls: list[dict] = []
            for tc in tool_calls:
                func = tc.get("function", {})
                try:
                    arguments = json.loads(func.get("arguments", "{}"))
                except json.JSONDecodeError:
                    arguments = {}
                calls.append({
                    "name": func.get("name", "unknown"),
                    "arguments": arguments,
                })
            return {"type": "tool_calls", "calls": calls}

        return {
            "type": "message",
            "content": msg.get("content", ""),
        }

    raise RuntimeError(f"LLM API call failed after 3 retries: {last_error}")


def _chat_completions_url(api_base: str) -> str:
    """Normalize API base to the chat completions endpoint."""
    base = api_base.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


def _write_trace(
    trace_dir: Path,
    label: str,
    system: str,
    user: str,
    response: str,
) -> None:
    """Write prompt/response trace files for debugging."""
    trace_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_") or "llm"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    prefix = trace_dir / f"llm_{stamp}_{safe}"
    (prefix.with_suffix(".prompt.txt")).write_text(
        f"[system]\n{system}\n\n[user]\n{user}", encoding="utf-8"
    )
    (prefix.with_suffix(".response.txt")).write_text(response, encoding="utf-8")


# ── Mock responses ─────────────────────────────────────────────────


_MOCK_TOOL_STEP = 0


def _mock_tool_response(messages: list[dict]) -> dict:
    """Deterministic mock for function calling. First search, then finish."""
    global _MOCK_TOOL_STEP
    _MOCK_TOOL_STEP += 1
    last_msg = messages[-1].get("content", "") if messages else ""

    if _MOCK_TOOL_STEP <= 1 and "finish" not in last_msg.lower():
        return {
            "type": "tool_calls",
            "calls": [{
                "name": "search_papers",
                "arguments": {"query": "channel attention image classification benchmark", "max_results": 3},
            }],
        }
    # Finish with structured args
    decision = _make_mock_design_decision()
    return {
        "type": "tool_calls",
        "calls": [{
            "name": "finish",
            "arguments": {
                "summary": decision["summary"],
                "confidence": decision["confidence"],
                "conclusion_status": decision["conclusion"]["status"],
                "conclusion_rationale": decision["conclusion"]["rationale"],
                "evidence": decision["evidence"],
                "recommended_actions": decision["recommended_actions"],
                "experiment_plan": decision.get("experiment_plan"),
                "risks": decision["risks"],
                "needs_user_input": decision.get("needs_user_input", []),
            },
        }],
    }


def _make_mock_design_decision() -> dict:
    """Mock decision dict for testing."""
    return {
        "summary": "Verify channel attention parameter efficiency on CIFAR-10",
        "confidence": "medium",
        "conclusion": {"status": "needs_more_experiments", "rationale": "L2-norm attention is scientifically plausible."},
        "evidence": [
            {"source": "literature", "description": "SE-Net established channel attention for CNNs"},
        ],
        "experiment_plan": {
            "version": 1,
            "goal": {"summary": "Validate L2-norm attention", "hypothesis": "L2-norm attn >= 2% improvement", "success_criteria": ["top-1 accuracy >= baseline + 2%"]},
            "experiment_matrix": {
                "datasets": [{"name": "CIFAR-10", "split": "standard", "rationale": "Standard benchmark"}],
                "methods": [
                    {"name": "proposed_l2_attention", "type": "new_method", "implementation_status": "needs_code", "rationale": "Core method"},
                    {"name": "resnet18_baseline", "type": "baseline", "implementation_status": "needs_repro", "rationale": "Standard baseline"},
                ],
                "metrics": [{"name": "top1_accuracy", "rationale": "Primary metric"}],
            },
            "tasks": {
                "coding_tasks": [{"id": "code_001", "workspace_path": "./", "task_goal": "Implement L2-norm attention", "rationale": "Core method"}],
                "repro_tasks": [],
                "run_tasks": [{"id": "run_001", "command_goal": "Run CIFAR-10 bounded experiment", "rationale": "Core comparison"}],
            },
            "analysis_plan": {"comparisons": ["proposed vs resnet18"], "plots": [], "failure_checks": []},
            "risks": [{"description": "May not generalize", "mitigation": "Follow-up on ImageNet"}],
        },
        "recommended_actions": [
            {"priority": "high", "type": "coding_task",
             "rationale": "Implement proposed method",
             "action_id": "patch_training_loop",
             "depends_on": [],
             "project_ref": "current_project",
             "plan": {"kind": "coding_task", "workspace_path": "./", "task_goal": "Implement L2-norm attention"}},
            {"priority": "high", "type": "run_task",
             "rationale": "Validate after code patch",
             "action_id": "run_with_patch",
             "depends_on": ["patch_training_loop"],
             "project_ref": "current_project",
             "plan": {"kind": "run_task", "command_goal": "Run CIFAR-10 bounded experiment", "requires_gpu": True}},
        ],
        "risks": ["CIFAR-10 may not generalize"],
        "needs_user_input": [],
    }


