"""Agentic loop for ExpAgent v2 — the scientific advisor.

Uses Function Calling (OpenAI-compatible) for reliable tool use.
Features: file_cache memory, grace stop, adaptive context.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from .llm import call_llm
from .models import AdvisorContext, ContextPolicy, ScientificDecision
from .prompts import SYSTEM_PROMPT, build_initial_prompt
from .tools import read_file, search_papers
from .validator import validate_decision

MAX_STEPS = 20
MAX_EXTRA_AFTER_PROGRESS = 6


def advise(
    ctx: AdvisorContext,
    *,
    model: str = "deepseek-chat",
    api_base: str = "https://api.deepseek.com/v1",
    api_key_env: str = "DEEPSEEK_API_KEY",
    mock: bool = False,
    trace_dir: Path | None = None,
) -> tuple[ScientificDecision, list[dict]]:
    """Run the ExpAgent agentic loop and return a ScientificDecision.

    Uses function calling for reliable tool use with file_cache memory,
    grace stop, and adaptive context.
    """
    policy = ContextPolicy.for_model(model)
    return _run_loop(ctx, model, api_base, api_key_env, mock, trace_dir, policy)


# ── Agentic loop ───────────────────────────────────────────────────


def _run_loop(ctx, model, api_base, api_key_env, mock, trace_dir, policy):
    """Function-calling agentic loop with memory and grace stop."""
    initial = build_initial_prompt(ctx)
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": initial},
    ]

    file_cache: dict[str, str] = {}
    trace: list[dict] = []
    effective_max = MAX_STEPS

    for step in range(1, MAX_STEPS + MAX_EXTRA_AFTER_PROGRESS + 1):
        remaining = effective_max - step if step <= effective_max else 0

        # Grace stop pressure
        if step >= effective_max - 2 or (step >= MAX_STEPS - 2 and effective_max == MAX_STEPS):
            pressure = (
                "\n\n[GRACE STOP: You have very few steps remaining. "
                "You MUST call finish now with your best scientific judgment, "
                "even if incomplete. Prioritize conclusion + recommended_actions.]"
            )
            messages.append({"role": "user", "content": pressure})

        try:
            result = call_llm(
                model=model, api_base=api_base, api_key_env=api_key_env,
                system=SYSTEM_PROMPT, messages=messages,
                mock=mock, trace_dir=trace_dir, trace_label=f"step{step}",
            )
        except Exception as e:
            trace.append({"action": "api_error", "summary": str(e)[:100]})
            messages.append({"role": "user", "content": f"[API call failed: {e}. Continue with what you know.]"})
            continue

        if result["type"] == "tool_call":
            name = result["name"]
            args = result["arguments"]

            if name == "search_papers":
                output = _execute_search(args)
                key = f"search_step{step}"
                file_cache[key] = output
                trace.append({"action": "search_papers", "summary": f"q={args.get('query','')[:60]}, {_count_results(output)}"})
                if effective_max == MAX_STEPS:
                    effective_max += MAX_EXTRA_AFTER_PROGRESS

            elif name == "read_file":
                path = args.get("path", "")
                output = read_file(path) if path else "No path provided."
                file_cache[path or f"file_step{step}"] = output
                trace.append({"action": "read_file", "summary": f"path={path[:80]}"})
                if effective_max == MAX_STEPS:
                    effective_max += MAX_EXTRA_AFTER_PROGRESS

            elif name == "finish":
                yaml_text = args.get("decision_yaml", "")
                try:
                    decision = _parse_decision(yaml_text)
                except Exception as e:
                    output = f"Parse error: {e}. Please output valid YAML."
                    trace.append({"action": "finish", "summary": f"parse error: {str(e)[:80]}"})
                    messages.append(_make_assistant_msg(name, args, step))
                    messages.append({"role": "tool", "tool_call_id": f"call_{step}", "content": output})
                    continue

                vr = validate_decision(decision)
                trace.append({"action": "finish", "summary": decision.summary[:100]})
                if vr.status == "ok":
                    return decision, trace
                issues_text = "\n".join(f"- {i}" for i in vr.issues)
                output = f"Validation issues:\n{issues_text}\nFix and call finish again."
                trace.append({"action": "validate", "summary": f"{len(vr.issues)} issues"})
            else:
                output = f"Unknown tool: {name}"
                trace.append({"action": "unknown", "summary": name})

            messages.append(_make_assistant_msg(name, args, step))
            messages.append({"role": "tool", "tool_call_id": f"call_{step}", "content": output[:policy.search_results_chars]})
            messages = _trim_messages(messages, policy)
            _inject_file_cache(messages, file_cache, policy)

        elif result["type"] == "message":
            content = result.get("content", "")
            trace.append({"action": "message", "summary": content[:100]})
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": "Use a tool: search_papers, read_file, or finish."})

        else:
            trace.append({"action": "error", "summary": str(result.get("error", ""))[:100]})
            break

    return None, trace


def _make_assistant_msg(name: str, args: dict, step: int) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": f"call_{step}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
        }],
    }


# ── Tool execution ─────────────────────────────────────────────────


def _execute_search(args: dict) -> str:
    try:
        results = search_papers(
            query=args.get("query", ""),
            source=args.get("source", "semantic_scholar"),
            max_results=min(args.get("max_results", 5), 10),
            year_from=args.get("year_from"),
            venue_filter=args.get("venue_filter") or None,
        )
    except Exception as e:
        return f"Search error: {e}"
    if not results:
        return "No results found."
    lines = [f"Found {len(results)} papers:"]
    for i, r in enumerate(results, 1):
        authors = ", ".join(r.authors[:3])
        if len(r.authors) > 3:
            authors += " et al."
        v = f" ({r.venue})" if r.venue else ""
        y = f" [{r.year}]" if r.year else ""
        lines.append(f"\n{i}. {r.title}{v}{y}")
        lines.append(f"   Authors: {authors}")
        if r.abstract:
            lines.append(f"   Abstract: {r.abstract[:300]}")
        if r.url:
            lines.append(f"   URL: {r.url}")
        if r.paper_id:
            lines.append(f"   ID: {r.paper_id}")
    return "\n".join(lines)


def _count_results(output: str) -> str:
    m = re.search(r"Found (\d+) papers?", output)
    return f"{m.group(1)}p" if m else "?"


def _parse_decision(yaml_text: str) -> ScientificDecision:
    if not yaml_text:
        raise ValueError("Empty decision YAML")
    yaml_text = yaml_text.strip()
    for fence in ("```yaml", "```"):
        if yaml_text.startswith(fence):
            yaml_text = yaml_text[len(fence):].strip()
    if yaml_text.endswith("```"):
        yaml_text = yaml_text[:-3].strip()
    try:
        data = yaml.safe_load(yaml_text)
    except yaml.YAMLError:
        try:
            data = json.loads(yaml_text)
        except json.JSONDecodeError:
            raise ValueError(f"Could not parse: {yaml_text[:200]}")
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict, got {type(data).__name__}")
    return ScientificDecision.model_validate(data)


# ── Context management ─────────────────────────────────────────────


def _trim_messages(messages: list[dict], policy: ContextPolicy) -> list[dict]:
    if len(messages) <= 4:
        return messages
    header = messages[:2]
    body = messages[2:]
    max_pairs = policy.step_history
    if len(body) > max_pairs * 2:
        body = body[-(max_pairs * 2):]
    return header + body


def _inject_file_cache(messages: list[dict], cache: dict[str, str], policy: ContextPolicy) -> None:
    if not cache:
        return
    entries = list(cache.items())[-policy.file_cache_count:]
    lines = ["\n[File cache — reference these search results:]"]
    for key, value in entries:
        lines.append(f"\n--- {key} ---\n{value[:policy.file_cache_chars]}")
    msg = "\n".join(lines)
    if messages and messages[-1]["role"] == "user":
        messages[-1]["content"] += msg
    else:
        messages.append({"role": "user", "content": msg})


