"""Agentic loop for ExpAgent v2 — the scientific advisor.

Uses Function Calling for reliable tool use.
Context: CodingAgent-style "rebuild prompt from state each turn" +
FC-compatible tool_pairs (keeps only recent 4 pairs, old info compressed).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from .llm import call_llm
from .models import AdvisorContext, ContextPolicy, ScientificDecision
from .prompts import SYSTEM_PROMPT, build_initial_prompt
from .tools import read_file, save_paper, search_papers
from .validator import validate_decision

MAX_STEPS = 20
MAX_EXTRA_AFTER_PROGRESS = 6
MAX_TOOL_PAIRS = 4  # keep recent 4 pairs for FC continuity, compress older ones


def advise(
    ctx: AdvisorContext,
    *,
    model: str = "deepseek-chat",
    api_base: str = "https://api.deepseek.com/v1",
    api_key_env: str = "DEEPSEEK_API_KEY",
    mock: bool = False,
    trace_dir: Path | None = None,
) -> tuple[ScientificDecision, list[dict]]:
    """Run the ExpAgent agentic loop and return a ScientificDecision."""
    policy = ContextPolicy.for_model(model)
    return _run_loop(ctx, model, api_base, api_key_env, mock, trace_dir, policy)


# ── Agentic loop ───────────────────────────────────────────────────


def _run_loop(ctx, model, api_base, api_key_env, mock, trace_dir, policy):
    """FC agentic loop: rebuild prompt each turn, keep only recent tool_pairs."""
    initial = build_initial_prompt(ctx)
    file_cache: dict[str, str] = {}
    trace: list[dict] = []
    effective_max = MAX_STEPS
    tool_pairs: list[dict] = []   # recent tool pairs for FC continuity
    compressed: list[str] = []    # compressed history for user_prompt
    paper_index: list[dict] = []  # saved papers (lightweight, always in prompt)

    for step in range(1, MAX_STEPS + MAX_EXTRA_AFTER_PROGRESS + 1):
        remaining = effective_max - step if step <= effective_max else 0

        # Rebuild user prompt from state each turn (CodingAgent style)
        user_prompt = _build_user_prompt(initial, compressed, paper_index, file_cache, policy)

        # Grace stop pressure
        if step >= effective_max - 2 or (step >= MAX_STEPS - 2 and effective_max == MAX_STEPS):
            user_prompt += (
                "\n\n[GRACE STOP: Very few steps remain. "
                "You MUST call finish now with your best scientific judgment.]"
            )

        # Build messages: system + recent tool_pairs + fresh user prompt
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for p in tool_pairs:
            messages.append(p)
        messages.append({"role": "user", "content": user_prompt})

        try:
            result = call_llm(
                model=model, api_base=api_base, api_key_env=api_key_env,
                system=SYSTEM_PROMPT, messages=messages,
                mock=mock, trace_dir=trace_dir, trace_label=f"step{step}",
            )
        except Exception as e:
            trace.append({"action": "api_error", "summary": str(e)[:100]})
            continue

        if result["type"] == "message":
            trace.append({"action": "message", "summary": result.get("content", "")[:100]})
            continue

        if result["type"] != "tool_call":
            trace.append({"action": "error", "summary": str(result.get("error", ""))[:100]})
            break

        name = result["name"]
        args = result["arguments"]

        if name == "search_papers":
            output = _execute_search(args)
            file_cache[f"search_{step}"] = output
            trace.append({"action": "search_papers", "summary": f"q={args.get('query','')[:60]}, {_count_results(output)}"})
            # Compress: record what was searched
            compressed.append(f"[Step {step}] Searched: {args.get('query','')[:120]} → {_count_results(output)}")
            if effective_max == MAX_STEPS:
                effective_max += MAX_EXTRA_AFTER_PROGRESS

        elif name == "read_file":
            path = args.get("path", "")
            output = read_file(path) if path else "No path provided."
            file_cache[path or f"file_{step}"] = output
            trace.append({"action": "read_file", "summary": f"path={path[:80]}"})
            compressed.append(f"[Step {step}] Read: {path[:120]} ({len(output)} chars)")
            if effective_max == MAX_STEPS:
                effective_max += MAX_EXTRA_AFTER_PROGRESS

        elif name == "save_paper":
            output = save_paper(
                paper_id=args.get("paper_id", ""),
                title=args.get("title", ""),
                first_author=args.get("first_author", ""),
                year=args.get("year"),
                abstract=args.get("abstract", ""),
                url=args.get("url", ""),
                code_url=args.get("code_url", ""),
                one_liner=args.get("one_liner", ""),
            )
            entry = {
                "title": args.get("title", ""),
                "first_author": args.get("first_author", ""),
                "year": args.get("year"),
                "one_liner": args.get("one_liner", ""),
                "paper_id": args.get("paper_id", ""),
                "slug": _slugify(args.get("title", "")),
            }
            paper_index.append(entry)
            trace.append({"action": "save_paper", "summary": args.get("title", "")[:80]})
            compressed.append(f"[Step {step}] Saved paper: {entry['title'][:120]}")

        elif name == "finish":
            yaml_text = args.get("decision_yaml", "")
            try:
                decision = _parse_decision(yaml_text)
            except Exception as e:
                trace.append({"action": "finish", "summary": f"parse error: {str(e)[:80]}"})
                tool_pairs = _make_pair(name, args, f"Parse error: {e}. Fix and call finish again.", step)
                continue

            vr = validate_decision(decision)
            trace.append({"action": "finish", "summary": decision.summary[:100]})
            if vr.status == "ok":
                return decision, trace
            issues_text = "\n".join(f"- {i}" for i in vr.issues)
            trace.append({"action": "validate", "summary": f"{len(vr.issues)} issues"})
            tool_pairs = _make_pair(name, args, f"Validation issues:\n{issues_text}\nFix and call finish again.", step)
            continue

        else:
            output = f"Unknown tool: {name}"
            trace.append({"action": "unknown", "summary": name})

        # Add new tool pair, limit to MAX_TOOL_PAIRS
        tool_pairs += _make_pair(name, args, output, step)
        if len(tool_pairs) > MAX_TOOL_PAIRS * 2:
            tool_pairs = tool_pairs[-(MAX_TOOL_PAIRS * 2):]

    # Loop exhausted
    from .models import ScientificConclusion
    return ScientificDecision(
        summary="Step budget exhausted.",
        confidence="low",
        conclusion=ScientificConclusion(status="inconclusive",
                                        rationale="ExpAgent ran out of steps."),
        evidence=[], recommended_actions=[],
        risks=["ExpAgent step budget exhausted"],
    ), trace


# ── Helpers ───────────────────────────────────────────────────────


def _build_user_prompt(initial: str, compressed: list[str], paper_index: list[dict],
                       file_cache: dict[str, str], policy: ContextPolicy) -> str:
    """Rebuild user prompt from state each turn."""
    parts = [initial]

    if paper_index:
        parts.append("\n## Saved Papers")
        for i, p in enumerate(paper_index[-policy.paper_index_entries:], 1):
            yr = f" ({p['year']})" if p.get('year') else ""
            parts.append(f"[{i}] {p['title']}{yr} · {p.get('first_author','?')} et al.")
            parts.append(f"    {p.get('one_liner','')[:120]}")
            parts.append(f"    paper: {p.get('paper_id','')}  file: papers/{p.get('slug','')}.md")

    if compressed:
        parts.append("\n## Step History")
        shown = compressed[-policy.step_history:]
        for line in shown:
            parts.append(line)

    if file_cache:
        entries = list(file_cache.items())[-policy.file_cache_count:]
        parts.append("\n## Recent File Reads")
        for key, text in entries:
            tail = text[-policy.file_cache_chars:]
            parts.append(f"[{key}] ({len(text)} chars, tail {len(tail)}):\n{tail}")

    parts.append("\n---\nWhat is your next action?")
    return "\n".join(parts)


def _make_pair(name: str, args: dict, output: str, step: int) -> list[dict]:
    """Create the [assistant_tool_call, tool_result] pair for FC."""
    return [
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": f"call_{step}", "type": "function",
                         "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}]},
        {"role": "tool", "tool_call_id": f"call_{step}", "content": output[:8000]},
    ]


def _make_assistant_msg(name: str, args: dict, step: int) -> dict:
    return _make_pair(name, args, "", step)[0]


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
        a = ", ".join(r.authors[:3])
        if len(r.authors) > 3:
            a += " et al."
        v = f" ({r.venue})" if r.venue else ""
        y = f" [{r.year}]" if r.year else ""
        lines.append(f"\n{i}. {r.title}{v}{y}")
        lines.append(f"   Authors: {a}  ID: {r.paper_id}")
        if r.abstract:
            lines.append(f"   Abstract: {r.abstract[:300]}")
        if r.url:
            lines.append(f"   URL: {r.url}")
    lines.append("\n---")
    lines.append("If any paper is a relevant baseline, call save_paper to persist it to the paper library.")
    return "\n".join(lines)


def _count_results(output: str) -> str:
    m = re.search(r"Found (\d+) papers?", output)
    return f"{m.group(1)}p" if m else "?"


def _parse_decision(yaml_text: str) -> ScientificDecision:
    if not yaml_text:
        raise ValueError("Empty")
    yaml_text = yaml_text.strip()
    for f in ("```yaml", "```"):
        if yaml_text.startswith(f):
            yaml_text = yaml_text[len(f):].strip()
    if yaml_text.endswith("```"):
        yaml_text = yaml_text[:-3].strip()
    try:
        data = yaml.safe_load(yaml_text)
    except yaml.YAMLError:
        data = json.loads(yaml_text)
    return ScientificDecision.model_validate(data)


def _slugify(title: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", title.strip()).strip("_")[:80]
