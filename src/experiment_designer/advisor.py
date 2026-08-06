"""Agentic loop for ExpAgent — the scientific advisor.

Uses Function Calling for reliable tool use.
Context: CodingAgent-style "rebuild prompt from state each turn" +
FC-compatible tool_pairs (keeps only recent 4 pairs, old info compressed).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from .context import LoopState
from .context_policy import ContextPolicy
from .llm import call_llm
from .models import AdvisorContext, ScientificDecision
from .prompts import SYSTEM_PROMPT, build_initial_prompt, build_turn_prompt
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
    run_dir: Path | None = None,
) -> tuple[ScientificDecision, list[dict]]:
    """Run the ExpAgent agentic loop and return a ScientificDecision.

    Args:
        run_dir: Output directory for this run (papers, logs). If None,
                 defaults to Path.cwd() / "runs" / <timestamp>.
    """
    if run_dir is None:
        from datetime import datetime, timezone
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        run_dir = Path.cwd() / "runs" / stamp
    policy = ContextPolicy.for_model(model)
    return _run_loop(ctx, model, api_base, api_key_env, mock, trace_dir, policy, run_dir)


# ── Agentic loop ───────────────────────────────────────────────────


def _run_loop(ctx, model, api_base, api_key_env, mock, trace_dir, policy, run_dir):
    """FC agentic loop: rebuild prompt each turn, keep only recent tool_pairs."""
    papers_dir = run_dir / "papers"
    state = LoopState(situation=build_initial_prompt(ctx))
    trace: list[dict] = []
    effective_max = MAX_STEPS
    tool_pairs: list[dict] = []   # recent tool pairs for FC continuity

    for step in range(1, MAX_STEPS + MAX_EXTRA_AFTER_PROGRESS + 1):
        if step > effective_max and effective_max >= MAX_STEPS:
            break  # no extra progress granted, stop at base budget

        # Rebuild user prompt from state each turn (CodingAgent style)
        user_prompt = build_turn_prompt(state, policy)

        # Grace stop pressure
        if step >= effective_max - 2:
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
            state.file_cache[f"search_{step}"] = output
            trace.append({"action": "search_papers", "summary": f"q={args.get('query','')[:60]}, {_count_results(output)}"})
            # Compress: record what was searched
            state.compressed.append(f"[Step {step}] Searched: {args.get('query','')[:120]} → {_count_results(output)}")
            if effective_max == MAX_STEPS:
                effective_max += MAX_EXTRA_AFTER_PROGRESS

        elif name == "read_file":
            path = args.get("path", "")
            output = read_file(path) if path else "No path provided."
            state.file_cache[path or f"file_{step}"] = output
            trace.append({"action": "read_file", "summary": f"path={path[:80]}"})
            state.compressed.append(f"[Step {step}] Read: {path[:120]} ({len(output)} chars)")
            if effective_max == MAX_STEPS:
                effective_max += MAX_EXTRA_AFTER_PROGRESS

        elif name == "note_finding":
            topic = args.get("topic", "")
            finding = args.get("finding", "")
            source = args.get("source", "")
            state.findings.append({"topic": topic, "finding": finding, "source": source})
            trace.append({"action": "note_finding", "summary": topic[:80]})
            output = f"Finding recorded: {topic}"

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
                output_dir=str(papers_dir),
            )
            entry = {
                "title": args.get("title", ""),
                "first_author": args.get("first_author", ""),
                "year": args.get("year"),
                "one_liner": args.get("one_liner", ""),
                "paper_id": args.get("paper_id", ""),
                "slug": _slugify(args.get("title", "")),
            }
            state.paper_index.append(entry)
            trace.append({"action": "save_paper", "summary": args.get("title", "")[:80]})
            state.compressed.append(f"[Step {step}] Saved paper: {entry['title'][:120]}")

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


def _make_pair(name: str, args: dict, output: str, step: int) -> list[dict]:
    """Create the [assistant_tool_call, tool_result] pair for FC."""
    return [
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": f"call_{step}", "type": "function",
                         "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}]},
        {"role": "tool", "tool_call_id": f"call_{step}", "content": output[:8000]},
    ]


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
