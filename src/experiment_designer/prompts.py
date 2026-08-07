"""System prompt, tool schemas, and turn-prompt builders.

Style-aligned with CodingAgent's controller/prompts.py:
- SYSTEM_PROMPT: the system-level instruction
- TOOLS: Function Calling tool schemas
- build_turn_prompt: rebuild user prompt fresh from state each turn
- build_initial_prompt: first-turn prompt from AdvisorContext
- build_plan_prompt / build_revise_prompt: planner.py wrappers
"""

from __future__ import annotations

from .context import LoopState
from .context_policy import ContextPolicy
from .models import AdvisorContext, ArtifactRef


# ── Function Calling tool schemas ──────────────────────────────────


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_papers",
            "description": "Search for scientific papers. Be specific and purposeful — not blind keyword searches.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Specific search query. E.g. 'channel attention image classification CIFAR-10 benchmark' not 'attention'"},
                    "source": {"type": "string", "enum": ["semantic_scholar", "dblp", "arxiv"], "default": "semantic_scholar", "description": "Which API to search"},
                    "venue_filter": {"type": "string", "description": "Filter by venue, e.g. 'CVPR', 'ICLR', 'NeurIPS'. Leave empty for no filter."},
                    "year_from": {"type": "integer", "description": "Only papers published after this year"},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file to see its content. After reading, immediately call note_finding to record key insights so you don't forget.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to read"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_paper",
            "description": "Save a paper to the library for later reference. Use when search results contain papers worth citing as baselines or SOTA.",
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_id": {"type": "string", "description": "arXiv id, DOI, or paper id"},
                    "title": {"type": "string", "description": "Paper title"},
                    "first_author": {"type": "string", "description": "First author's last name"},
                    "year": {"type": "integer", "description": "Publication year"},
                    "abstract": {"type": "string", "description": "Paper abstract"},
                    "url": {"type": "string", "description": "Paper URL"},
                    "code_url": {"type": "string", "description": "Code repository URL if known"},
                    "one_liner": {"type": "string", "description": "One sentence: why this matters for the current research"},
                },
                "required": ["paper_id", "title", "one_liner"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "note_finding",
            "description": "Record a key finding after reading a file or paper. This persists in context so you don't need to re-read the same file. Use after read_file to capture what you learned.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Short label, e.g. 'SENet CIFAR-10 training setup'"},
                    "finding": {"type": "string", "description": "What you found. Be specific: numbers, method names, hyperparameters."},
                    "source": {"type": "string", "description": "Which file or paper this came from"},
                },
                "required": ["topic", "finding"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Output your scientific decision as a JSON object when you have enough information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "decision_json": {"type": "string", "description": "Complete JSON object with the ScientificDecision structure"},
                },
                "required": ["decision_json"],
            },
        },
    },
]


# ── System prompt ──────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a machine learning research scientist serving as a scientific advisor.
Your job is to analyze research situations and give structured scientific recommendations.

## Your role

You receive a description of the current research situation from a project manager (ResAgent).
The situation might be:
- A new research idea that needs an experiment plan designed
- Experiment results that need scientific interpretation
- A failure that needs diagnosis (is it scientific, code, or environment?)
- A request to compare methods or review an existing plan
- Any other scientific question about ML experiments

Your task: observe the situation, use tools when you need more information, then output a ScientificDecision.

## Tools

You can call these tools. Return them as JSON actions.

### search_papers
Search for scientific papers. Use this BEFORE designing experiments (to find SOTA baselines), when analyzing results (to see how others solved similar problems), or when diagnosing failures.
Format:
{"thinking": "why you need to search and what you're looking for", "action": "search_papers", "query": "specific directed query", "source": "semantic_scholar", "venue_filter": "optional: CVPR|ICLR|NeurIPS|ICML|AAAI", "year_from": 2022, "max_results": 3}

Rules for searching:
- Be SPECIFIC. Not "attention mechanism" but "channel attention image classification CIFAR-10 benchmark"
- Use venue_filter to narrow to top venues when looking for SOTA
- Search with a purpose, not blindly — explain in "thinking" WHY you need this information
- Default to semantic_scholar. Use "arxiv" for very recent preprints. Use "dblp" for venue-specific searches.

### read_file
Read an artifact file (experiment result, log, report). Use this when you need to see actual numbers from experiment results.
Format:
{"thinking": "why you need to read this file", "action": "read_file", "path": "/path/to/result.md"}

### finish
Call finish with a complete JSON object in the decision_json field. The value is a JSON-encoded string containing your ScientificDecision.

Example call:
{"thinking": "final summary of reasoning", "action": "finish", "decision_json": "{\"summary\": \"...\", \"confidence\": \"medium\", \"conclusion\": {\"status\": \"supported\", \"rationale\": \"...\"}, ...}"}

## decision_json structure (MUST be valid JSON)

The JSON object inside decision_json follows this schema. Include all required fields.

Required fields: summary, confidence, conclusion, evidence, recommended_actions, risks
Optional fields: experiment_plan, result_analysis, failure_diagnosis, needs_user_input

{
  "summary": "one sentence summarizing your scientific verdict",
  "confidence": "medium",
  "conclusion": {
    "status": "supported",
    "rationale": "detailed scientific reasoning"
  },
  "evidence": [
    {"source": "literature", "description": "what this evidence shows"}
  ],
  "experiment_plan": {
    "version": 1,
    "goal": {
      "summary": "one sentence",
      "hypothesis": "specific testable claim",
      "success_criteria": ["concrete criterion"]
    },
    "experiment_matrix": {
      "datasets": [{"name": "...", "rationale": "..."}],
      "methods": [{"name": "...", "type": "baseline", "implementation_status": "needs_repro", "rationale": "..."}],
      "metrics": [{"name": "...", "rationale": "..."}]
    },
    "tasks": {
      "coding_tasks": [{"id": "code_001", "workspace_path": "", "task_goal": "...", "rationale": "..."}],
      "repro_tasks": [],
      "run_tasks": []
    },
    "analysis_plan": {"comparisons": [""], "plots": [""], "failure_checks": [""]},
    "risks": [{"description": "...", "mitigation": "..."}]
  },
  "result_analysis": {
    "summary": "analysis text",
    "findings": ["key finding"]
  },
  "failure_diagnosis": {
    "failure_type": "scientific",
    "diagnosis": "what went wrong",
    "is_recoverable": true
  },
  "recommended_actions": [
    {
      "priority": "high",
      "type": "repro_task",
      "rationale": "WHY",
      "plan": {
        "kind": "repro_task",
        "task_goal": "summary",
        "paper_url": "...",
        "repo_url": "...",
        "experiment_goal": "concrete goal"
      }
    }
  ],
  "risks": ["risk description"],
  "needs_user_input": []
}

risks:  # ALWAYS include — what scientific risks exist?
  - "<risk description>"

needs_user_input:  # OPTIONAL — questions that need human answers
  - "<question>"
```

## Your responsibility

You decide WHAT to do and WHY. You are the scientific advisor. The orchestrator (ResAgent) handles WHERE and HOW.

For each recommended action, fill these scientific fields:
- kind, task_goal, rationale: REQUIRED for all actions
- paper_url, repo_url: which paper/repo (for repro tasks)
- experiment_goal: what experiment to run (for repro/run tasks)
- expected_metrics: what metrics to evaluate

Operational fields (workspace_path, constraints, verify_commands, compute_budget, expected_runtime) are NOT your concern — ResAgent fills them based on the execution environment. Leave them empty.

## Rules

### Scientific rigor
- Every claim MUST be backed by evidence. If there's no evidence, say so and mark confidence as low.
- ALWAYS check for missing baselines. If a comparison is missing a standard baseline, flag it.
- ALWAYS check for fairness: same dataset split, same epoch budget, same hyperparameter tuning budget.
- If experiment results are from bounded/small runs, note that they may not generalize.
- Success criteria must be concrete and measurable, not vague.

### Recommended actions
- Fill the scientific fields listed above. Do NOT fill operational fields.
- Actions should be prioritized: what's most scientifically important right now?
- Don't recommend more than 5 actions — prioritize.
- If the scientific direction looks unpromising, say so (status: not_supported) rather than recommending endless experiments.

### Tool usage
- Think BEFORE searching — what specific information do you need?
- Search for baselines when designing experiments.
- IMMEDIATELY after read_file, call note_finding with the key result. Do NOT re-read without recording first. If you skip note_finding, you WILL forget what you read.
- Check Findings section before re-reading — it may already have what you need.
- Don't search for the same thing twice.

### When to stop searching
- If after reading the paper and searching you still cannot find exact details (e.g., specific hyperparameters), record what you DO know via note_finding and PROCEED to finish.
- Not all papers report every detail. Mark uncertainty in your confidence and conclusion instead of endlessly searching.
- Before searching, check Findings — if you already recorded an answer, don't search for the same thing again.

### Confidence
- high: strong evidence from multiple sources, clear conclusion
- medium: reasonable evidence but gaps remain
- low: insufficient evidence, significant uncertainty, or contradictory signals
"""


def build_initial_prompt(ctx: AdvisorContext) -> str:
    """Build the initial user prompt for the agentic loop.

    This is called once at the start of advise(). The LLM will then
    choose actions (search_papers, read_file, finish) and the loop
    will execute them and feed results back.
    """
    parts: list[str] = []

    parts.append("## Current Situation\n")
    parts.append(ctx.situation)

    if ctx.artifacts:
        parts.append("\n## Available Artifacts\n")
        parts.append("You can read these files with read_file to see detailed results:\n")
        for a in ctx.artifacts:
            parts.append(f"- `{a.id}` ({a.type}): {a.summary}")
            if a.path:
                parts.append(f"  Path: `{a.path}`")

    if ctx.existing_plan:
        import yaml as _yaml
        parts.append("\n## Current Experiment Plan\n")
        parts.append("```yaml")
        parts.append(_yaml.dump(
            ctx.existing_plan.model_dump(exclude_defaults=False),
            allow_unicode=True, sort_keys=False,
        ).strip())
        parts.append("```")

    parts.append("\n---")
    parts.append("\nAnalyze the situation. Use tools if you need more information, then output your ScientificDecision.")

    return "\n".join(parts)


# ── Plan/revise prompt builders ──────────────────────────────────
# Used by planner.py to translate DesignInput → situation string
# for advise().
# They translate DesignInput / revision feedback into the situation string
# that AdvisorContext expects.


def build_plan_prompt(inp: "DesignInput") -> str:
    """Build a situation string for initial experiment design."""
    from .models import DesignInput
    parts: list[str] = []
    parts.append("TASK: Design an initial experiment plan for this research idea.")
    parts.append(f"\nResearch Idea: {inp.research_idea}")
    parts.append(f"Target Task: {inp.target_task}")
    parts.append(f"Compute Budget: GPU={inp.compute_budget.gpu}, max_runtime={inp.compute_budget.max_runtime}, max_trials={inp.compute_budget.max_trials}")
    if inp.constraints:
        parts.append("Constraints:")
        for c in inp.constraints:
            parts.append(f"  - {c}")
    if inp.existing_assets.implemented_methods:
        parts.append("Existing implementations:")
        for m in inp.existing_assets.implemented_methods:
            loc = f" at {m.location}" if m.location else ""
            parts.append(f"  - {m.name}{loc}")
    if inp.existing_assets.available_datasets:
        parts.append(f"Available datasets: {', '.join(inp.existing_assets.available_datasets)}")
    if inp.existing_assets.known_baselines:
        parts.append(f"Known baselines (not implemented): {', '.join(inp.existing_assets.known_baselines)}")
    if inp.literature_context:
        parts.append("Literature context:")
        for lc in inp.literature_context:
            parts.append(f"  - {lc}")
    return "\n".join(parts)


def build_turn_prompt(state: LoopState, policy: ContextPolicy) -> str:
    """Build a fresh user prompt from structured state each turn.

    CodingAgent style: the full prompt is rebuilt from state, not appended
    to a growing messages array. FC tool_pairs are handled separately in advisor.py.
    """
    parts: list[str] = [state.situation]

    # Findings from reading (LLM's conclusions, not raw text)
    if state.findings:
        parts.append("\n## Findings from Reading")
        shown = state.findings[-policy.paper_index_entries:]
        for i, f in enumerate(shown, 1):
            parts.append(f"[{i}] {f['topic']}")
            parts.append(f"    {f['finding'][:300]}")
            if f.get('source'):
                parts.append(f"    source: {f['source']}")

    # Paper index (always in context — lightweight entries)
    if state.paper_index:
        parts.append("\n## Saved Papers")
        shown = state.paper_index[-policy.paper_index_entries:]
        for i, p in enumerate(shown, 1):
            yr = f" ({p['year']})" if p.get('year') else ""
            parts.append(f"[{i}] {p['title']}{yr} · {p.get('first_author', '?')} et al.")
            parts.append(f"    {p.get('one_liner', '')[:policy.observation_tail]}")
            parts.append(f"    paper: {p.get('paper_id', '')}  file: papers/{p.get('slug', '')}.md")

    # Compressed step history (search queries + key results preserved)
    if state.compressed:
        parts.append("\n## Step History")
        shown = state.compressed[-policy.step_history:]
        for line in shown:
            parts.append(line)

    # File cache (recently read files — tail only)
    if state.file_cache:
        entries = list(state.file_cache.items())[-policy.file_cache_count:]
        parts.append("\n## Recent File Reads")
        for key, text in entries:
            tail = text[-policy.file_cache_chars:]
            parts.append(f"[{key}] ({len(text)} chars, tail {len(tail)}):\n{tail}")

    parts.append("\n---\nWhat is your next action?")
    return "\n".join(parts)


def build_revise_prompt(current_plan: "ExperimentPlan", feedback: str) -> str:
    """Build a situation string for plan revision."""
    import yaml as _yaml
    parts: list[str] = []
    parts.append("TASK: Revise the current experiment plan based on user feedback.")
    parts.append(f"\nUser Feedback: {feedback}")
    parts.append("\nCurrent Plan:")
    parts.append("```yaml")
    parts.append(_yaml.dump(current_plan.model_dump(exclude_defaults=False), allow_unicode=True, sort_keys=False).strip())
    parts.append("```")
    return "\n".join(parts)
