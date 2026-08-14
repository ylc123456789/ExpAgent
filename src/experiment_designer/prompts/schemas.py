"""Function Calling tool schemas for the ExpAgent agentic loop.

These JSON-Schema-like dicts are passed to the LLM as tool definitions.
They mirror the Pydantic models in experiment_designer.models so that
structured tool calls validate against the same shape.
"""

from __future__ import annotations


# ── Function Calling nested schemas ───────────────────────────────

_EVIDENCE_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "source": {"type": "string", "enum": ["artifact", "literature", "reasoning"]},
        "description": {"type": "string", "minLength": 1},
    },
    "required": ["source", "description"],
}

_SUGGESTED_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["coding_task", "repro_task", "run_task", "literature_search", "ask_user", "literature_reference", "result_analysis"]},
        "code_availability": {"type": "string", "enum": ["public", "upon_request", "none", ""]},
        "task_goal": {"type": "string"},
        "constraints": {"type": "array", "items": {"type": "string"}},
        "verify_commands": {"type": "array", "items": {"type": "string"}},
        "expected_artifacts": {"type": "array", "items": {"type": "string"}},
        "paper_url": {"type": "string"},
        "repo_url": {"type": "string"},
        "experiment_goal": {"type": "string"},
        "expected_metrics": {"type": "array", "items": {"type": "string"}},
        "command_goal": {"type": "string"},
        "expected_runtime": {"type": "string"},
        "requires_gpu": {"type": "boolean"},
        "search_query": {"type": "string"},
        "question": {"type": "string"},
    },
    "required": ["kind"],
}

_RECOMMENDED_ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "priority": {"type": "string", "enum": ["high", "medium", "low"]},
        "type": {"type": "string", "enum": ["repro_task", "coding_task", "run_task", "literature_search", "literature_reference", "ask_user", "result_analysis"]},
        "rationale": {"type": "string", "minLength": 1},
        "plan": _SUGGESTED_PLAN_SCHEMA,
        "action_id": {"type": "string", "description": "Unique, non-empty id within this decision."},
        "depends_on": {"type": "array", "items": {"type": "string"}, "description": "IDs of actions in this same decision that must complete before this one."},
        "project_ref": {"type": "string", "description": "Logical project identifier shared across dependent actions."},
        "workspace_intent": {"type": "string", "enum": ["shared", "isolated", ""], "description": "Workspace sharing intent for run/repro tasks. shared = operate on project_ref in place; isolated = private clone/copy; empty = undecided."},
        "required": {"type": "boolean", "description": "Whether ResAgent must complete this action before finishing. Default true; false only for genuinely optional follow-ups."},
    },
    "required": ["priority", "type", "rationale", "plan"],
}

_DATASET_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "split": {"type": "string"},
        "rationale": {"type": "string", "minLength": 1},
    },
    "required": ["name", "rationale"],
}

_METHOD_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "type": {"type": "string", "enum": ["new_method", "baseline", "ablation"]},
        "implementation_status": {"type": "string", "enum": ["needs_code", "needs_repro", "existing"]},
        "rationale": {"type": "string", "minLength": 1},
    },
    "required": ["name", "type", "implementation_status", "rationale"],
}

_METRIC_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "rationale": {"type": "string", "minLength": 1},
    },
    "required": ["name", "rationale"],
}

_CODING_TASK_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "minLength": 1},
        "workspace_path": {"type": "string"},
        "task_goal": {"type": "string", "minLength": 1},
        "constraints": {"type": "array", "items": {"type": "string"}},
        "verify_commands": {"type": "array", "items": {"type": "string"}},
        "expected_artifacts": {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": "string", "minLength": 1},
    },
    "required": ["id", "task_goal", "rationale"],
}

_REPRO_TASK_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "minLength": 1},
        "paper_url": {"type": "string", "minLength": 1},
        "repo_url": {"type": "string"},
        "experiment_goal": {"type": "string", "minLength": 1},
        "expected_metrics": {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": "string", "minLength": 1},
    },
    "required": ["id", "paper_url", "experiment_goal", "rationale"],
}

_RUN_TASK_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "minLength": 1},
        "command_goal": {"type": "string", "minLength": 1},
        "expected_runtime": {"type": "string"},
        "requires_gpu": {"type": "boolean"},
        "rationale": {"type": "string", "minLength": 1},
    },
    "required": ["id", "command_goal", "rationale"],
}

_GOAL_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "minLength": 1},
        "hypothesis": {"type": "string", "minLength": 1},
        "success_criteria": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 1}},
    },
    "required": ["summary", "hypothesis", "success_criteria"],
}

_EXPERIMENT_MATRIX_SCHEMA = {
    "type": "object",
    "properties": {
        "datasets": {"type": "array", "minItems": 1, "items": _DATASET_SCHEMA},
        "methods": {"type": "array", "minItems": 1, "items": _METHOD_SCHEMA},
        "metrics": {"type": "array", "minItems": 1, "items": _METRIC_SCHEMA},
    },
    "required": ["datasets", "methods", "metrics"],
}

_TASK_BUNDLE_SCHEMA = {
    "type": "object",
    "properties": {
        "coding_tasks": {"type": "array", "items": _CODING_TASK_SCHEMA},
        "repro_tasks": {"type": "array", "items": _REPRO_TASK_SCHEMA},
        "run_tasks": {"type": "array", "items": _RUN_TASK_SCHEMA},
    },
    "required": ["coding_tasks", "repro_tasks", "run_tasks"],
}

_ANALYSIS_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "comparisons": {"type": "array", "items": {"type": "string"}},
        "plots": {"type": "array", "items": {"type": "string"}},
        "failure_checks": {"type": "array", "items": {"type": "string"}},
    },
}

_PLAN_RISK_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {"type": "string", "minLength": 1},
        "mitigation": {"type": "string", "minLength": 1},
    },
    "required": ["description", "mitigation"],
}

_EXPERIMENT_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "version": {"type": "integer"},
        "goal": _GOAL_SCHEMA,
        "experiment_matrix": _EXPERIMENT_MATRIX_SCHEMA,
        "tasks": _TASK_BUNDLE_SCHEMA,
        "analysis_plan": _ANALYSIS_PLAN_SCHEMA,
        "risks": {"type": "array", "minItems": 1, "items": _PLAN_RISK_SCHEMA},
    },
    "required": ["goal", "experiment_matrix", "tasks", "analysis_plan", "risks"],
}

_RESULT_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "findings": {"type": "array", "items": {"type": "string"}},
    },
}

_FAILURE_DIAGNOSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "failure_type": {"type": "string", "enum": ["transient", "system", "scientific", "code", "data", "budget"]},
        "diagnosis": {"type": "string"},
        "is_recoverable": {"type": "boolean"},
    },
}


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
            "description": "Output your scientific decision as structured arguments. Fill at minimum: summary, confidence, conclusion_status, conclusion_rationale, evidence, recommended_actions, risks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "minLength": 1, "description": "One-sentence verdict"},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "conclusion_status": {"type": "string", "enum": ["supported", "not_supported", "inconclusive", "needs_more_experiments"]},
                    "conclusion_rationale": {"type": "string", "minLength": 1, "description": "Detailed scientific reasoning"},
                    "evidence": {"type": "array", "minItems": 1, "items": _EVIDENCE_ITEM_SCHEMA,
                        "description": "Evidence supporting the conclusion"},
                    "recommended_actions": {"type": "array", "items": _RECOMMENDED_ACTION_SCHEMA,
                        "description": "Prioritized actions. plan.kind must match the action type; operational fields may be empty."},
                    "experiment_plan": _EXPERIMENT_PLAN_SCHEMA,
                    "result_analysis": _RESULT_ANALYSIS_SCHEMA,
                    "failure_diagnosis": _FAILURE_DIAGNOSIS_SCHEMA,
                    "risks": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 1},
                        "description": "Scientific risks"},
                    "needs_user_input": {"type": "array", "items": {"type": "string"},
                        "description": "Questions for human"},
                },
                "required": ["summary", "confidence", "conclusion_status", "conclusion_rationale", "evidence", "recommended_actions", "risks"],
            },
        },
    },
]
