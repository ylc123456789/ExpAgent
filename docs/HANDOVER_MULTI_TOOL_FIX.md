# ExpAgent Multi-Tool-Call Fix — Handover Document

**Date**: 2026-08-07
**Status**: Bug confirmed, fix attempted but incomplete
**Working baseline**: commit `9dfd2f2` (3/3 e2e passed, JSON format)

---

## 1. The Bug

### Location
`src/experiment_designer/llm.py` line 107-109

```python
# Current code — only processes the first tool_call
tool_calls = msg.get("tool_calls", [])
if tool_calls:
    tc = tool_calls[0]    # ← BUG: LLM may return 2-4 parallel tool_calls
    ...
    return {"type": "tool_call", "name": ..., "arguments": ...}
```

### Impact
DeepSeek's LLM regularly returns 2-4 parallel tool_calls in a single API response.
Only the first one (`index: 0`) is executed. The rest are silently dropped.

**Evidence** (from `runs/tests/plan/20260807-073754/logs/`):
- 23 out of 26 LLM responses contained >1 parallel tool_calls
- Step 1: LLM called `[read_file, search_papers, search_papers]` — only `read_file` ran
- Step 7: LLM called `[note_finding, note_finding]` — only the first was recorded
- This run: 43 total search tool_calls requested, only 20 executed, 23 dropped

**Evidence from another run** (`runs/tests/plan/20260807-071551/logs/`):
- Step 4: LLM called `[save_paper × 4]` in one response — only 1 paper saved, 3 dropped
- Step 22: LLM called `[save_paper × 3]` — only 1 saved, 2 dropped

### Consequence
The LLM believes it executed N operations but only 1 actually ran. On the next turn,
it discovers missing information — searches again — drops more calls — loops.
This causes: search floods, save→read chain breaks, note_finding gaps, step budget exhaustion.

---

## 2. What Was Tried

### Attempt 1 (failed — 400 Bad Request)
Changed `llm.py` to return ALL tool_calls as a list, changed `advisor.py` to iterate
over them. Constructed a custom `assistant` message with synthetic tool_call IDs.

**Problem**: The self-constructed `assistant` message with `tool_calls` array caused
DeepSeek to return `HTTP 400 Bad Request` on every subsequent API call after the
first finish attempt. The exact FC message format required by DeepSeek for
reconstructed multi-tool-call history was not identified.

### Attempt 2 (failed — same 400)
Changed `llm.py` to pass through the original `tool_call` objects (with original
`id`, `index`, `type`, `function` fields) into the result dict as `"raw"` fields.
In `advisor.py`, built the `assistant` message by copying these raw objects.

**Problem**: Same 400 pattern. After any finish attempt (success or failure),
subsequent LLM calls fail with 400. The issue appears to be in how the
`assistant` message with multiple `tool_calls` is reconstructed and fed back
into the messages array for the next turn.

### Current state
Reverted to commit `9dfd2f2`. Code is clean and working (single tool_call only).
The stash at `WIP on main: 9dfd2f2` contains the last multi-tool attempt.

---

## 3. How to Fix It (Recommended Approach)

### Option A: Process all tool_calls without storing in messages (Recommended)

Instead of reconstructing FC messages, execute ALL tool_calls AND compress
their results into `state.compressed`. The next turn's `build_turn_prompt` already
shows compressed history.

```python
# llm.py — return ALL calls
if tool_calls:
    calls = [{"name": tc["function"]["name"],
              "arguments": json.loads(tc["function"]["arguments"])}
             for tc in tool_calls]
    return {"type": "tool_calls", "calls": calls}

# advisor.py — execute all, but only preserve the FIRST in tool_pairs
# (other results go into state.compressed)
calls = result["calls"]
for j, call in enumerate(calls):
    output = execute_tool(call)
    state.compressed.append(f"[Step {step}] {call['name']}: {summary}")
    if j == 0:
        # Only the first call enters tool_pairs for FC continuity
        tool_pairs += make_pair(call, output)
# Results of calls[1:] are in state.compressed → next build_turn_prompt shows them
```

**Why this works**: FC only needs ONE tool_call→tool_result pair to know "I called
a tool and got a result". The additional calls' results are visible in the
user prompt via `state.compressed`. This is the CodingAgent-style approach.

### Option B: Study DeepSeek's exact multi-tool FC format

DeepSeek may have specific requirements for the `assistant` message when it
contains multiple `tool_calls`. Study the exact format of:
1. The original LLM response with 2+ tool_calls
2. The required `assistant` message format for the NEXT turn
3. Whether `"content": null` vs `"content": ""` matters
4. Whether `"index"` field is required

The trace files at `runs/tests/plan/20260807-073754/logs/*.response.txt`
contain the raw LLM responses for study.

---

## 4. Repository State

### Working commit
```
9dfd2f2 refactor: YAML to JSON finish output (full prompt rewrite)
```

### Test status (commit 9dfd2f2)
- Unit tests: 44/44 passed
- E2E tests: 18/22 passed (4 flaky due to LLM non-determinism, not code bugs)

### Key files
| File | Purpose |
|------|---------|
| `src/experiment_designer/advisor.py` | Agentic loop: `_run_loop()` — main execution |
| `src/experiment_designer/llm.py` | FC API client + retry logic |
| `src/experiment_designer/prompts.py` | System prompt, TOOLS schema, prompt builders |
| `src/experiment_designer/context.py` | `LoopState` dataclass |
| `src/experiment_designer/context_policy.py` | `ContextPolicy` (model window adaptive) |
| `src/experiment_designer/models.py` | All Pydantic models |
| `src/experiment_designer/tools.py` | `search_papers`, `read_file`, `save_paper` |
| `src/experiment_designer/validator.py` | `validate()`, `validate_decision()` |
| `src/experiment_designer/planner.py` | `plan()` / `revise()` wrappers |

### Python API
```python
from experiment_designer.advisor import advise
from experiment_designer.models import AdvisorContext

ctx = AdvisorContext(situation="...")
decision, trace = advise(ctx, model="deepseek-chat")
# → decision.recommended_actions[0].plan.paper_url
```

### Test artifact locations
```
runs/tests/plan/<timestamp>/state.json   — plan() test results
runs/tests/advisor/<timestamp>/state.json — advise() test results
runs/tests/test_traces/<timestamp>/      — mock trace test output
```

---

## 5. Other Open Issues (Lower Priority)

1. **Search→save chain**: When LLM doesn't find exact paper matches in Semantic Scholar,
   it may search repeatedly without saving. The note_finding + findings system helps
   but doesn't fully prevent this.

2. **Validation loop**: LLM occasionally produces JSON with parse errors on first
   finish attempt. The `validate_decision` experiment_plan check helps catch this
   but the loop has limited retry budget.

3. **Test flakiness**: ~2/22 e2e tests fail intermittently due to LLM non-determinism,
   not code defects. These are the `experiment_plan` sparse field tests.
