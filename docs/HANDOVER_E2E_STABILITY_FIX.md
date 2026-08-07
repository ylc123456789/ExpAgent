# ExpAgent E2E Stability Fix — Handover Document

**Date**: 2026-08-08
**Status**: Implemented in working tree; targeted verification passed
**Current local HEAD**: `107dc87 refactor: structured finish parameters — no JSON-in-JSON escaping`
**Uncommitted changes on top**: `src/experiment_designer/prompts.py`, `tests/test_e2e.py`, `tests/test_planner.py`
**Related docs**:
- `docs/HANDOVER_MULTI_TOOL_FIX.md`
- `docs/MULTI_TOOL_CALL_FIX_POSTMORTEM.md`

---

## 1. Executive Summary

The e2e failures were **not pure LLM non-determinism**. They were caused by a combination of:

1. an outdated e2e fixture pointing to a non-existent project path,
2. a system prompt that still described the old `decision_json` string contract after the implementation had moved to structured finish arguments,
3. a `finish` function schema that was too shallow to constrain nested experiment-plan fields,
4. planner/validator/test mismatches around required rationales.

Implemented fixes in this handover:

- Synced `SYSTEM_PROMPT` with the structured `finish` arguments.
- Deepened `TOOLS.finish` JSON schema to constrain nested `experiment_plan`, `recommended_actions`, and task rationale fields.
- Fixed `tests/test_e2e.py::simple_idea` to create a real minimal project under `tmp_path` instead of referencing `/home/cyl/my_project`.
- Added regression tests for the prompt/schema contract.

Verification:

```text
Non-e2e: 48 passed, 22 deselected
Targeted previously-failing e2e: 2 passed in 198.43s
.pytest_cache/v/cache/lastfailed: {}
```

---

## 2. Environment

Project is inside WSL:

```text
WSL distro: Ubuntu-D
Project root: /home/cyl/ExpAgent
Python env: /home/cyl/miniconda3/envs/ResAgent/bin/python
```

Run commands from PowerShell on Windows like:

```powershell
wsl -d Ubuntu-D -- bash -lc 'PYTHONPATH=/home/cyl/ExpAgent/src /home/cyl/miniconda3/envs/ResAgent/bin/python -m pytest /home/cyl/ExpAgent/tests -q'
```

---

## 3. Evidence That This Was Not Only "LLM 波动"

### 3.1 Last failed tests

Before the fix, `.pytest_cache/v/cache/lastfailed` contained:

```json
{
  "tests/test_e2e.py::TestPlanGeneration::test_simple_idea_generates_valid_plan": true,
  "tests/test_e2e.py::TestPlanGeneration::test_every_design_choice_has_rationale": true
}
```

After the targeted rerun, it is:

```json
{}
```

### 3.2 Non-existent fixture path caused search/read loops

`tests/test_e2e.py` used to hardcode:

```text
/home/cyl/my_project
/home/cyl/my_project/models/resnet.py
```

That path does not exist in WSL. In the failing run:

```text
runs/tests/plan/20260807-214704/state.json
```

the trace had:

```text
47 trace actions
30 search_papers
7 read_file
3 finish attempts
NO_RESULT
```

The logs explicitly showed the model reacting to `read_file` failure and continuing to search. This is an environment/fixture bug, not model randomness.

### 3.3 Missing nested rationale was a real schema/contract gap

In:

```text
runs/tests/plan/20260807-215038/state.json
```

the final result had:

```json
"run_tasks": [
  {
    "id": "run_001",
    "command_goal": "Run 10-epoch training for all 3 arms with identical schedule",
    "rationale": ""
  }
]
```

while `recommended_actions` did contain a run_task rationale. The validator rejects empty task rationale, and `test_every_design_choice_has_rationale` checks it. The system had no schema-level constraint preventing `experiment_plan.tasks.run_tasks[].rationale` from being empty.

### 3.4 Prompt and implementation were out of sync

Commit `107dc87` moved finish to structured parameters, but `SYSTEM_PROMPT` still said:

```text
Call finish with a complete JSON object in the decision_json field.
The value is a JSON-encoded string containing your ScientificDecision.
```

It also still contained the old nested `decision_json` example and a trailing YAML fragment:

```text
risks:  # ALWAYS include ...
needs_user_input:  # OPTIONAL ...
```

So the function schema and advisor had moved on, but the prompt had not.

---

## 4. Changes Implemented

### 4.1 `src/experiment_designer/prompts.py` — prompt sync

The `SYSTEM_PROMPT` finish section now matches the structured arguments contract.

Key points now stated explicitly:

- `finish` uses structured fields, not a string payload.
- Required minimum fields:
  - `summary`
  - `confidence`
  - `conclusion_status`
  - `conclusion_rationale`
  - `evidence`
  - `recommended_actions`
  - `risks`
- `conclusion_status` enum:
  - `supported`
  - `not_supported`
  - `inconclusive`
  - `needs_more_experiments`
- Every dataset, method, metric, coding task, repro task, and run task must include non-empty `rationale`.
- `method.type` enum:
  - `new_method`
  - `baseline`
  - `ablation`
- `method.implementation_status` enum:
  - `needs_code`
  - `needs_repro`
  - `existing`
- `recommended_actions[].plan.kind` must match the action type.
- Operational fields may be left empty when unknown.

Removed from the prompt:

- `decision_json`
- old "JSON-encoded string" instruction
- trailing YAML fragment inside the JSON schema section

### 4.2 `src/experiment_designer/prompts.py` — deepened finish schema

Added nested JSON schemas and wired them into `TOOLS.finish.parameters`.

Important constraints now present in schema:

#### Top-level finish required fields

```text
summary
confidence
conclusion_status
conclusion_rationale
evidence
recommended_actions
risks
```

Also:

```text
evidence: minItems = 1
risks: minItems = 1
```

#### Evidence item

Required:

```text
source
description
```

`source` enum:

```text
artifact
literature
reasoning
```

#### Recommended action

Required:

```text
priority
type
rationale
plan
```

`type` enum:

```text
repro_task
coding_task
run_task
literature_search
literature_reference
ask_user
```

`plan.kind` enum:

```text
coding_task
repro_task
run_task
literature_search
ask_user
literature_reference
```

#### Experiment plan

If `experiment_plan` is present, required:

```text
goal
experiment_matrix
tasks
analysis_plan
risks
```

`goal` required:

```text
summary
hypothesis
success_criteria
```

`success_criteria`:

```text
minItems = 1
```

`experiment_matrix` required:

```text
datasets
methods
metrics
```

each with:

```text
minItems = 1
```

`methods[]` required:

```text
name
type
implementation_status
rationale
```

`tasks` required keys:

```text
coding_tasks
repro_tasks
run_tasks
```

Task item schemas now require rationale:

- `coding_tasks[]`: `id`, `task_goal`, `rationale`
- `repro_tasks[]`: `id`, `paper_url`, `experiment_goal`, `rationale`
- `run_tasks[]`: `id`, `command_goal`, `rationale`

This directly targets the previous failure mode:

```json
run_tasks[0].rationale = ""
```

### 4.3 `tests/test_e2e.py` — real fixture path

`simple_idea` now creates a real minimal project under pytest `tmp_path`:

```text
<tmp>/my_project/models/resnet.py
```

and uses that real path in both:

- the natural-language `research_idea`
- `ExistingMethod.location`

This removes the deterministic `read_file("/home/cyl/my_project/...") -> file not found` loop.

### 4.4 `tests/test_planner.py` — regression tests

Added `TestFinishSchema` with two tests:

1. `test_finish_tool_schema_constrains_nested_plan`
   - asserts method enums are constrained
   - asserts `methods[].rationale` is required
   - asserts `run_tasks[].command_goal` and `run_tasks[].rationale` are required

2. `test_system_prompt_matches_structured_finish`
   - asserts `SYSTEM_PROMPT` no longer contains `decision_json`
   - asserts it uses the structured `conclusion_status` contract

---

## 5. Verification

### 5.1 Non-e2e tests

Command:

```powershell
wsl -d Ubuntu-D -- bash -lc 'PYTHONPATH=/home/cyl/ExpAgent/src /home/cyl/miniconda3/envs/ResAgent/bin/python -m pytest /home/cyl/ExpAgent/tests -q'
```

Result:

```text
48 passed, 22 deselected in 2.21s
```

### 5.2 Targeted previously-failing e2e

Command:

```powershell
wsl -d Ubuntu-D -- bash -lc 'PYTHONPATH=/home/cyl/ExpAgent/src /home/cyl/miniconda3/envs/ResAgent/bin/python -m pytest /home/cyl/ExpAgent/tests/test_e2e.py::TestPlanGeneration::test_simple_idea_generates_valid_plan /home/cyl/ExpAgent/tests/test_e2e.py::TestPlanGeneration::test_every_design_choice_has_rationale -q -o addopts=""'
```

Result:

```text
2 passed in 198.43s (0:03:18)
```

### 5.3 Pytest failure cache

After the targeted e2e:

```json
.pytest_cache/v/cache/lastfailed == {}
```

### 5.4 Latest artifacts

Latest plan runs:

```text
runs/tests/plan/20260807-225116
runs/tests/plan/20260807-224900
```

No `file not found` / `No such file` was found in the latest run artifacts.

---

## 6. Current Repository State

`git status --short` still shows several source files as modified:

```text
 M src/experiment_designer/__init__.py
 M src/experiment_designer/advisor.py
 M src/experiment_designer/llm.py
 M src/experiment_designer/models.py
 M src/experiment_designer/planner.py
 M src/experiment_designer/prompts.py
 M src/experiment_designer/tools.py
 M tests/test_e2e.py
 M tests/test_planner.py
```

But ignoring line-ending whitespace, the actual logic diff is only:

```text
src/experiment_designer/prompts.py
tests/test_e2e.py
tests/test_planner.py
```

The other modified source files are pre-existing line-ending-only differences and were not logically edited in this handover.

Do not commit the line-ending-only files accidentally. Review with:

```powershell
wsl -d Ubuntu-D -- git -C /home/cyl/ExpAgent diff --ignore-space-at-eol --stat
wsl -d Ubuntu-D -- git -C /home/cyl/ExpAgent diff --ignore-space-at-eol -- src/experiment_designer/prompts.py tests/test_e2e.py tests/test_planner.py
```

---

## 7. Remaining Issues / Risks

### 7.1 `experiment_plan: null` can still happen in design-mode calls

Latest run:

```text
runs/tests/plan/20260807-225116/state.json
```

has:

```json
"experiment_plan": null
```

but `recommended_actions` have rationales. This allowed `test_every_design_choice_has_rationale` to pass, but it exposes two remaining gaps:

1. `validate_decision()` does not know the call is a design-mode `plan()` request, so it does not require `experiment_plan`.
2. `test_every_design_choice_has_rationale` does not assert datasets/methods/tasks are non-empty before checking rationales, so some checks can pass vacuously when `experiment_plan` is null.

Recommended next decision: decide whether `plan()` / `revise()` must always return an embedded `experiment_plan`. If yes, enforce it explicitly instead of relying on e2e quality assertions.

### 7.2 Full e2e was not rerun

Only the two previously failing e2e tests were rerun. Before considering this fully closed, run the full e2e suite once:

```powershell
wsl -d Ubuntu-D -- bash -lc 'PYTHONPATH=/home/cyl/ExpAgent/src /home/cyl/miniconda3/envs/ResAgent/bin/python -m pytest /home/cyl/ExpAgent/tests/test_e2e.py -q -o addopts=""'
```

Expect it to be slow and to consume real DeepSeek/search API budget.

### 7.3 Loop guards are still minimal

The system still lacks hard guards for:

- repeated reads of the same missing file,
- repeated semantically identical searches,
- many searches without new `save_paper` / `note_finding` progress.

The fixture fix removes the known missing-file trigger, but the general guard is still not implemented.

### 7.4 Search/API rate variability remains

Full e2e still uses real DeepSeek and real paper search APIs. Rate limits, fallbacks, and search relevance can still affect LLM behavior. For truly deterministic CI, e2e should be split into smoke and quality tiers, with recorded search/LLM fixtures for the deterministic tier.

---

## 8. Recommended Next Steps

Priority order:

1. **Decide the design-mode contract**
   - If `plan()` must always produce `experiment_plan`, enforce it in code or validation.
   - If `experiment_plan` is optional, make e2e tests assert fallback validity explicitly.

2. **Remove vacuous e2e passes**
   In `test_every_design_choice_has_rationale`, first assert:
   ```python
   assert result.experiment_matrix.datasets
   assert result.experiment_matrix.methods
   assert result.experiment_matrix.metrics
   assert result.tasks.coding_tasks or result.tasks.repro_tasks or result.tasks.run_tasks
   ```

3. **Add missing-file guard**
   When `read_file` returns file-not-found, write an explicit compressed warning such as:
   ```text
   MISSING FILE: <path>. Do not re-read; proceed with uncertainty or ask user.
   ```

4. **Add search dedup/progress guard**
   Cache search results per normalized query within a run. If several searches produce no new papers/findings, inject stop pressure.

5. **Split e2e into tiers**
   - `e2e_smoke`: structural correctness, must pass.
   - `e2e_quality`: baseline inference, hypothesis quality, revision stability; run with retries/nightly or report as quality metrics.

---

## 9. Rollback Notes

The uncommitted handover changes are limited to:

```text
src/experiment_designer/prompts.py
tests/test_e2e.py
tests/test_planner.py
```

To inspect before rollback:

```powershell
wsl -d Ubuntu-D -- git -C /home/cyl/ExpAgent diff --ignore-space-at-eol -- src/experiment_designer/prompts.py tests/test_e2e.py tests/test_planner.py
```

If the deeper schema ever causes provider-side rejection, check the raw response logs under:

```text
runs/tests/plan/<timestamp>/logs/*.response.txt
runs/tests/advisor/<timestamp>/logs/*.response.txt
```

The first relaxation candidates should be `minItems` / deep `required` fields, not the enum values.

---

## 10. Acceptance Criteria Met

- `SYSTEM_PROMPT` no longer describes `decision_json`.
- `finish` schema constrains nested method enums and task rationales.
- e2e fixture no longer references `/home/cyl/my_project`.
- Regression tests added.
- Non-e2e suite passes.
- Previously failing targeted e2e tests pass.
- `lastfailed` is empty.
