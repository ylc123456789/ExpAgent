# ReproAgent and CodingAgent Decoupling Requirements

## Purpose

ExpAgent is the upper-level experiment design agent. It must be able to call both
ReproAgent and CodingAgent without depending on hard-coded local paths or nested
repository assumptions.

Target relationship:

```text
ExpAgent
  -> ReproAgent
       -> CodingAgent
  -> CodingAgent
```

ReproAgent should still be able to use CodingAgent internally, but CodingAgent
must also be callable directly by ExpAgent for original experiment code,
ablation code, runners, and analysis scripts.

## Main Problem To Fix

ReproAgent should not assume that CodingAgent lives at a fixed relative path
inside the ReproAgent repository.

Examples of coupling to remove:

- Hard-coded paths such as `../CodingAgent`, `./CodingAgent`, `coding_agent/`, or
  project-specific absolute paths.
- Imports that only work when CodingAgent is physically nested inside
  ReproAgent.
- CLI commands that construct CodingAgent paths from ReproAgent's repository
  layout.
- Configuration defaults that silently point to a bundled CodingAgent copy and
  cannot be overridden.

## Desired Design

CodingAgent should be treated as an external dependency with a configurable
location and a stable invocation interface.

Recommended configuration shape:

```yaml
agents:
  codingagent_path: /home/cyl/CodingAgent
```

For larger systems, ExpAgent should be able to pass this path into ReproAgent:

```bash
reproagent run \
  --task repro_task.yaml \
  --codingagent-path /home/cyl/CodingAgent
```

ReproAgent should then use the provided path for every CodingAgent call.

## Required Changes In ReproAgent

1. Add a single configuration entry for CodingAgent location.

   Supported sources, in priority order:

   ```text
   CLI argument:     --codingagent-path
   Environment var:  CODINGAGENT_PATH
   Config file:      agents.codingagent_path
   Existing default: only as a fallback, if still needed
   ```

2. Route all CodingAgent calls through one adapter/module.

   Suggested shape:

   ```text
   reproagent/
     integrations/
       codingagent.py
   ```

   This adapter should be the only place that knows how to invoke CodingAgent.
   Other ReproAgent modules should call the adapter instead of building paths or
   subprocess commands directly.

3. Make path resolution explicit.

   The adapter should:

   - Expand `~`.
   - Resolve relative paths against the current working directory or config file
     location, with documented behavior.
   - Check that the path exists.
   - Check that it looks like a CodingAgent checkout.
   - Produce a clear error if the path is missing or invalid.

4. Avoid importing CodingAgent through fragile relative imports.

   Prefer one of these stable approaches:

   - Call CodingAgent through its CLI.
   - Call a small documented Python API, if CodingAgent exposes one.

   If Python imports are used, they should be isolated inside the adapter and
   should not mutate `sys.path` globally unless there is no better option.

5. Keep ReproAgent runnable as a standalone project.

   ReproAgent should still work outside ExpAgent. Users should be able to run it
   by providing a CodingAgent path through CLI/config/env.

## Required Changes In CodingAgent

CodingAgent should expose a stable interface that other agents can call without
knowing its internal file layout.

Preferred interface:

```bash
codingagent run --task coding_task.yaml --output output_dir
```

Minimum requirements:

- Accept a structured task file.
- Write outputs to a caller-provided output directory.
- Return a nonzero exit code on failure.
- Produce a machine-readable result file, for example:

  ```text
  coding_result.yaml
  ```

- Avoid assuming the caller's current working directory is the CodingAgent repo.

## Task Schema Compatibility

ExpAgent expects to generate coding tasks with this kind of shape:

```yaml
id: code_001
repo_path: /path/to/target/repo
task_goal: Implement the proposed method variant.
constraints:
  - Do not change baseline training behavior.
verify_commands:
  - python -m pytest
expected_artifacts:
  - patch.diff
  - verification_report.md
```

ReproAgent should be able to forward equivalent coding tasks to CodingAgent
without rewriting them around local repository layout assumptions.

## ExpAgent Integration Expectations

ExpAgent will maintain its own config, for example:

```yaml
agents:
  reproagent_path: /home/cyl/reproagent
  codingagent_path: /home/cyl/CodingAgent
```

Expected call patterns:

```text
ExpAgent -> ReproAgent:
  pass repro task + codingagent_path

ExpAgent -> CodingAgent:
  pass coding task directly
```

This means ReproAgent and CodingAgent should both support being called from
outside their own repository directories.

## Acceptance Criteria

ReproAgent decoupling is done when:

- No ReproAgent module outside the CodingAgent adapter constructs hard-coded
  CodingAgent paths.
- ReproAgent can run with `--codingagent-path /home/cyl/CodingAgent`.
- ReproAgent can run with `CODINGAGENT_PATH=/home/cyl/CodingAgent`.
- Invalid CodingAgent paths produce clear errors.
- ReproAgent tests cover path resolution and adapter invocation.

CodingAgent interface stabilization is done when:

- There is a documented CLI or Python API for external callers.
- The interface accepts structured task input.
- The interface writes structured result output.
- Calls work from a directory outside the CodingAgent repository.

ExpAgent compatibility is done when:

- ExpAgent can validate configured ReproAgent and CodingAgent paths.
- ExpAgent can generate task files that match the downstream schemas.
- ExpAgent can perform a dry-run compatibility check without modifying either
  downstream project.

## Notes

This document is a downstream requirement record. ExpAgent should not directly
modify ReproAgent or CodingAgent in this development thread. Changes to those
projects should be handled in their own repositories and then synchronized back
through configured paths or version-pinned external references.
