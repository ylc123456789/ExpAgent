# E1 主线收敛 — 设计说明（暂不实现）

> 状态：设计说明，不在 `codex/readability-cleanup` 分支实现
> 目标：让 ExpAgent 只剩一条生产主线 —— `advise()` → `ScientificDecision.recommended_actions`（V2 科学动作图）

## 1. 问题

当前 ExpAgent 对"下一步做什么"存在两套表示：

1. **V2（权威）**：`ScientificDecision.recommended_actions: list[ScientificAction]`，六类 capability，逻辑引用，无物理字段。跨模块唯一消费路径（ResAgent adapter 直接读它）。
2. **旧设计主线（仅 CLI/REPL 自用）**：`ScientificDecision.experiment_plan: ExperimentPlan | None` 及其模型树（`ExperimentPlan`/`TaskBundle`/`CodingTask`/`ReproTask`/`RunTask`/`ResearchGoal`/`ExperimentMatrix`/`MethodSpec`/`DatasetSpec`/`MetricSpec`/`AnalysisPlan`/`Risk`/`DesignInput`/`ComputeBudget`/`ExistingAssets`/`ExistingMethod`），以及 `plan()/revise()`（`controller/planner.py`）把 V2 actions **反投影**成 `TaskBundle`（`_extract_coding/repro/run_tasks`）。

反投影只映射 `modify_code → CodingTask`、`reproduce_experiment → ReproTask`、`execute_experiment → RunTask` 三类，**`analyze_results` / `search_literature` / `ask_user` 在旧视图中被丢弃**。此外 `CodingTask.workspace_path` 字段与 V2"不输出物理路径"原则相悖。

## 2. 迁移目标

- 生产主线唯一化：`advise()` 是唯一入口，`ScientificDecision.recommended_actions` 是唯一"下一步"表示。
- CLI/REPL 直接调用并展示 `advise()` 结果，不再经过 `plan()/revise()` 和 `ExperimentPlan`。
- 三类非实验 capability（`analyze_results`/`search_literature`/`ask_user`）不再因反投影丢失。

## 3. 迁移步骤（顺序执行，每步一提交 + 测试）

1. **CLI/REPL 直连 advise()**：`main.py` 的 `advise` 子命令与 `repl.py` 改为调用 `advise()` 并直接渲染 `ScientificDecision`（含 `recommended_actions` 的人类可读展示）。
2. **停用反投影**：删除 `planner.py` 的 `_extract_coding_tasks` / `_extract_repro_tasks` / `_extract_run_tasks` / `_populate_tasks_from_actions`；`plan()/revise()` 退化为调用 `advise()` 并返回 decision（若仍需过渡兼容，返回 decision 本身而非 ExperimentPlan）。
3. **补契约测试**：新增 CLI（`--no-interactive` / `advise`）、REPL、以及 ResAgent adapter 对 `recommended_actions` 消费路径的确定性测试；补 `analyze_results`/`search_literature`/`ask_user` 在展示中不丢失的用例。
4. **删除旧模型与展示代码**：验证通过后删除 `plan()`/`revise()`、`ExperimentPlan` 模型树、`validate()`、`presentation.py`、`repl.py`（若已直连则精简/删除）、`prompts/schemas.py` 中的 `experiment_plan` 嵌套 schema、`ScientificDecision.experiment_plan` 字段。

## 4. 契约影响（需总体审查）

- `ScientificDecision.experiment_plan` 字段在 phase0 冻结字段列表中（`tests/test_phase0_contract.py::test_cross_module_model_field_contracts`）。删除该字段 = 跨模块契约变化，需：
  1. 总体审查批准；
  2. 更新 phase0 lock（`__all__` 与 `ScientificDecision.model_fields` 的 hash/列表）。
- ResAgent adapter 当前**不消费** `experiment_plan`（只读 `summary`/`recommended_actions`/`analysis_required`/`supersedes_action_ids`），故删除对 ResAgent 运行期无影响，但契约冻结需要显式解冻。
- `agent.yaml` 能力卡与六类 capability 词表保持不变。

## 5. 非目标

- 不改 `SYSTEM_PROMPT` 与 V2 action contract（capability 名称、字段语义）。
- 不新增渲染框架或中间层；展示用 `presentation.py` 的现有原语直接渲染。
- 不在本分支实现。

## 6. 验收

- `advise()` 是唯一生产入口；CLI/REPL 无 `plan()/revise()` 调用。
- `recommended_actions` 六类 capability 在 CLI/REPL 展示中完整可见。
- ResAgent adapter 契约测试通过（`recommended_actions → AgentTask` 转换不变）。
- 旧 `ExperimentPlan`/`TaskBundle` 模型与 `validate()` 删除后全量测试通过（含更新后的 phase0 lock）。
- `SYSTEM_PROMPT` hash 不变；V2 contract 字段列表不变（除显式批准删除的 `experiment_plan`）。
