# ExpAgent 只读审查报告与整理方案

> 日期：2026-08-22
> 范围：ExpAgent（`src/experiment_designer/`）
> 性质：只读审查 + 安全整理分类，行为风险项单独列报
> 依据：`ResAgent/docs/active/FOUR_MODULE_CODE_REVIEW_AND_SIMPLIFICATION_PLAN.md`

## 1. 基线（Phase A）

| 项目 | 值 |
|---|---|
| 仓库路径 | `/home/cyl/ExpAgent` |
| 默认分支 / HEAD | `main` / `11e9a27`（干净，与远端同步） |
| 整理分支 | `codex/readability-cleanup`（自 `main`） |
| Python | 系统 `python3` → 3.12（无专用 conda env） |
| 全量测试 | **79 passed, 22 deselected (e2e)** |
| CLI 入口 | `python3 -m experiment_designer.main`（`expagent` 控制台脚本未安装到 PATH，但 `pyproject.toml` 已声明） |
| 生产依赖 | pydantic>=2.0、pyyaml、defusedxml、pdfplumber |
| 生产文件 | 24 个 `.py`，3775 行 |

## 2. 真实入口

1. **跨模块 Python API（唯一外部消费点）**：
   - `from experiment_designer.agent import advise`
   - `from experiment_designer.models import AdvisorContext, ArtifactRef`
   - 消费者：仅 `ResAgent/src/resagent/adapters/expagent/adapter.py`（`advise_adhoc`、`_build_advisor_context`、`_call_advise`）。
   - `CodingAgent`、`reproagent` 对 `experiment_designer` **零引用**（全仓 grep 证实）。
2. **CLI**：`main.py::main`（子命令 `advise` + 旧 `--idea`/`--no-interactive`/REPL 模式）。
3. **REPL**：`repl.py::run_repl`（仅由 `main.py` 无子命令分支进入）。

## 3. 主流程调用图

```
ResAgent.adapter.advise()
  └─ experiment_designer.agent.advise(ctx, model, ..., run_dir)
       └─ ContextPolicy.for_model(model)
       └─ controller.loop._run_loop(ctx, ...)
            ├─ prompts.rendering.build_initial_prompt   (situation + artifacts + existing_plan)
            ├─ prompts.rendering.build_turn_prompt       (LoopState → 每轮重建 prompt)
            ├─ llm.call_llm                             (FC；mock 或 OpenAI 兼容 API)
            ├─ 工具分派:
            │    search_papers → tools.papers.search_papers
            │    read_file     → tools.files.read_file
            │    note_finding  → LoopState.findings
            │    save_paper    → tools.papers.save_paper → LoopState.paper_index
            │    finish        → ScientificDecision + validator.validate_decision
            │                     ├─ session.write_state       (state.json)
            │                     ├─ report.write_decision     (scientific_decision.json)
            │                     ├─ session.write_session_card (session.yaml)
            │                     └─ 返回 (decision, trace)
            └─ (预算耗尽) → 失败 decision + write_state + write_session_card(failed)
```

`plan()` / `revise()`（`controller/planner.py`）是第二入口，仅被 CLI/REPL 使用，内部转调 `advise()` 后把 `recommended_actions` 反投影为 `ExperimentPlan.tasks`。

## 4. 文件职责表

| 文件 | 唯一职责 | 备注 |
|---|---|---|
| `agent.py` | 顶层 `advise()`：解析 run_dir + context policy，委托 loop | 跨模块契约入口 |
| `models.py` | 全部 Pydantic 模型（输入/输出/持久化 + V2 action union） | 见 §6 问题 E1/E2 |
| `controller/loop.py` | FC agentic loop + 工具分派 + 落盘 | 核心主线 |
| `controller/planner.py` | `plan()`/`revise()` 旧设计流程包装 + `_extract_*_tasks` | 第二主线，见 §6 问题 E1 |
| `controller/validator.py` | `validate()`（旧计划）+ `validate_decision()`（V2）确定性校验 | 见 §6 问题 E2 |
| `prompts/system.py` | SYSTEM_PROMPT | 字节级冻结（Phase 0） |
| `prompts/schemas.py` | FC 工具 JSON schema（含 `experiment_plan` 嵌套） | 见 §6 问题 E2 |
| `prompts/rendering.py` | build_*_prompt 状态渲染 | |
| `context/builder.py` | LoopState | |
| `context/policy.py` | ContextPolicy.for_model() | |
| `llm.py` | OpenAI 兼容客户端（3 重试）+ mock | |
| `session.py` | session.yaml + state.json + `list_run_files` | 见 §6 问题 D2 |
| `report.py` | experiment_plan.yaml / scientific_decision.json / validation_report.md | |
| `presentation.py` | 终端渲染原语 | 仅 REPL 使用 |
| `repl.py` | 交互式 REPL | 仅 CLI 使用 |
| `main.py` | CLI 参数解析 + 装配 | |
| `config.py` | LLM 配置解析 | 见 §6 问题 R1 |
| `tools/files.py` | read_file | |
| `tools/papers.py` | search_papers / save_paper（S2/DBLP/arXiv） | |

## 5. 状态读写与副作用

- **读**：`ctx.situation`/`ctx.artifacts`/`ctx.existing_plan`（调用方传入）；`thread_dir/thread.yaml`（续接）；`read_file` 读外部文件；外部 API（S2/DBLP/arXiv/LLM）。
- **写**（run_dir 内）：`state.json`、`scientific_decision.json`、`session.yaml`、`papers/*.md`、`logs/*.txt`（trace）；`thread.yaml`（append）。
- **副作用**：网络请求（LLM、文献检索、arXiv PDF 下载）。无文件系统外副作用，无 Task/workspace/环境管理（符合职责边界）。

## 6. 问题清单（含文件/行号证据）

### 6.1 死代码 / 无调用者（Dead / Legacy）

**D1 — `_extract_yaml`（`controller/planner.py:162-182`）**
- 现状：私有函数，docstring 自称 "backward-compat re-export"（实为非 re-export 的遗留 helper）。
- 真实调用路径：**无生产调用者**；仅 `tests/test_planner.py:349,357,363,369,373,375` 直接调用。全四仓 grep 仅命中定义处与测试。
- 为什么是问题：FC 结构化输出落地后，YAML 提取路径已无消费者；docstring 措辞具有误导性。
- 是否改变行为：否。
- 处理：**删除** + 删除 `tests/test_planner.py` 中 `TestPlannerExtractYaml` 三个用例。
- 分类：安全整理。

**D2 — `list_run_files`（`session.py:71-83`）**
- 现状：公共函数，docstring 称 "useful for orchestrators (ResAgent)"。
- 真实调用路径：**无任何调用者**。ExpAgent src/tests 未引用；ResAgent adapter 仅 `import ... advise / AdvisorContext, ArtifactRef`，未 import 本函数；未在 `__init__.py.__all__` 导出。
- 为什么是问题：属 "为未来扩展提前增加但没有实际使用" 的抽象（计划 §1 点名此类）。
- 是否改变行为：否。
- 处理：**删除**。
- 分类：安全整理。

### 6.2 双主线 / 所有权（Split Mainline / Ownership）——后续独立迁移

**E1 — 旧 `ExperimentPlan`/`TaskBundle`/`plan()`/`revise()` 与 V2 `recommended_actions` 双表示并存**
- 现状：`ScientificDecision` 同时携带 `experiment_plan`（ExperimentPlan：goal/matrix/metrics/tasks/analysis_plan/risks）与 `recommended_actions`（V2 科学动作图）。`planner.plan()/revise()` 又把 `recommended_actions` 反投影成 `TaskBundle`（`_extract_coding/repro/run_tasks`，`planner.py:79-125`）。
- 真实调用路径：`plan()/revise()` 仅被 ExpAgent 自己的 `main.py`（`--no-interactive`）与 `repl.py` 使用；**跨模块无消费者**（ResAgent 直接消费 `advise()` 的 `ScientificDecision`，不碰 ExperimentPlan）。
- 为什么是问题：同一"下一步做什么"存在两套模型 + 一套派生视图；`analyze_results/search_literature/ask_user` 三类 capability 在 TaskBundle 反投影中被丢弃；`CodingTask.workspace_path` 字段（`models.py:105`）与 V2 "不输出物理路径" 原则相悖。
- 是否改变行为：若删除会改变 CLI/REPL 行为。
- 处理：**后续独立迁移**（设计见 `docs/E1_MAINLINE_CONVERGENCE_DESIGN.md`），不在本分支实现。这不是本轮待删除的死代码：旧设计入口仍被 CLI/REPL 使用，删除前须先完成 CLI/REPL 直连 `advise()` 的迁移与契约测试。
- 分类：行为风险（独立迁移任务）。

**E2 — action graph 的 schema / 模型 / 校验规则三处重复表达**
- 现状：六类 capability 词表同时出现在 (1) `models.py` 的 discriminated union `ScientificAction`；(2) `prompts/schemas.py::_SCIENTIFIC_ACTION_SCHEMA`（FC JSON schema，扁平对象）；(3) `validator.py:289-311` 的 capability 专属字段校验。`validate()`（旧计划校验，`validator.py:12-117`）与 `validate_decision()`（V2，`validator.py:261-337`）是两套校验器。
- 为什么是问题：词表与规则分散，易漂移。
- 是否改变行为：合并需改动公共模型/校验，有行为风险。
- 处理：**仅报告**（计划 §8 冻结 V2 contract，不得自行改变）。
- 分类：行为风险。

### 6.3 冗余（Redundancy）

**R1 — `config.py::_parse_llm_section`（`config.py:50-81`）手写 YAML 解析器**
- 现状：约 32 行手写行解析，而 `pyyaml` 已是硬依赖。
- 为什么是问题：重复实现 PyYAML；`llm:` section 解析可用 `yaml.safe_load` 一行替代。
- 是否改变行为：改写为 `yaml.safe_load` 对当前 `config.yaml`（3 键）行为等价，但属"顺手重写"边界。
- 处理：**仅报告**，本轮不改（无直接测试覆盖，改写收益低、有极小行为漂移风险）。
- 分类：内部结构整理（低优先级）。

**R2 — slugify 逻辑重复（`loop.py:309` vs `tools/papers.py:117`）**
- 现状：两处用相同正则 `re.sub(r"[^A-Za-z0-9_.-]+", "_", ...).strip("_")[:80]` 生成 slug。
- 处理：**仅报告**（跨两文件，收益低）。
- 分类：内部结构整理（低优先级）。

### 6.4 文档不一致（Readability，非代码）

**D3 — `docs/ARCHITECTURE_SYNC.md` 内部自相矛盾**
- `§5` "对 CodingAgent 的映射" 表仍列 `workspace_path → workspace_path`（`ARCHITECTURE_SYNC.md:84`），与 `§3/§4` 的 V2 "不输出物理路径" 原则相悖。
- `§7` 仍描述 `advisor.py/planner.py/validator.py` 兼容壳（`ARCHITECTURE_SYNC.md:140-141`）——这些壳已在 V2 收尾删除，目录树已无这些文件。
- 处理：**仅报告**（文档非代码，可后续单独修正）。

## 7. 测试与生产路径对应

| 测试文件 | 覆盖的生产路径 |
|---|---|
| `test_phase0_contract.py` | `__all__` 导出、CLI --help hash、SYSTEM_PROMPT hash、跨模块模型字段列表 |
| `test_execution_contract_v1.py` | V2 action graph 全部不变量（依赖、覆盖、required、物理字段边界、mock/schema） |
| `test_validator.py` | `validate()`（旧计划校验）全部分支 |
| `test_planner.py` | `plan()/revise()` mock 流程、session card、thread、多 tool call、**`_extract_yaml`（死代码）** |
| `test_models.py` | 旧计划模型树（DesignInput/ExperimentPlan/CodingTask/ReproTask/RunTask） |
| `test_e2e.py`（deselected） | 真实 LLM 闭环 |

缺口：`validate_decision()` 的 capability 专属字段校验（`validator.py:289-311`）在 `test_execution_contract_v1.py` 已有覆盖；`config.py` 无直接测试。

## 8. 修改清单（Phase D）

本轮仅执行**安全整理**（计划 §5 "安全整理 → 可直接处理"），每个提交一个主题：

| # | 提交主题 | 内容 | 分类 |
|---|---|---|---|
| 1 | remove unused legacy yaml extractor | 删 `planner.py::_extract_yaml` + `TestPlannerExtractYaml` | 安全整理 |
| 2 | remove unreachable list_run_files helper | 删 `session.py::list_run_files` | 安全整理 |

**本轮不处理**：
- E1（双主线）→ 另立为独立迁移任务，设计见 `docs/E1_MAINLINE_CONVERGENCE_DESIGN.md`。
- E2（三处词表）、R1（手写 YAML 解析）、R2（slugify 重复）、D3（文档）→ 低收益/文档类，暂缓。

## 9. 未处理风险

- E1（双主线）已标记为**后续独立迁移**，不在本分支实现（设计见 `docs/E1_MAINLINE_CONVERGENCE_DESIGN.md`）。迁移会触碰 `ScientificDecision.experiment_plan` 字段与 phase0 冻结契约，需总体审查 + 更新 phase0 lock。
- 跨模块契约（V2 action contract、`AdvisorContext`/`ArtifactRef`、session.yaml schema）本轮未触碰。
- Prompt 未改动（`SYSTEM_PROMPT` hash 不变）。
