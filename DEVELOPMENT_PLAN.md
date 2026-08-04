# ExpAgent 开发文档 (v2)

## 1. 模块定位

ExpAgent 是完整科研 agent 中的"实验设计模块"，**以 LLM 为核心引擎**，确定性规则只做最基础的安全网。

```
research idea → LLM 推理 → experiment_plan.yaml → 交给下游 agent 执行
                                          │
                                          └── coding_tasks  (→ CodingAgent)
                                          └── repro_tasks   (→ ReproAgent)
                                          └── run_tasks     (→ Runner)
                                          └── analysis_plan (→ AnalysisAgent)
```

在系统中的位置：

```
LiteratureAgent / IdeaAgent / 用户想法
              ↓
         ExpAgent (纯设计，不调度)
              ↓
    experiment_plan.yaml (静态文件)
              ↓
      Orchestrator (未来模块，负责调度执行)
         ├── → CodingAgent
         ├── → ReproAgent (可并行多个)
         └── → Runner
```

ExpAgent **不负责调度执行**。它输出静态的 `experiment_plan.yaml`，由未来的 Orchestrator 决定如何并行调度。这样 ExpAgent 保持简单、可测试，不关心并发和运维问题。

## 2. 不做什么

- 不直接写代码 (CodingAgent)
- 不直接改 repo (CodingAgent)
- 不直接跑训练 (Runner)
- 不自动检索论文 (LiteratureAgent)
- 不写论文正文 (WritingAgent)
- **不调度/编排下游 agent (Orchestrator)**

## 3. 输入

```yaml
research_idea: "想验证的新方法或研究假设"  # 必填
target_task: "任务类型, e.g. image classification / time series"  # 必填
compute_budget:
  gpu: "RTX 4090 / A100 / CPU only"
  max_runtime: "2 hours"
  max_trials: 10
constraints:
  - "不能下载大数据集"
  - "先做小规模验证"
literature_context:  # 可选，来自 LiteratureAgent
  - "相关论文摘要/SOTA 信息"
existing_assets:  # 可选，用户已有的代码/数据/baseline
  implemented_methods:
    - name: "my_attention_variant"
      location: "/path/to/repo/models/attention.py"
  available_datasets:
    - "CIFAR-10 (已下载)"
  known_baselines:
    - "ResNet-50 在 ImageNet 上的结果 (已知但未实现)"
```

### 关于 existing_assets

`available_methods` 不再是输入。ExpAgent 应该自己推理需要对比哪些 baseline、设计哪些 ablation。
用户只需要通过 `existing_assets` 告诉 ExpAgent 已经有什么，避免重复工作。
LLM 负责补全"还缺什么"。

## 4. 输出

核心输出是 **`experiment_plan.yaml`**，机器和人都可读，不再需要单独的 `human_review.md`。
所有设计理由内嵌在 YAML 各字段的 `rationale` 中。

```yaml
version: 1

goal:
  summary: "本轮实验要验证什么"
  hypothesis: "如果方法有效，应该观察到什么现象"
  success_criteria:
    - "metric A 高于 baseline X 至少 Y%"
    - "runtime 不超过 Z"

experiment_matrix:
  datasets:
    - name: MNIST
      split: standard
      rationale: "轻量验证，快速迭代"
  methods:
    - name: proposed_method
      type: new_method
      implementation_status: needs_code
      rationale: "本次要验证的核心方法"
    - name: resnet18_baseline
      type: baseline
      implementation_status: needs_repro
      rationale: "最通用的图像分类基线，几乎所有论文都报告"
  metrics:
    - name: accuracy
      rationale: "主要评估指标"
    - name: loss
    - name: runtime

tasks:
  coding_tasks:
    - id: code_001
      repo_path: "/path/to/repo"
      task_goal: "在 models/ 中实现 proposed_method"
      constraints:
        - "不改变 baseline 训练流程"
      verify_commands:
        - "python -m pytest tests/test_model.py"
      expected_artifacts:
        - "patch.diff"
        - "verification_report.md"

  repro_tasks:
    - id: repro_001
      paper_url: "..."
      repo_url: "..."
      experiment_goal: "复现 baseline 方法的小规模实验"
      compute_budget:
        max_runtime: "1 hour"
        gpu_required: true

  run_tasks:
    - id: run_001
      command_goal: "运行 proposed_method 的 bounded experiment"
      expected_runtime: "30 minutes"
      requires_gpu: true

analysis_plan:
  comparisons:
    - proposed_method vs resnet18_baseline
  plots:
    - "accuracy curve"
    - "loss curve"
  failure_checks:
    - "是否只是训练更久导致提升"
    - "是否参数量显著增加"

risks:
  - description: "baseline 代码可能无法复现"
    mitigation: "准备多个候选 baseline"
  - description: "当前实验只是 bounded，不等价于完整论文实验"
```

## 5. 工作流 (LLM-driven)

```
1. 读取 research_idea + context
2. LLM 分析 idea → 识别假设、推断所需 baseline/ablation/dataset
3. LLM 生成 experiment_plan 结构化输出 (YAML)
4. Pydantic 解析 + 兜底校验
5. Validator 确定性检查 (hypothesis/baseline/metric/dataset/risk 必须齐全)
6. 若 validation 不通过 → 把 issues 返回给 LLM 做 revision
7. 写入 experiment_plan.yaml
```

## 6. 架构原则

**LLM-first**：核心推理由 LLM 完成。代码层只做：
- 组装 prompt
- 调用 LLM
- 解析/校验 LLM 输出
- 写文件

**确定性规则只做安全网**：
- 检查 LLM 输出是否可以解析为合法的 ExperimentPlan
- 检查关键字段是否缺失 (hypothesis/baseline/metric/dataset/risk)
- 检查 coding_tasks 和 repro_tasks 的格式是否可被下游消费

**Artifact first**：每一步产出可保存、可审计的文件。

**与下游解耦**：ExpAgent 通过配置存储 reproagent_path 和 codingagent_path，采用 CLI arg → 环境变量 → config 文件的三级优先级（与 ReproAgent → CodingAgent 的对接方式一致）。MVP 阶段只验证路径存在，不实际调用。

## 7. 和 CodingAgent / ReproAgent 的关系

### 对接方式

照搬 ReproAgent 的 `integrations/codingagent.py` 模式：

```python
# ExpAgent config (config.yaml)
agents:
  reproagent_path: /home/cyl/reproagent
  codingagent_path: /home/cyl/CodingAgent
```

路径解析优先级：CLI arg > 环境变量 > config 文件。

### Task 格式对齐

ExpAgent 的 `coding_tasks` 直接映射到 `CodeTaskSpec` 字段：

| ExpAgent field | CodeTaskSpec field |
|---|---|
| `repo_path` | `repo_path` |
| `task_goal` | `task_goal` |
| `constraints` | `constraints` |
| `verify_commands` | `verify_commands` |

ExpAgent 的 `repro_tasks` 直接映射到 `ReproTask` 字段：

| ExpAgent field | ReproTask field |
|---|---|
| `paper_url` | `paper_url` |
| `repo_url` | `repo_url` |
| `experiment_goal` | `experiment_goal` |

### MVP 阶段

ExpAgent **不实际调用** CodingAgent / ReproAgent。
它只做：
1. 验证配置的 agent 路径存在且合法
2. 输出的 task 格式对齐下游 schema
3. 未来由 Orchestrator 读取 experiment_plan.yaml 并调度执行

**ExpAgent 不能修改 CodingAgent / ReproAgent 的代码**。如果需要下游做改动，由本文档记录需求，在对应 agent 的会话中处理。

## 8. 项目结构

```
ExpAgent/
  README.md
  DEVELOPMENT_PLAN.md
  pyproject.toml
  config.yaml                    # agent 路径配置
  src/experiment_designer/
    __init__.py
    main.py                      # CLI
    models.py                    # Pydantic 数据结构
    llm.py                       # LLM 调用 (urllib, 与 ReproAgent 风格一致)
    planner.py                   # LLM 驱动的实验计划生成
    prompts.py                   # system prompt + user prompt 模板
    validator.py                 # 确定性兜底校验
    report.py                    # 写 experiment_plan.yaml
    config.py                    # agent 路径解析 (CLI > env > config)
  tests/
    test_models.py
    test_planner.py
    test_validator.py
```

## 9. 核心数据模型

```python
# 输入
DesignInput         # 研究想法 + 上下文 + 预算 + 约束
ExistingAssets      # 用户已有的代码/数据/baseline

# 输出
ExperimentPlan      # 顶层结构
ResearchGoal        # goal 部分
ExperimentMatrix    # datasets / methods / metrics
CodingTask          # 对齐 CodeTaskSpec
ReproTask           # 对齐 ReproTask
RunTask             # 运行任务
AnalysisPlan        # 分析计划
Risk                # 风险 + 缓解措施

# 校验
ValidationResult    # status + issues list
```

## 10. Plan Validation (确定性安全网)

必须检查：

- 是否有明确 hypothesis
- 是否有 baseline（至少一个）
- 是否有 metric（至少一个）
- 是否有 dataset 或数据来源
- 是否区分 proposed / baseline / ablation
- 是否说明算力预算
- 是否把 coding_tasks 和 repro_tasks 拆开
- 是否有成功标准 (success_criteria)
- 是否有风险 + 缓解措施

不通过时返回 `ValidationResult(status="needs_revision", issues=[...])`，由 planner 将 issues 反馈给 LLM 重新生成。

## 11. MVP 成功标准

- 输入 research_idea → 输出结构化 experiment_plan.yaml
- LLM 驱动，非模板填充
- 能拆出 coding_tasks / repro_tasks / run_tasks
- 每个决策有 rationale
- 能指出风险和缓解措施
- Validator 兜底，防止空泛输出
- 有基本 tests
- 使用 ResAgent 共享虚拟环境

## 12. 推荐开发顺序

### Phase 1: 骨架
1. pyproject.toml + package 结构
2. models.py (所有 Pydantic models)
3. config.py (agent 路径解析)
4. llm.py (OpenAI-compatible client)
5. tests/test_models.py

### Phase 2: 核心
6. prompts.py (system prompt + user prompt)
7. planner.py (LLM 调用 → YAML 解析 → ExperimentPlan)
8. validator.py (确定性兜底)
9. report.py (写 experiment_plan.yaml)
10. tests/test_planner.py + test_validator.py

### Phase 3: CLI + 集成
11. main.py (CLI: experiment-designer plan ...)
12. 用几个真实 idea 测试端到端流程

## 13. 下游解耦需求 (给 ReproAgent / CodingAgent)

本文档记录了 ExpAgent 对下游模块的接口需求，由对应会话处理。

### 对 ReproAgent 的需求
- 支持通过 `--codingagent-path` CLI 参数传递 CodingAgent 路径 (已实现 ✓)
- 支持通过 `CODINGAGENT_PATH` 环境变量 (已实现 ✓)
- 支持通过 config 文件配置 (已实现 ✓)

### 对 CodingAgent 的需求
- 提供稳定的 CLI 或 Python API
- 接受结构化 task 输入
- 返回结构化结果
- 可从 CodingAgent repo 外部调用

详见 `docs/downstream_decoupling_requirements.md`。
