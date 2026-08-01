# ExpAgent 开发文档

## 1. 模块定位

ExpAgent 是完整科研 agent 中的“实验设计模块”。

它负责把研究想法、文献现状、已有代码能力、算力预算，转化为结构化实验计划。它不直接写代码、不直接训练模型、不直接复现论文，而是生成可以交给其他模块执行的任务。

核心目标：

```text
research idea -> experiment plan -> coding/reproduction/run tasks
```

它连接：

```text
LiteratureAgent -> ExperimentDesigner -> CodingAgent
                                   -> ReproAgent
                                   -> Runner / AnalysisAgent
```

## 2. 不做什么

MVP 阶段不要做：

```text
不直接写代码
不直接改 repo
不直接跑训练
不自动检索论文
不写论文正文
不做复杂多 agent 黑板系统
```

这些分别交给：

```text
CodingAgent        写代码/改代码
ReproAgent         复现别人方法/baseline
LiteratureAgent    查论文/SOTA/代码
WritingAgent       写论文
AnalysisAgent      分析结果/画图
```

ExpAgent 只负责“设计实验”。

## 3. 输入

MVP 输入：

```yaml
research_idea: "想验证的新方法或研究假设"
target_task: "任务类型，比如 image classification / time series / NLP / GNN"
base_repo: "已有代码仓库路径或 URL，可选"
available_methods:
  - "已有自己的方法"
  - "已有 baseline"
literature_context:
  - "相关论文摘要/SOTA 信息，可选"
compute_budget:
  gpu: "RTX 4090 / A100 / CPU only"
  max_runtime: "2 hours"
  max_trials: 10
constraints:
  - "不能下载大数据集"
  - "先做小规模验证"
```

最小 CLI 可以是：

```bash
experiment-designer plan \
  --idea idea.md \
  --context context.md \
  --output experiment_plan.yaml
```

## 4. 输出

核心输出是 `experiment_plan.yaml`。

推荐结构：

```yaml
version: 1

goal:
  summary: "本轮实验要验证什么"
  hypothesis: "如果方法有效，应该观察到什么现象"
  success_criteria:
    - "metric A 高于 baseline"
    - "runtime 不超过某个范围"

experiment_matrix:
  datasets:
    - name: MNIST
      split: standard
      reason: "轻量验证"
  methods:
    - name: proposed_method
      type: new_method
      implementation_status: needs_code
    - name: baseline_1
      type: baseline
      implementation_status: existing_or_repro
  metrics:
    - accuracy
    - loss
    - runtime

tasks:
  coding_tasks:
    - id: code_001
      goal: "在现有 repo 中实现 proposed_method"
      target_files:
        - "models/"
      constraints:
        - "不改变 baseline 训练流程"
      expected_artifacts:
        - "patch.diff"
        - "verification_report.md"

  repro_tasks:
    - id: repro_001
      paper_url: "..."
      repo_url: "..."
      goal: "复现 baseline 方法的小规模实验"
      expected_artifacts:
        - "result.md"
        - "logs/"

  run_tasks:
    - id: run_001
      command_goal: "运行 proposed_method 的 bounded experiment"
      expected_runtime: "30 minutes"
      requires_gpu: true

analysis_plan:
  comparisons:
    - proposed_method vs baseline_1
  plots:
    - "accuracy curve"
    - "loss curve"
  failure_checks:
    - "是否只是训练更久导致提升"
    - "是否参数量显著增加"

risks:
  - "baseline 代码可能无法复现"
  - "数据集下载可能失败"
  - "当前实验只是 bounded，不等价于完整论文实验"
```

## 5. 工作流

MVP 使用线性 workflow，不做复杂黑板架构。

```text
1. 读取研究想法和上下文
2. 识别实验目标
3. 拆分假设
4. 设计 baseline / ablation / metric / dataset
5. 判断哪些任务需要写代码
6. 判断哪些任务需要调用 ReproAgent
7. 判断哪些任务可以直接运行
8. 生成 experiment_plan.yaml
9. 生成 human_review.md
```

## 6. 和 CodingAgent 的关系

ExpAgent 不调用 CodingAgent 也可以独立运行。

但它输出的 `coding_tasks` 应该能被 CodingAgent 直接消费。

Coding task 格式要明确：

```yaml
id: code_001
repo_path: "/path/to/repo"
task_goal: "实现某个模块"
constraints:
  - "不要改动训练入口"
  - "不要改变 baseline 行为"
verify_commands:
  - "python -m pytest tests/test_model.py"
expected_output:
  - "patch.diff"
  - "report.md"
```

原则：

```text
ExpAgent 负责说清楚要改什么、为什么改、怎么验证。
CodingAgent 负责实际修改代码。
```

## 7. 和 ReproAgent 的关系

ExpAgent 输出 `repro_tasks`，交给 ReproAgent 跑 baseline 或 SOTA。

Repro task 格式：

```yaml
id: repro_001
paper_url: "..."
repo_url: "..."
experiment_goal: "复现该方法在某数据集上的 bounded 结果"
compute_budget:
  max_runtime: "1 hour"
  gpu_required: true
expected_metrics:
  - accuracy
  - loss
  - runtime
```

原则：

```text
ExpAgent 负责决定该复现谁。
ReproAgent 负责尝试把别人代码跑起来。
```

## 8. 内部文件结构

建议项目结构：

```text
ExpAgent/
  README.md
  DEVELOPMENT_PLAN.md
  pyproject.toml
  src/experiment_designer/
    __init__.py
    main.py          # CLI
    models.py        # Pydantic 数据结构
    llm.py           # LLM 调用
    planner.py       # 实验计划生成
    validator.py     # plan validation
    report.py        # 写 yaml/md
    prompts.py       # prompt 模板
  tests/
    test_models.py
    test_planner.py
    test_validator.py
```

## 9. 核心模型

建议先定义这些 Pydantic model：

```text
DesignInput
ResearchGoal
ExperimentPlan
ExperimentMatrix
CodingTask
ReproTask
RunTask
AnalysisPlan
Risk
```

不要一开始设计太多抽象类。先把输入输出稳定下来。

## 10. Plan Validation

必须有一个轻量 validation，防止 LLM 生成空泛计划。

检查项：

```text
是否有明确 hypothesis
是否有 baseline
是否有 metric
是否有 dataset 或数据来源
是否区分 proposed / baseline / ablation
是否说明算力预算
是否把写代码任务和复现任务拆开
是否标注 bounded/full experiment
是否有成功标准
是否有风险和失败处理
```

如果不满足，输出：

```yaml
status: needs_revision
issues:
  - "缺少 baseline"
  - "没有明确 metric"
```

## 11. MVP 成功标准

MVP 不要求真的跑实验。

只要做到：

```text
输入一个 research idea
输出一个结构化 experiment_plan.yaml
能拆出 coding_tasks / repro_tasks / run_tasks
能解释实验设计理由
能指出风险和缺口
有基本 tests
```

就算第一版成功。

## 12. 推荐开发顺序

第一阶段：

```text
1. 建 repo 和 pyproject
2. 定义 models.py
3. 做 mock LLM planner
4. 输出 experiment_plan.yaml 和 human_review.md
5. 写 validation
6. 加 CLI
```

第二阶段：

```text
1. 接真实 LLM
2. 加 OpenAI-compatible API
3. 增加 prompt 模板
4. 用几个真实 idea 测试
```

第三阶段：

```text
1. 对接 CodingAgent task format
2. 对接 ReproAgent task format
3. 让 Orchestrator 调用它
```

## 13. 关键原则

```text
Artifact first.
```

每一步都要产出可保存、可审计、可传给其他模块的文件。

不要只返回一段自然语言。

核心产物应该是：

```text
experiment_plan.yaml
human_review.md
validation_report.md
```

## 14. 未来集成位置

在完整科研 agent 中，ExpAgent 应该被 Orchestrator 调用，而不是直接控制全局流程。

典型上游输入：

```text
LiteratureAgent 的论文地图
IdeaAgent 的研究假设
用户给定的研究想法
已有 repo / 方法描述
算力预算
```

典型下游输出：

```text
给 CodingAgent 的 coding_tasks
给 ReproAgent 的 repro_tasks
给 Runner 的 run_tasks
给 AnalysisAgent 的 analysis_plan
```

第一版只需要把这些任务结构化写入文件，不需要真的调用下游模块。

