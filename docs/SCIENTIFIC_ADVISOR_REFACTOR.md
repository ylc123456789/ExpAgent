# ExpAgent Scientific Advisor Refactor (HISTORICAL — 2026-08-03)

> ⚠️ This document was the original design proposal before implementation.
> The final implementation differs significantly. See DEVELOPMENT_PLAN.md
> and ARCHITECTURE_SYNC.md for current architecture.

## 1. 背景

当前大系统逐渐拆成几个模块：

```text
ResAgent     项目经理 / 顶层 orchestrator
ExpAgent     科学顾问 / 实验科学家
CodingAgent  程序员 / repo-local coding agent
ReproAgent   复现实验工程师 / paper-repo reproduction agent
```

现在需要优化 ExpAgent 的定位。它不应该只是一次性的：

```text
idea -> experiment_plan.yaml
```

更合理的定位是：

```text
ExpAgent = scientific advisor for experiment reasoning
```

也就是围绕研究想法、实验设计、实验结果解释、实验方案修订、论文/方法对比，提供科学判断和结构化建议。

## 2. 核心边界

一句话边界：

```text
ExpAgent 负责“科学上怎么想”。
ResAgent 负责“系统上怎么做”。
```

ExpAgent 负责：

```text
设计初始实验方案
审查实验方案是否合理
分析实验结果是否支持 hypothesis
根据失败或新结果修订实验设计
判断 baseline / ablation / metric 是否公平和充分
必要时检索论文来支持科学判断
输出 recommended_actions 给 ResAgent
```

ExpAgent 不负责：

```text
不调用 CodingAgent
不调用 ReproAgent
不执行训练/评测命令
不维护全局 task queue
不管理项目预算
不决定全局终止
不写全局 research_state
不直接调度多个 agent
```

## 3. ResAgent 与 ExpAgent 的关系

ResAgent 是顶层 controller，可以是 agentic loop。

ResAgent 的职责：

```text
维护全局 research_state
调用 ExpAgent / CodingAgent / ReproAgent
登记 artifacts
判断任务失败类型
控制重试
管理预算和人工确认
决定下一步执行哪个模块
```

ExpAgent 的职责：

```text
基于 idea / context / literature / experiment results 做科学判断
返回结构化 scientific decision
```

典型流程：

```text
用户输入 idea
  -> ResAgent 调 ExpAgent: design_initial_plan
  -> ExpAgent 输出 experiment_plan + recommended_actions
  -> ResAgent 调 CodingAgent / ReproAgent 执行任务
  -> ReproAgent 或 CodingAgent 返回结果/失败
  -> ResAgent 判断执行失败类型
      -> 网络/临时下载失败: ResAgent 自己重试
      -> 科学目标不合理/结果需要解释: ResAgent 调 ExpAgent analyze/revise
  -> ExpAgent 输出结果解释和下一步科学建议
  -> ResAgent 决定是否执行建议
```

## 4. 新的 ExpAgent modes

建议把 ExpAgent 从单一 `plan` 能力扩展成多个 mode。

### 4.1 design_initial_plan

输入：

```text
research_idea
target_task
existing_assets
compute_budget
constraints
optional literature_context
```

输出：

```text
experiment_plan.yaml
scientific_decision.yaml
```

用途：从 idea 生成初始实验方案。

### 4.2 review_plan

输入：

```text
experiment_plan.yaml
research_idea
constraints
```

输出：

```text
plan_review.yaml
```

用途：审查实验方案是否合理，是否缺 baseline、metric、dataset、ablation、公平性控制。

### 4.3 analyze_results

输入：

```text
hypothesis
experiment_plan.yaml
result artifacts
logs / metrics / summaries
```

输出：

```text
result_analysis.yaml
```

用途：判断实验结果是否支持 idea，以及是否需要更多实验。

### 4.4 revise_plan

输入：

```text
previous experiment_plan.yaml
result_analysis.yaml
failure reports
new constraints
```

输出：

```text
revised_experiment_plan.yaml
scientific_decision.yaml
```

用途：根据结果、失败、预算变化修改实验方案。

### 4.5 diagnose_failure

输入：

```text
failed task summary
logs
experiment context
```

输出：

```text
failure_diagnosis.yaml
```

用途：区分科学失败、实现失败、环境失败、数据失败、预算不足。

注意：ExpAgent 可以判断“这看起来像科学/实验设计问题”，但是否重试、是否调用 CodingAgent/ReproAgent，由 ResAgent 决定。

### 4.6 compare_methods

输入：

```text
proposed method description
baseline descriptions
metrics
experimental settings
```

输出：

```text
method_comparison_review.yaml
```

用途：判断对比实验是否公平、是否缺少关键 baseline、是否可能存在配置不一致。

## 5. 文献检索作为 ExpAgent 工具

建议把论文检索能力放进 ExpAgent 内部，作为 scientific reasoning 的工具，而不是顶层独立 agent 的第一阶段。

原因：论文检索通常服务于科学判断：

```text
设计实验时查 SOTA / baseline / dataset / metric
分析失败时查相关方法是否遇到类似问题
修订方案时查替代 baseline 或诊断实验
扩展 idea 时查最近相关工作
```

因此 ExpAgent 可以有内部 tools：

```text
search_papers
read_paper_metadata
find_code_repo
extract_baselines
extract_datasets_and_metrics
summarize_related_work
```

MVP 阶段不一定真的接外部检索 API。可以先定义接口和 mock/fallback。

重要边界：

```text
ExpAgent 可以调用 literature tools。
ExpAgent 不能调用 CodingAgent / ReproAgent。
ResAgent 调用 ExpAgent，而不是直接调用 literature tools。
```

## 6. 统一输出：ScientificDecision

为了让 ResAgent 容易消费，建议新增统一输出模型 `ScientificDecision`。

示例：

```yaml
version: 1
mode: analyze_results
summary: "当前 bounded 实验暂时不支持 hypothesis"
confidence: medium

conclusion:
  status: not_supported
  rationale: "proposed method test accuracy 低于 baseline，且 loss curve 不稳定"

evidence:
  - artifact_id: repro_result_001
    claim: "baseline accuracy = 83.7%"
  - artifact_id: run_result_001
    claim: "proposed accuracy = 82.1%"

recommended_actions:
  - id: action_001
    type: revise_experiment
    priority: high
    rationale: "需要排除训练预算不足导致的假阴性"
    suggested_task:
      kind: run_task
      command_goal: "Run proposed method for longer budget with same baseline config"

  - id: action_002
    type: coding_task
    priority: medium
    rationale: "需要记录 attention map 诊断模型是否学到预期模式"
    suggested_task:
      kind: coding_task
      task_goal: "Add attention map logging without changing training semantics"

risks:
  - "当前实验是 bounded run，不能替代 full reproduction"
  - "baseline 与 proposed method 的参数量可能不一致"

needs_user_input:
  - "是否允许增加 2 小时 GPU 预算跑更长训练？"
```

## 7. RecommendedAction 类型

建议先定义有限 action 类型，但它们只是建议，不是执行指令。

```text
revise_experiment
coding_task
repro_task
run_task
literature_search
ask_user
stop_direction
continue_direction
```

解释：

```text
coding_task      建议 ResAgent 调 CodingAgent
repro_task       建议 ResAgent 调 ReproAgent
run_task         建议 ResAgent 调 Runner 或未来 RunAgent
literature_search 建议 ExpAgent 自己已经做过或下一轮需要查文献
ask_user         建议 ResAgent 问用户
stop_direction   科学上建议停止当前方向
continue_direction 科学上建议继续当前方向
```

ResAgent 可以接受、拒绝、排序、延迟这些建议。

## 8. 输入中的 Artifact 引用

ExpAgent 不应该直接扫描整个 runs 目录。ResAgent 应该把相关 artifact 摘要传给 ExpAgent。

建议输入格式：

```yaml
artifacts:
  - id: repro_result_001
    type: repro_result
    path: /path/to/result.md
    summary: "torchdiffeq MNIST baseline completed, test acc 99.02% after 5 epochs"
  - id: patch_001
    type: code_patch
    path: /path/to/diff.patch
    summary: "Added loss logging to odenet_mnist.py"
```

ExpAgent 可以读取 path 指向的文本文件，但不应自行决定全局 artifact registry。

## 9. 建议的数据模型

在 `models.py` 中可以新增：

```text
AdvisorMode
ArtifactRef
ScientificDecision
ScientificConclusion
EvidenceItem
RecommendedAction
ResultAnalysis
PlanReview
FailureDiagnosis
```

不要删除现有 `ExperimentPlan`，它应该继续作为 `design_initial_plan` 和 `revise_plan` 的核心产物。

## 10. CLI 建议

保留现有 `plan`/REPL 能力，但增加显式 mode CLI。

示例：

```bash
expagent advise   --mode design_initial_plan   --input design_input.yaml   --output runs/idea-001
```

```bash
expagent advise   --mode analyze_results   --input analysis_input.yaml   --output runs/analyze-001
```

输出目录建议：

```text
runs/<run-id>/
  experiment_plan.yaml          # 如果 mode 产生计划
  scientific_decision.yaml      # 总是产生
  validation_report.md
  llm_request.json              # 可选
  llm_response.txt              # 可选
```

## 11. Validation 建议

不同 mode 使用不同 validation。

`design_initial_plan` 必须检查：

```text
hypothesis 非空
至少一个 dataset
至少一个 metric
至少一个 baseline 或解释为什么暂时没有 baseline
至少一个 risk
任务能映射到 coding/repro/run 中的一类
```

`analyze_results` 必须检查：

```text
conclusion.status 非空
evidence 非空
每个结论至少有一个 artifact 或 metric 支撑
recommended_actions 可以为空，但如果为空必须解释为什么
```

`revise_plan` 必须检查：

```text
说明相对于上一版改了什么
说明为什么修改
保留或重新定义 success criteria
```

`diagnose_failure` 必须检查：

```text
failure_type 属于有限集合
区分 transient/system/scientific/code/data/budget
给出 ResAgent 可消费的建议
```

## 12. MVP 改造顺序

建议按这个顺序改：

```text
1. 改文档定位：ExpAgent = Scientific Advisor
2. 在 models.py 增加 ScientificDecision / RecommendedAction / ArtifactRef
3. 保留现有 plan 功能，映射为 mode=design_initial_plan
4. 增加 analyze_results 的 mock 实现和测试
5. 增加 revise_plan 的 mock 实现和测试
6. 增加 advise CLI，但先不删旧 CLI
7. 给 ResAgent 预留稳定 Python API：advise(input) -> outputs
8. 后续再接 literature tools
```

第一轮不要急着实现真实论文检索，先把 mode、模型、产物结构稳定下来。

## 13. 成功标准

本次 ExpAgent 优化完成后，应做到：

```text
仍然可以从 idea 生成 experiment_plan.yaml
可以读取已有 result/artifact summary 并输出 result_analysis/scientific_decision
可以建议 coding_task / repro_task / run_task，但不执行
可以被未来 ResAgent 当作纯 advisory module 调用
测试覆盖各 mode 的 model validation 和 mock planner
```

## 14. 最重要的设计原则

```text
ExpAgent gives scientific recommendations.
ResAgent makes orchestration decisions.
```

如果一个功能涉及“是否调用某个 agent、是否重试、是否并行、是否结束项目、是否消耗预算”，它属于 ResAgent。

如果一个功能涉及“这个实验是否科学、结果说明什么、下一步实验应该验证什么、需要查什么论文”，它属于 ExpAgent。
