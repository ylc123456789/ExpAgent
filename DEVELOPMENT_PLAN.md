# ExpAgent 开发文档

## 1. 模块定位

ExpAgent 是完整科研 agent 系统中的"科学顾问（Scientific Advisor）"。

**一句话**: 输入研究现状 → agentic loop（可搜索论文、读文件）→ 输出 ScientificDecision（含完整实验方案和建议动作）。

```
ResAgent / 用户
    │  situation + artifacts
    ▼
ExpAgent (agentic loop)
    ├── search_papers  (Semantic Scholar / DBLP / arXiv)
    ├── read_file      (读实验结果、日志)
    ├── save_paper     (保存论文 metadata 到磁盘)
    └── finish         (输出 ScientificDecision)
    │
    ▼
ScientificDecision
    ├── conclusion + evidence
    ├── experiment_plan (可选)
    └── recommended_actions (含完整可执行计划)
         ├── repro_task  → ResAgent → ReproAgent
         ├── coding_task → ResAgent → CodingAgent
         └── run_task    → ResAgent → Runner
```

在系统中的位置：

```
ResAgent (Orchestrator)
    ├── ExpAgent      ← 本模块（科学顾问）
    ├── CodingAgent   ← 写代码
    └── ReproAgent    ← 复现论文
```

## 2. 不做什么

- 不调用 CodingAgent / ReproAgent
- 不执行训练/评测命令
- 不维护全局 task queue
- 不管理项目预算

## 3. 输入

```python
AdvisorContext(
    situation="自然语言描述当前研究情况...",
    artifacts=[ArtifactRef(id="...", type="run_log", summary="...")],
    existing_plan=ExperimentPlan | None,
)
```

## 4. 输出

核心输出是 `ScientificDecision`（`scientific_decision.yaml`）：

```yaml
summary: "一句话总结"
confidence: high | medium | low

conclusion:
  status: supported | not_supported | inconclusive | needs_more_experiments
  rationale: "科学推理过程"

evidence:
  - source: artifact | literature | reasoning
    description: "证据描述"

experiment_plan:  # 可选，design/revise 场景时包含
  goal: {...}
  experiment_matrix: {...}
  tasks: {...}

recommended_actions:  # 核心：交给 ResAgent 的建议
  - priority: high | medium | low
    type: repro_task | coding_task | run_task | literature_search | ask_user
    rationale: "为什么建议这个"
    plan:  # 完整可执行计划，下游可直接使用
      kind: repro_task
      paper_url: "..."
      repo_url: "..."
      experiment_goal: "..."
      code_availability: public | upon_request | none

risks: [...]
needs_user_input: [...]
```

## 5. 工作流 (Agentic Loop)

```
1. 构建初始 prompt (situation + artifacts)
2. Loop (max 20 steps + 6 grace):
   a. 每轮从 LoopState 重建 user prompt (CodingAgent 风格)
   b. FC 调用 LLM，返回 tool_call
   c. 执行工具 (search_papers / read_file / save_paper)
   d. 结果注入 state，记录压缩历史
   e. 保留最近 4 对 tool_pairs 供 FC 连续性
   f. finish → 解析 ScientificDecision → 校验 → 返回
3. API 层 3 次重试 (网络/5xx)，loop 层捕获并作为 api_error 步骤继续
```

## 6. 架构

```
src/experiment_designer/
  context.py           LoopState (结构化状态)
  context_policy.py    ContextPolicy (模型窗口自适应，3档)
  prompts.py           SYSTEM_PROMPT + TOOLS + build_turn_prompt + build_initial_prompt
  advisor.py           agentic loop (advise / _run_loop)
  llm.py               FC API 调用 (retry 3x，对齐 CodingAgent/ReproAgent)
  tools.py             search_papers, read_file, save_paper
  models.py            Pydantic 模型 (DesignInput, ScientificDecision, 等)
  planner.py           plan() / revise() wrapper
  validator.py         validate() / validate_decision()
  report.py            write_plan / write_decision
  config.py            LLM 配置解析
  main.py              CLI (advise + REPL)
```

## 7. 论文管理 (Index + On-Demand Read)

```
上下文常驻 (PaperIndex)            磁盘 (papers/)
─────────────────────────         ────────────────
[SENet, 2018 · CVPR]       →    papers/senet.md
  L2-norm 通道注意力基线             (完整 metadata)
[CBAM, 2018 · ECCV]        →    papers/cbam.md
  空间+通道注意力基线

LLM 保存论文后可通过 read_file 按需读取全文。
```

## 8. 上下文策略

`ContextPolicy.for_model(model)` 根据模型窗口自适应：

| 窗口 | step_history | paper_index | file_cache |
|------|-------------|-------------|------------|
| >= 500K | 20 | 15 | 12 × 8000 |
| >= 128K | 10 | 8  | 6 × 4000  |
| < 128K  | 4  | 8  | 3 × 2000  |

## 9. 产物组织 (对齐 ReproAgent)

```
runs/<timestamp>/
├── scientific_decision.yaml
├── experiment_plan.yaml
├── validation_report.md
├── papers/              # 本 run 下载的论文
└── logs/                # trace 文件
```

## 10. Python API

```python
from experiment_designer.advisor import advise
from experiment_designer.models import AdvisorContext

ctx = AdvisorContext(situation="描述当前研究情况...")
decision, trace = advise(ctx)
# → decision.recommended_actions[0].plan.paper_url
```

## 11. CLI

```bash
expagent                                    # REPL 交互模式
expagent --idea idea.md --no-interactive    # 直接生成
expagent advise --context "..."             # ResAgent 调用
```
