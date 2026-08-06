# ExpAgent Architecture — Downstream Agent Sync

## 1. 定位

ExpAgent = **Scientific Advisor**。不调度任何 agent。

```
ResAgent → 读 scientific_decision.yaml → 调度 CodingAgent / ReproAgent / Runner
ExpAgent → situation + artifacts → agentic loop → ScientificDecision
```

## 2. Python API

```python
from experiment_designer.advisor import advise
from experiment_designer.models import AdvisorContext, ArtifactRef

ctx = AdvisorContext(
    situation="自然语言描述当前研究情况...",
    artifacts=[ArtifactRef(id="result_001", type="run_log", summary="82.1% accuracy")],
    existing_plan=None,
)
decision, trace = advise(ctx, model="deepseek-chat")
```

## 3. 输出格式: ScientificDecision

每个 `recommended_action.plan` 是**自包含**的完整方案，下游可直接使用：

```yaml
recommended_actions:
  - priority: high
    type: repro_task
    rationale: "需要 SE-Net 作为通道注意力 baseline"
    plan:
      kind: repro_task
      paper_url: "https://arxiv.org/abs/1709.01507"
      repo_url: "https://github.com/moskomule/senet.pytorch"
      experiment_goal: "在 CIFAR-10 上复现 SE-ResNet-18 (10 epochs)"
      code_availability: public
      compute_budget:
        gpu: "RTX 4090"
        max_runtime: "30 minutes"
```

## 4. Action 类型

| type | 含义 | code_availability |
|------|------|-------------------|
| `repro_task` | 复现论文方法 | `public` / `upon_request` / `none` |
| `coding_task` | 写/改代码 | N/A |
| `run_task` | 运行实验 | N/A |
| `literature_search` | 需要进一步检索 | N/A |
| `ask_user` | 需要人工回答 | N/A |

## 5. 下游消费方式

### ResAgent 调用 ExpAgent

```bash
expagent advise --context "设计一个实验..." -o runs/001/
# → runs/001/scientific_decision.yaml
```

### 对 CodingAgent 的映射

| ExpAgent plan field | CodeTaskSpec field |
|---------------------|-------------------|
| `workspace_path` | `workspace_path` |
| `task_goal` | `task_goal` |
| `constraints` | `constraints` |
| `verify_commands` | `verify_commands` |

### 对 ReproAgent 的映射

| ExpAgent plan field | ReproTask field |
|---------------------|-----------------|
| `paper_url` | `paper_url` |
| `repo_url` | `repo_url` |
| `experiment_goal` | `experiment_goal` |

## 6. 产物结构 (对齐 ReproAgent)

```
runs/<timestamp>/
├── scientific_decision.yaml   # 核心输出
├── experiment_plan.yaml       # 如果有
├── validation_report.md
├── papers/                    # 本 run 下载的论文 metadata
└── logs/                      # trace 文件
```

## 7. 上下文管理

- `context.py` — LoopState (结构化状态)
- `context_policy.py` — ContextPolicy.for_model() (3档窗口自适应)
- `prompts.py` — SYSTEM_PROMPT + TOOLS + prompt builders
- 每轮从 state 重建 user prompt (CodingAgent 风格)
- 保留最近 4 对 tool_pairs 供 FC 连续性

## 8. API 重试

对齐 CodingAgent/ReproAgent：
- API 层: 3 次重试 (网络/5xx)，退避 2s/4s/8s
- Loop 层: 捕获失败作为 api_error 步骤，loop 继续

## 9. 当前测试状态

```
单元测试: 44/44 passed
E2E (DeepSeek API): 22/22 passed
```
