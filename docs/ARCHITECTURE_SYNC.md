# ExpAgent Architecture — Downstream Agent Sync

## 1. 定位

ExpAgent = **Scientific Advisor**。不调度任何 agent。

```
ResAgent → 读 scientific_decision.json → 调度 CodingAgent / ReproAgent / Runner
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

每个 `recommended_action.plan` 是**自包含**的完整方案，下游可直接使用。

`conclusion` 可选（纯问答/讨论类请求时为 None）。

```json
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
# → runs/001/scientific_decision.json
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

## 6. 产物结构

```
runs/<timestamp>/
├── session.yaml               # 会话索引卡 (§3 跨模块契约)
├── state.json                 # 完整运行记录 (每步 action + observation)
├── scientific_decision.json   # 核心输出 (JSON)
├── experiment_plan.yaml       # 如果有
├── validation_report.md
├── papers/                    # 本 run 下载的论文 (metadata + arXiv 全文)
└── logs/                      # LLM prompt/response trace
```

## 7. 目录结构

```
experiment_designer/
├── __init__.py          # 稳定公共 API (根导出)
├── main.py              # CLI 参数解析 + REPL + 依赖装配
├── agent.py             # 顶层公共运行 API: advise()
├── models.py            # Pydantic 输入/输出/持久化模型
├── config.py            # LLM 配置解析 (CLI > env > config file)
├── controller/          # Agentic loop + 动作分派 + 校验
│   ├── loop.py          #   主循环 _run_loop + 工具分派
│   ├── planner.py       #   plan()/revise() 设计流程包装
│   └── validator.py     #   validate()/validate_decision() 确定性校验
├── prompts/             # Prompt 与工具 schema (字节级冻结)
│   ├── system.py        #   SYSTEM_PROMPT
│   ├── schemas.py       #   TOOLS + 嵌套 JSON schema
│   └── rendering.py     #   build_*_prompt 状态渲染
├── context/             # 循环状态与上下文预算
│   ├── builder.py       #   LoopState
│   └── policy.py        #   ContextPolicy.for_model()
├── tools/               # 外部副作用工具
│   ├── files.py         #   read_file
│   └── papers.py        #   search_papers / save_paper
├── llm.py               # OpenAI 兼容 API 客户端 (3 次重试)
├── session.py           # session.yaml + state.json 追踪
└── report.py            # experiment_plan.yaml + scientific_decision.json
```

兼容性：`advisor.py` / `planner.py` / `validator.py` 是薄转发模块，保留旧导入路径
(ResAgent 通过 `from experiment_designer.advisor import advise` 调用)。

## 8. 上下文管理

- `context/builder.py` — LoopState (结构化状态)
- `context/policy.py` — ContextPolicy.for_model() (3档窗口自适应)
- `prompts/` — SYSTEM_PROMPT + TOOLS + prompt builders
- 每轮从 state 重建 user prompt (CodingAgent 风格)
- 保留最近 4 对 tool_pairs 供 FC 连续性

## 9. API 重试

对齐 CodingAgent/ReproAgent：
- API 层: 3 次重试 (网络/5xx)，退避 2s/4s/8s
- Loop 层: 捕获失败作为 api_error 步骤，loop 继续

## 10. 当前测试状态

```
单元测试: 61 passed, 22 deselected (e2e marker)
E2E (DeepSeek API): 22/22 passed
```
