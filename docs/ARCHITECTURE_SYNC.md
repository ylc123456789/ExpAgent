# ExpAgent v2.1 — Technical Sync for Downstream Agents

## 1. ExpAgent 当前定位

ExpAgent 是纯科学顾问（Scientific Advisor），不负责调度任何 agent。

```
ResAgent (未来) → 读 scientific_decision.yaml → 调度 CodingAgent / ReproAgent
ExpAgent       → 输入 situation + artifacts → 输出 ScientificDecision
```

## 2. 核心架构变化

### 2.1 单一引擎：Function Calling Agentic Loop

ExpAgent 只有一个执行路径，不再有 fallback 或 multi-mode：

```text
advise(ctx) → _run_loop()
                ├── call_llm() with tools (search_papers, read_file, finish)
                ├── file_cache 记忆系统
                ├── grace stop (20 + 6 extra steps)
                └── adaptive context (ContextPolicy.for_model)
```

### 2.2 Function Calling（关键改进）

LLM 调用使用 OpenAI-compatible **function calling** (`tools` 参数)，不再从文本中解析 JSON。

下游如果需要调用 ExpAgent，可以复刻 `llm.py` 中的 pattern：

```python
# tools 定义
TOOLS = [{"type": "function", "function": {"name": "...", "parameters": {...}}}, ...]

# API 请求
body = {"model": ..., "messages": [...], "tools": TOOLS, "tool_choice": "auto"}

# 响应解析
choice["message"]["tool_calls"][0]["function"]  # → {name, arguments}
```

这比文本解析 JSON 可靠得多（22/22 e2e 全过 vs 之前 14/22）。

### 2.3 自适应上下文

`ContextPolicy.for_model(model)` 根据模型窗口大小自动调整：

| 窗口大小 | step_history | file_cache | search_results |
|----------|-------------|------------|----------------|
| >= 500K (deepseek-v4-pro) | 20 | 12 × 8000 | 8000 chars |
| >= 128K (deepseek-chat)   | 10 | 6 × 4000  | 3000 chars |
| < 128K                    | 4  | 3 × 2000  | 1500 chars |

模型窗口映射在 `models.py` 的 `MODEL_CONTEXT_WINDOWS` dict 中维护。

## 3. 已删除：Agent 路径耦合

ExpAgent 不再持有 CodingAgent/ReproAgent 的路径配置。以下已删除：

- `config.yaml` 中 `agents:` 段
- CLI 的 `--codingagent-path` / `--reproagent-path`
- `config.py` 中的路径解析函数

ExpAgent 只保留 LLM 配置（`llm:` 段）。

## 4. 输出格式：ScientificDecision

ExpAgent 的统一输出（`scientific_decision.yaml`）：

```yaml
summary: "一句话结论"
confidence: high | medium | low

conclusion:
  status: supported | not_supported | inconclusive | needs_more_experiments
  rationale: "科学推理过程"

evidence:
  - source: artifact | literature | reasoning
    description: "证据描述"

experiment_plan:           # 可选，design/revise 场景时有
  version: 1
  goal: {...}
  experiment_matrix: {...}
  tasks: {...}

recommended_actions:       # 核心：给 ResAgent 的建议
  - priority: high | medium | low
    type: repro_task | coding_task | run_task | literature_search | literature_reference | ask_user
    rationale: "科学理由"
    plan:                  # 自包含完整方案，下游可直接使用
      kind: repro_task
      paper_url: "..."
      repo_url: "..."
      experiment_goal: "..."
      code_availability: public | upon_request | none | ""
      ...

risks: [...]
needs_user_input: [...]
```

### RecommendedAction 类型说明

| type | 含义 | code_availability |
|------|------|-------------------|
| `repro_task` | 需要复现论文方法 | `public` (有repo) / `upon_request` (联系作者) / `none` (无代码，引用论文数据) |
| `coding_task` | 需要写/改代码 | N/A |
| `run_task` | 需要运行实验 | N/A |
| `literature_search` | 需要进一步检索 | N/A |
| `literature_reference` | 直接引用论文报告的数字 | N/A |
| `ask_user` | 需要人工回答 | N/A |

## 5. Python API

```python
from experiment_designer.advisor import advise
from experiment_designer.models import AdvisorContext, ArtifactRef

ctx = AdvisorContext(
    situation="自然语言描述当前研究情况...",
    artifacts=[ArtifactRef(id="result_001", type="run_log", summary="82.1% accuracy")],
    existing_plan=None,  # 可选，有现成 plan 时传入
)
decision, trace = advise(ctx, model="deepseek-chat", ...)
# → decision.recommended_actions[0].plan.paper_url  # 直接可用
```

## 6. 对下游的影响

### 对 CodingAgent

无需改动。ExpAgent 的 `recommended_actions[type=coding_task].plan` 字段直接映射 `CodeTaskSpec`：

| ExpAgent plan field | CodeTaskSpec field |
|---------------------|-------------------|
| `repo_path` | `repo_path` |
| `task_goal` | `task_goal` |
| `constraints` | `constraints` |
| `verify_commands` | `verify_commands` |

### 对 ReproAgent

无需改动。ExpAgent 的 `recommended_actions[type=repro_task].plan` 字段直接映射 `ReproTask`：

| ExpAgent plan field | ReproTask field |
|---------------------|-----------------|
| `paper_url` | `paper_url` |
| `repo_url` | `repo_url` |
| `experiment_goal` | `experiment_goal` |

### 对 ResAgent (未来 Orchestrator)

这就是 ResAgent 需要消费的接口。ResAgent 读取 `scientific_decision.yaml`，遍历 `recommended_actions`，把每个 action 的 `plan` 传给对应下游 agent 执行。

## 7. 当前测试状态

```
单元测试: 44/44 passed
E2E (真实 DeepSeek API): 22/22 passed
```
