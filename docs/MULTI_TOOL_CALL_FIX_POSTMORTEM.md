# Multi-Tool-Call Fix Postmortem

**Date**: 2026-08-07
**Status**: Fixed in working tree
**Related**: `docs/HANDOVER_MULTI_TOOL_FIX.md`

---

## 1. 问题回顾

DeepSeek 在一次 API 响应里经常返回 2-4 个并行 `tool_calls`。

旧代码在 `src/experiment_designer/llm.py` 中只处理第一个：

```python
tool_calls = msg.get("tool_calls", [])
if tool_calls:
    tc = tool_calls[0]    # 后面的 tool_calls 被静默丢弃
```

结果是：LLM 以为自己执行了 N 个操作，但实际只执行了 1 个。下一轮它发现信息缺失，会再次搜索/读取/保存，形成搜索洪水、save→read 链断裂、`note_finding` 缺口，最终耗尽 step budget。

---

## 2. 为什么之前的修法不成功

之前失败的核心原因不是“不能执行多个 tool_calls”，而是**试图把多个 tool_calls 重新构造进 Function Calling 历史消息**。

### Attempt 1 / Attempt 2 的共同点

两种尝试都做了这些事：

1. 让 `llm.py` 返回全部 `tool_calls`
2. 让 `advisor.py` 执行全部调用
3. 自己构造一个包含多个 `tool_calls` 的 `assistant` 消息
4. 把这个消息重新喂回下一轮 `messages`

问题出在第 3/4 步。

DeepSeek 对 FC 历史消息格式很严格。一旦 `assistant` 消息里包含 `tool_calls`，后续消息必须和它期望的 tool_call/tool_result 结构完全匹配。之前尝试了两种构造方式：

- 合成 `call_<step>` 这样的 tool_call id
- 直接透传原始 `id/index/type/function` 字段

但都还是在 finish 之后触发 `HTTP 400 Bad Request`。

### 关键脆弱点

旧 loop 在 finish 失败时会这样做：

```python
tool_pairs = _make_pair(name, args, "Parse error ...", step)
continue
```

也就是**替换掉整个 tool_pairs**。

如果前一轮 LLM 实际返回的是多个并行 tool_calls，而我们只重建了其中一部分历史，或者 id/content/index 的细节和 DeepSeek 期望不完全一致，那么下一轮请求的消息历史就已经和 API 真实返回不一致了。这个不一致在 finish 后更容易暴露，因为 finish 会触发 parse/validation retry，随后的 API 调用直接 400。

当时没有确认清楚 DeepSeek 对多 `tool_calls` assistant 消息的精确要求，包括：

- `content` 应该是 `null` 还是 `""`
- `index` 是否必须保留
- 多个 `tool_calls` 是否必须全部有对应 `tool` 结果
- 合成 id 是否可接受
- finish 失败后重试时，历史消息应该怎样表示

继续沿着这条路修，本质是在逆向一个未完全明确的 FC 序列化格式，风险高、收益低。

---

## 3. 本次修复思路

采用交接文档里的 Option A：**执行全部 tool_calls，但不把全部调用都存进 FC 历史消息**。

ExpAgent 的 loop 本来就是 CodingAgent 风格：

```text
每轮重新从 LoopState build_turn_prompt
```

也就是说，长期信息本来就不完全依赖 FC messages，而是依赖：

- `state.compressed`：压缩步骤历史
- `state.findings`：`note_finding` 结果
- `state.paper_index`：已保存论文索引
- `state.file_cache`：最近读取内容尾部

所以不需要把全部并行 tool_calls 都塞进 `messages`。只需要保留一个 tool_call→tool_result pair，让 FC 上下文知道“我刚才调用过工具并拿到了结果”；其余并行调用的结果通过 `state.compressed` 进入下一轮 user prompt。

---

## 4. 具体改动

### `src/experiment_designer/llm.py`

原来只返回第一个 tool_call：

```python
return {
    "type": "tool_call",
    "name": func.get("name", "unknown"),
    "arguments": arguments,
}
```

现在返回全部调用：

```python
calls: list[dict] = []
for tc in tool_calls:
    func = tc.get("function", {})
    try:
        arguments = json.loads(func.get("arguments", "{}"))
    except json.JSONDecodeError:
        arguments = {}
    calls.append({
        "name": func.get("name", "unknown"),
        "arguments": arguments,
    })
return {"type": "tool_calls", "calls": calls}
```

mock 响应也同步改成 `type == "tool_calls"`。

### `src/experiment_designer/advisor.py`

主循环从“处理一个 call”改成“按顺序处理全部 calls”：

```python
first_pair: list[dict] | None = None

for call_index, call in enumerate(result["calls"]):
    name = call["name"]
    args = call["arguments"]
    output = execute_one_call(...)

    update_state(...)

    if call_index == 0:
        first_pair = _make_pair(name, args, output, step)
```

循环结束后，只把第一个 call 放进 FC 历史：

```python
if first_pair:
    tool_pairs += first_pair
    tool_pairs = tool_pairs[-(MAX_TOOL_PAIRS * 2):]
```

其余 call 的结果已经写入 `state.compressed` / `findings` / `paper_index`，下一轮 `build_turn_prompt(state, policy)` 会自然展示。

### finish 语义

`finish` 仍然是终止动作：

- 如果 `finish` 成功：写 `state.json` 并直接返回 decision
- 如果 `finish` parse/validation 失败：
  - 错误写入 `state.compressed`
  - 只有当 `finish` 是本轮第一个 call 时，才把它做成 FC tool_pair
  - 如果它前面已经有其他 call，那些 call 已经执行并写入 state；finish 错误通过 compressed history 进入下一轮 prompt

这避免了为了重试 finish 而重建一段 DeepSeek 可能不接受的多 tool_call assistant 历史。

---

## 5. 回归测试

新增测试在 `tests/test_planner.py`：

1. `test_call_llm_returns_all_parallel_tool_calls`
   - mock 一个包含 2 个 `tool_calls` 的 API 响应
   - 断言 `call_llm()` 返回两个 calls，不再丢第二个

2. `test_advisor_executes_all_parallel_tool_calls`
   - monkeypatch `advisor.call_llm`
   - 第一轮返回两个 `note_finding`
   - 第二轮返回合法 `finish`
   - 断言两个 finding 都被执行并写入 `state.json`

验证结果：

```text
46 passed, 22 deselected
```

真实 DeepSeek e2e：

```text
tests/test_e2e.py::TestPlanGeneration::test_simple_idea_generates_valid_plan
1 passed in 224.50s
```

在最新真实运行 `runs/tests/plan/20260807-103633` 中：

- step1 原始响应包含 3 个并行 tool_calls：
  - `read_file`
  - `search_papers`
  - `search_papers`
- `state.json` 中对应动作都执行了
- 后续还出现一次响应里两个 `save_paper` 都保存成功
- 没有出现 `api_error` / HTTP 400

---

## 6. 经验总结

这次问题说明：对于“重建 prompt from state”的 agent loop，不要强行把所有并行工具调用都编码进 provider-specific FC 历史。

更稳的边界是：

- API 层：完整解析并返回 LLM 给的所有 tool_calls
- Loop 层：全部执行，全部写入结构化 state
- FC 历史层：只保留最小必要 continuity pair
- Prompt 层：下一轮通过 `build_turn_prompt(state)` 把额外结果重新展示给 LLM

这样既修掉了“静默丢调用”的 bug，也避开了 DeepSeek 多 tool_call 历史消息格式不明确的坑。
