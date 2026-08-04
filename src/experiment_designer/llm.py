"""LLM client — OpenAI-compatible API via urllib.

Matches ReproAgent's llm.py style: no openai package, no chat history,
fresh system+user per call, optional trace logging.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def call_llm(
    *,
    model: str,
    api_base: str,
    api_key_env: str,
    system: str,
    user: str,
    mock: bool = False,
    trace_dir: Path | None = None,
    trace_label: str = "llm",
) -> str:
    """Call an OpenAI-compatible chat completions API. No chat history.

    Args:
        model: Model name (e.g. 'deepseek-chat').
        api_base: Base URL for the API.
        api_key_env: Environment variable name holding the API key.
        system: System prompt.
        user: User prompt.
        mock: If True, return deterministic mock output.
        trace_dir: If set, write prompt/response traces there for debugging.
        trace_label: Label prefix for trace files.

    Returns:
        The LLM's raw text response.
    """
    if mock:
        text = _mock_plan(user)
        if trace_dir:
            _write_trace(trace_dir, trace_label, system, user, text)
        return text

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(
            f"{api_key_env} is not set. Set it or use mock=True for testing."
        )

    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.3,
    }

    url = _chat_completions_url(api_base)
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    text = data["choices"][0]["message"]["content"].strip()

    if trace_dir:
        _write_trace(trace_dir, trace_label, system, user, text)

    return text


def _chat_completions_url(api_base: str) -> str:
    """Normalize API base to the chat completions endpoint."""
    base = api_base.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


def _write_trace(
    trace_dir: Path,
    label: str,
    system: str,
    user: str,
    response: str,
) -> None:
    """Write prompt/response trace files for debugging."""
    trace_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_") or "llm"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    prefix = trace_dir / f"llm_{stamp}_{safe}"
    (prefix.with_suffix(".prompt.txt")).write_text(
        f"[system]\n{system}\n\n[user]\n{user}", encoding="utf-8"
    )
    (prefix.with_suffix(".response.txt")).write_text(response, encoding="utf-8")


# ── Mock responses ─────────────────────────────────────────────────


def _mock_plan(user: str) -> str:
    """Return a deterministic mock experiment plan for pipeline testing."""
    return """```yaml
version: 1
goal:
  summary: "验证通道注意力机制在 CIFAR-10 图像分类上的参数效率"
  hypothesis: "新的通道注意力变体在参数量增加 < 5% 的条件下，比标准 ResNet-18 的 top-1 accuracy 提升 >= 2%"
  success_criteria:
    - "参数量增加 < 5%"
    - "top-1 accuracy 提升 >= 2% vs ResNet-18 baseline"
    - "训练时间增加 < 10%"

experiment_matrix:
  datasets:
    - name: "CIFAR-10"
      split: "standard"
      rationale: "轻量验证，快速迭代，大多数图像分类论文都报告此数据集结果"
  methods:
    - name: "proposed_channel_attention"
      type: "new_method"
      implementation_status: "needs_code"
      rationale: "本次实验要验证的核心方法"
    - name: "resnet18_baseline"
      type: "baseline"
      implementation_status: "needs_repro"
      rationale: "最通用的图像分类基线，几乎每篇论文都报告"
    - name: "resnet18_se"
      type: "baseline"
      implementation_status: "needs_repro"
      rationale: "SENet 是最经典的通道注意力方法，作为直接对比基线"
    - name: "resnet18_no_attention"
      type: "ablation"
      implementation_status: "existing"
      rationale: "消融实验：验证注意力模块本身是否带来增益"
  metrics:
    - name: "top1_accuracy"
      rationale: "主要评估指标"
    - name: "params_count"
      rationale: "验证参数效率假设"
    - name: "train_time_per_epoch"
      rationale: "验证计算开销假设"
    - name: "flops"
      rationale: "补充计算复杂度指标"

tasks:
  coding_tasks:
    - id: "code_001"
      repo_path: "/home/cyl/my_project"
      task_goal: "在 models/ 中实现 proposed_channel_attention 模块"
      constraints:
        - "不改变 baseline 训练入口"
        - "保持与现有模型注册机制兼容"
      verify_commands:
        - "python -m pytest tests/test_attention.py"
      expected_artifacts:
        - "patch.diff"
        - "verification_report.md"
      rationale: "核心方法需要从头实现"

  repro_tasks:
    - id: "repro_001"
      paper_url: "https://arxiv.org/abs/1709.01507"
      repo_url: "https://github.com/moskomule/senet.pytorch"
      experiment_goal: "在 CIFAR-10 上复现 SE-ResNet-18 的 bounded 结果（10 epochs）"
      compute_budget:
        gpu: "RTX 4090"
        max_runtime: "30 minutes"
        max_trials: 3
      expected_metrics:
        - "top1_accuracy"
        - "params_count"
      rationale: "SENet 是通道注意力的代表方法，需要其 CIFAR-10 结果作为对比基线"

  run_tasks:
    - id: "run_001"
      command_goal: "运行 proposed_method 的 CIFAR-10 bounded 训练（10 epochs）"
      expected_runtime: "20 minutes"
      requires_gpu: true
      rationale: "核心对比实验"

analysis_plan:
  comparisons:
    - "proposed_channel_attention vs resnet18_baseline（主要对比）"
    - "proposed_channel_attention vs resnet18_se（通道注意力内部对比）"
    - "proposed_channel_attention vs resnet18_no_attention（消融）"
  plots:
    - "accuracy vs params_count 散点图（标注各方法）"
    - "训练曲线对比图"
  failure_checks:
    - "检查参数量增加是否主要来自 attention 模块还是其他部分"
    - "检查是否只是因为训练更久导致提升（控制 epoch 数一致）"
    - "验证实验结果在 3 次随机种子下稳定"

risks:
  - description: "SE-Net 复现代码可能无法直接在 CIFAR-10 上运行"
    mitigation: "准备 PyTorch 官方 torchvision 的 SE-ResNet 实现作为备选"
  - description: "CIFAR-10 是小数据集，结论可能不推广到 ImageNet 规模"
    mitigation: "后续阶段在 ImageNet 子集上验证"
  - description: "如果参数量控制不住，可能是不公平对比"
    mitigation: "设置参数量上限约束，若超限则调整通道数重新实验"
```"""
