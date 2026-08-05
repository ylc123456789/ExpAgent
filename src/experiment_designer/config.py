"""LLM configuration resolution — CLI arg > env var > config file."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path


def resolve_llm_config(
    config_path: str | Path | None = None,
    cli_model: str | None = None,
    cli_api_base: str | None = None,
    cli_api_key_env: str | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Resolve LLM config. CLI overrides > config file > defaults.

    Returns dict with keys: model, api_base, api_key_env.
    """
    source_env = env if env is not None else os.environ
    defaults = {
        "model": "deepseek-chat",
        "api_base": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
    }

    config_vals: dict[str, str] = {}
    if config_path:
        cfg = Path(config_path).expanduser().resolve()
        if cfg.exists():
            text = cfg.read_text(encoding="utf-8")
            llm = _parse_llm_section(text)
            for key in ("model", "api_base", "api_key_env"):
                if llm.get(key):
                    config_vals[key] = str(llm[key])

    merged = {**defaults, **config_vals}
    if cli_model:
        merged["model"] = cli_model
    if cli_api_base:
        merged["api_base"] = cli_api_base
    if cli_api_key_env:
        merged["api_key_env"] = cli_api_key_env

    return merged


def _parse_llm_section(text: str) -> dict[str, str]:
    """Simple YAML/JSON parser for the 'llm' config section."""
    # Try JSON first
    try:
        data = json.loads(text)
        return data.get("llm", {}) if isinstance(data, dict) else {}
    except (json.JSONDecodeError, ValueError):
        pass

    # Simple YAML parser
    result: dict[str, str] = {}
    in_llm = False
    base_indent = 0
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if stripped == "llm:":
            in_llm = True
            base_indent = indent
            continue
        if in_llm:
            if indent <= base_indent:
                in_llm = False
            else:
                for field in ("model", "api_base", "api_key_env"):
                    if stripped.startswith(f"{field}:"):
                        value = stripped.split(":", 1)[1].strip().strip("'").strip('"')
                        result[field] = value
    return result
