"""Agent path resolution — CLI arg > env var > config file.

Follows the same pattern as ReproAgent's integrations/codingagent.py.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path


def resolve_codingagent_path(
    cli_path: str | Path | None = None,
    config_path: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> Path | None:
    """Resolve CodingAgent path from CLI, CODINGAGENT_PATH, or config.

    Returns None if no path is configured (ExpAgent can run standalone).
    """
    source_env = env if env is not None else os.environ
    if cli_path:
        return _validate_agent_path(cli_path, base_dir=Path.cwd())
    env_path = source_env.get("CODINGAGENT_PATH")
    if env_path:
        return _validate_agent_path(env_path, base_dir=Path.cwd())
    if config_path:
        loaded = _load_config_path(Path(config_path), "codingagent_path")
        if loaded:
            return _validate_agent_path(loaded, base_dir=Path(config_path).expanduser().resolve().parent)
    return None


def resolve_reproagent_path(
    cli_path: str | Path | None = None,
    config_path: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> Path | None:
    """Resolve ReproAgent path from CLI, REPROAGENT_PATH, or config.

    Returns None if no path is configured (ExpAgent can run standalone).
    """
    source_env = env if env is not None else os.environ
    if cli_path:
        return _validate_agent_path(cli_path, base_dir=Path.cwd())
    env_path = source_env.get("REPROAGENT_PATH")
    if env_path:
        return _validate_agent_path(env_path, base_dir=Path.cwd())
    if config_path:
        loaded = _load_config_path(Path(config_path), "reproagent_path")
        if loaded:
            return _validate_agent_path(loaded, base_dir=Path(config_path).expanduser().resolve().parent)
    return None


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

    # Read config file values
    config_vals: dict[str, str] = {}
    if config_path:
        cfg = Path(config_path).expanduser().resolve()
        if cfg.exists():
            text = cfg.read_text(encoding="utf-8")
            data = _parse_yaml_simple(text) or _parse_json_simple(text) or {}
            llm = data.get("llm", {}) if isinstance(data, dict) else {}
            if isinstance(llm, dict):
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


# ── Internal helpers ──────────────────────────────────────────────


def _validate_agent_path(path: str | Path, base_dir: Path | None = None) -> Path:
    """Validate and resolve an agent checkout path."""
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = (base_dir or Path.cwd()) / candidate
    candidate = candidate.resolve()
    if not candidate.exists():
        raise ValueError(f"Agent path does not exist: {candidate}")
    if not candidate.is_dir():
        raise ValueError(f"Agent path is not a directory: {candidate}")
    return candidate


def _load_config_path(config_path: Path, key: str) -> str | None:
    """Extract a specific agent path key from a config file."""
    path = config_path.expanduser().resolve()
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        agents = data.get("agents", {}) if isinstance(data, dict) else {}
        value = agents.get(key) if isinstance(agents, dict) else None
        return str(value) if value else None
    return _load_yaml_config_path(text, key)


def _load_yaml_config_path(text: str, key: str) -> str | None:
    """Simple YAML parser for extracting agent paths without pyyaml dependency."""
    in_agents = False
    agents_indent = 0
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if stripped == "agents:":
            in_agents = True
            agents_indent = indent
            continue
        if in_agents and indent <= agents_indent:
            in_agents = False
        if in_agents and stripped.startswith(f"{key}:"):
            value = stripped.split(":", 1)[1].strip().strip("'").strip('"')
            return value or None
    return None


def _parse_yaml_simple(text: str) -> dict | None:
    """Crude YAML parser for top-level 'llm' dict. Returns None if not YAML-like."""
    result: dict = {}
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
            result.setdefault("llm", {})
            continue
        if in_llm:
            if indent <= base_indent:
                in_llm = False
            else:
                for field in ("model", "api_base", "api_key_env"):
                    if stripped.startswith(f"{field}:"):
                        value = stripped.split(":", 1)[1].strip().strip("'").strip('"')
                        result["llm"][field] = value
    return result if result else None


def _parse_json_simple(text: str) -> dict | None:
    """Try to parse as JSON. Returns None on failure."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
