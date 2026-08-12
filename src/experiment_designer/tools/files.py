"""Local file reading tool for the ExpAgent agentic loop."""

from __future__ import annotations

from pathlib import Path


def read_file(path: str, max_chars: int = 16_000) -> str:
    """Read a local artifact file with head/middle/tail sections.

    Args:
        path: Absolute or relative path to the file.
        max_chars: Maximum characters to return. Files longer than this
                   show beginning, middle, and ending sections.

    Returns:
        File content with position markers when truncated.
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return f"[ERROR: file not found: {p}]"
    if not p.is_file():
        return f"[ERROR: not a file: {p}]"

    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"[ERROR reading file: {e}]"

    if len(text) > max_chars:
        third = max_chars // 3
        head = text[:third]
        mid_start = len(text) // 2 - third // 2
        middle = text[mid_start:mid_start + third]
        tail = text[-third:]
        return (
            f"[File: {p} ({len(text)} chars total — showing head/middle/tail {third} each)]\n"
            f"--- BEGINNING ---\n{head}\n"
            f"--- MIDDLE (around char {mid_start}) ---\n{middle}\n"
            f"--- END ---\n{tail}"
        )

    return f"[File: {p} ({len(text)} chars)]\n{text}"
