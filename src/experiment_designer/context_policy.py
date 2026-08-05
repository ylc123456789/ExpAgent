"""Self-adaptive context limits, scaled to the model's context window.

Style-aligned with CodingAgent's context_policy.py and ReproAgent's
ContextPolicy model.
"""

from __future__ import annotations

from dataclasses import dataclass

# Known model context windows (tokens) — shared with models.py
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "deepseek-v4-pro": 1_000_000,
    "deepseek-v4-flash": 1_000_000,
    "deepseek-chat": 128_000,
    "deepseek-reasoner": 64_000,
    "gpt-4.1": 1_000_000,
    "gpt-4.1-mini": 1_000_000,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "claude-3.5-sonnet": 200_000,
    "claude-3-opus": 200_000,
}


@dataclass(frozen=True)
class ContextPolicy:
    """Resolved limits for packing prompt context.

    These are character-based heuristics (not exact token counts),
    scaled by the model's context window size.
    """

    step_history: int = 8         # how many compressed steps to show
    paper_index_entries: int = 8  # max papers in the index shown in context
    paper_index_summary_chars: int = 120  # max chars per paper one-line summary
    file_cache_count: int = 6     # max file cache entries to include
    file_cache_chars: int = 4000  # tail chars per file cache entry
    observation_chars: int = 500  # tail chars for compressed step observation
    last_result_chars: int = 6000 # max chars for the most recent full step

    @classmethod
    def for_model(cls, model: str | None) -> "ContextPolicy":
        """Resolve context limits for a model by its window size."""
        window = MODEL_CONTEXT_WINDOWS.get(
            (model or "").lower().split("/")[-1], 128_000
        )
        if window >= 500_000:
            return cls(
                step_history=20,
                paper_index_entries=15,
                paper_index_summary_chars=200,
                file_cache_count=12,
                file_cache_chars=8000,
                observation_chars=2000,
                last_result_chars=12000,
            )
        if window >= 128_000:
            return cls(
                step_history=10,
                paper_index_entries=8,
                paper_index_summary_chars=120,
                file_cache_count=6,
                file_cache_chars=4000,
                observation_chars=500,
                last_result_chars=6000,
            )
        return cls()
