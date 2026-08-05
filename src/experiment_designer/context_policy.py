"""Self-adaptive context limits, scaled to model context window size.

Style-aligned with CodingAgent's context_policy.py.
"""

from __future__ import annotations

from pydantic import BaseModel

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


class ContextPolicy(BaseModel):
    """Limits for packing prompt context, scaled to model window size."""

    step_history: int = 8
    paper_index_entries: int = 8
    file_cache_count: int = 6
    file_cache_chars: int = 4000
    observation_tail: int = 500
    last_result_chars: int = 6000
    search_results_chars: int = 3000

    @classmethod
    def for_model(cls, model: str | None) -> "ContextPolicy":
        """Resolve context policy for a model by its window size."""
        window = MODEL_CONTEXT_WINDOWS.get(
            (model or "").lower().split("/")[-1], 128_000
        )
        if window >= 500_000:
            return cls(
                step_history=20, paper_index_entries=15,
                file_cache_count=12, file_cache_chars=8000,
                observation_tail=2000, last_result_chars=12000,
                search_results_chars=8000,
            )
        if window >= 128_000:
            return cls(
                step_history=10, paper_index_entries=8,
                file_cache_count=6, file_cache_chars=4000,
                observation_tail=500, last_result_chars=6000,
                search_results_chars=3000,
            )
        return cls(
            step_history=4, file_cache_count=3,
            file_cache_chars=2000, observation_tail=300,
            last_result_chars=3000, search_results_chars=1500,
        )
