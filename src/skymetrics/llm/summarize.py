"""LLM-powered summarisation of airline reviews."""

from __future__ import annotations

from collections.abc import Sequence

from skymetrics.llm.client import OpenRouterClient

_SYSTEM = "You are an airline customer-experience analyst. Be concise and specific."


def build_prompt(reviews: Sequence[str], max_reviews: int = 40) -> str:
    """Build a summarisation prompt from a sample of reviews."""
    sample = [r for r in reviews if isinstance(r, str) and r.strip()][:max_reviews]
    joined = "\n".join(f"- {r}" for r in sample)
    return (
        "Summarise these British Airways customer reviews. Give:\n"
        "1. The top 3 complaints\n2. The top 3 praises\n3. One actionable recommendation.\n\n"
        f"Reviews:\n{joined}"
    )


def summarize_reviews(
    reviews: Sequence[str],
    client: OpenRouterClient | None = None,
    max_reviews: int = 40,
) -> str:
    """Summarise reviews via the LLM.

    Args:
        reviews: Review texts.
        client: An OpenRouterClient (injected in tests); created if omitted.
        max_reviews: Cap on reviews included in the prompt.

    Returns:
        The model's summary text.

    Raises:
        ValueError: If no non-empty reviews are provided.
    """
    if not any(isinstance(r, str) and r.strip() for r in reviews):
        raise ValueError("no reviews to summarise")
    client = client or OpenRouterClient()
    return client.complete(build_prompt(reviews, max_reviews), system=_SYSTEM)
