"""Retrieval-augmented Q&A over the review corpus.

A TF-IDF retriever finds the reviews most relevant to a question (pure and
deterministic), then the LLM answers grounded only in those retrieved
snippets, with the snippets returned as citations.
"""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from skymetrics.llm.client import OpenRouterClient

_SYSTEM = (
    "You answer questions about British Airways using ONLY the provided review "
    "excerpts. If the excerpts do not contain the answer, say so."
)


class ReviewRetriever:
    """TF-IDF retriever over a fixed review corpus."""

    def __init__(self, reviews: Sequence[str]):
        self.reviews = [r for r in reviews if isinstance(r, str) and r.strip()]
        if not self.reviews:
            raise ValueError("retriever needs at least one non-empty review")
        self._vec = TfidfVectorizer(stop_words="english")
        self._matrix = self._vec.fit_transform(self.reviews)

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        """Return the ``k`` reviews most similar to ``query``."""
        q = self._vec.transform([query])
        scores = cosine_similarity(q, self._matrix)[0]
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.reviews[i] for i in ranked[:k] if scores[i] > 0]


def build_prompt(query: str, context: Sequence[str]) -> str:
    """Build the grounded-answer prompt from retrieved context."""
    excerpts = "\n".join(f"- {c}" for c in context)
    return f"Question: {query}\n\nReview excerpts:\n{excerpts}\n\nAnswer:"


def answer_question(
    query: str,
    retriever: ReviewRetriever,
    client: OpenRouterClient | None = None,
    k: int = 5,
) -> dict[str, object]:
    """Answer a question grounded in retrieved reviews.

    Returns ``{"answer": str, "sources": list[str]}``. If nothing relevant is
    retrieved, returns a no-context answer without calling the LLM.
    """
    context = retriever.retrieve(query, k=k)
    if not context:
        return {"answer": "No relevant reviews found for that question.", "sources": []}
    client = client or OpenRouterClient()
    answer = client.complete(build_prompt(query, context), system=_SYSTEM)
    return {"answer": answer, "sources": context}
