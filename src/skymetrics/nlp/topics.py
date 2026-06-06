"""Lightweight topic / keyword extraction over a review corpus.

Uses TF-IDF term ranking — fast and deterministic, with no model downloads —
to surface the most distinctive terms and simple co-occurrence topics.
"""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.feature_extraction.text import TfidfVectorizer


def top_terms(
    reviews: Sequence[str], n: int = 15, max_features: int = 2000
) -> list[tuple[str, float]]:
    """Return the ``n`` highest mean-TF-IDF terms across the corpus.

    Args:
        reviews: Review texts.
        n: Number of terms to return.
        max_features: Vocabulary cap for the vectoriser.

    Returns:
        ``[(term, score), ...]`` sorted by descending score. Empty input or a
        corpus with no usable tokens yields an empty list.
    """
    cleaned = [r for r in reviews if isinstance(r, str) and r.strip()]
    if not cleaned:
        return []
    vec = TfidfVectorizer(stop_words="english", max_features=max_features)
    try:
        matrix = vec.fit_transform(cleaned)
    except ValueError:
        # Raised when the vocabulary ends up empty (e.g. all stop words).
        return []
    means = matrix.mean(axis=0).A1
    terms = vec.get_feature_names_out()
    ranked = sorted(zip(terms, means, strict=True), key=lambda kv: kv[1], reverse=True)
    return [(t, float(s)) for t, s in ranked[:n]]
