"""Aspect-based sentiment for airline reviews.

Splits a review into sentences, attributes each sentence to airline service
aspects by keyword, and scores per-aspect sentiment with the baseline
polarity. Pure and dependency-light so it runs fast in tests and the API.
"""

from __future__ import annotations

import re

from skymetrics.nlp.sentiment import polarity_score

# Service aspects mapped to trigger keywords found in BA reviews.
ASPECTS: dict[str, tuple[str, ...]] = {
    "seat": ("seat", "legroom", "recline", "cabin", "comfort"),
    "food": ("food", "meal", "catering", "drink", "snack", "menu"),
    "staff": ("staff", "crew", "service", "attendant", "steward", "cabin crew"),
    "delay": ("delay", "delayed", "late", "cancelled", "cancellation", "wait"),
    "baggage": ("baggage", "luggage", "bag", "suitcase"),
    "value": ("price", "value", "money", "expensive", "cost", "refund"),
    "wifi": ("wifi", "wi-fi", "internet", "entertainment", "screen"),
}

_SENTENCE_SPLIT = re.compile(r"[.!?]+")


def split_sentences(text: str) -> list[str]:
    """Split text into trimmed, non-empty sentences."""
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def aspect_sentiment(text: str) -> dict[str, float]:
    """Return ``{aspect: mean_polarity}`` for aspects mentioned in ``text``.

    Only aspects whose keywords appear are included. A sentence may count
    toward several aspects; each aspect averages the polarity of all its
    matching sentences.
    """
    scores: dict[str, list[float]] = {a: [] for a in ASPECTS}
    for sentence in split_sentences(text):
        lowered = sentence.lower()
        polarity = polarity_score(sentence)
        for aspect, keywords in ASPECTS.items():
            if any(kw in lowered for kw in keywords):
                scores[aspect].append(polarity)
    return {a: sum(v) / len(v) for a, v in scores.items() if v}
