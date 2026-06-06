"""Baseline sentiment analysis for airline reviews.

Extracted from the original Skytrax sentiment notebook into pure, testable
functions. This is the *baseline* TextBlob implementation; a transformer-based
model is added alongside it in a later slice for comparison.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd
from textblob import TextBlob

# Markers Skytrax prepends to review bodies that carry no sentiment signal.
_VERIFICATION_MARKERS = (
    "✅ Trip Verified",
    "Trip Verified",
    "✅ Verified Review",
    "Verified Review",
    "Not Verified",
)
_WHITESPACE = re.compile(r"\s+")


def clean_review_text(text: str) -> str:
    """Strip verification markers and normalise whitespace in a review.

    Args:
        text: Raw review text as scraped from Skytrax.

    Returns:
        Cleaned text with leading verification badges removed and runs of
        whitespace collapsed to single spaces. Non-string input yields "".
    """
    if not isinstance(text, str):
        return ""
    cleaned = text
    for marker in _VERIFICATION_MARKERS:
        cleaned = cleaned.replace(marker, " ")
    # A leading separator (the "|" Skytrax uses after the badge) adds no signal.
    cleaned = cleaned.lstrip(" |")
    return _WHITESPACE.sub(" ", cleaned).strip()


def polarity_score(text: str) -> float:
    """Return the TextBlob polarity of ``text`` in the range [-1.0, 1.0]."""
    return float(TextBlob(clean_review_text(text)).sentiment.polarity)


def classify_sentiment(polarity: float) -> str:
    """Map a polarity score to a label.

    Positive polarity is ``"positive"``; zero or negative is
    ``"negative/neutral"`` — matching the original notebook's binning.
    """
    return "positive" if polarity > 0 else "negative/neutral"


def analyze(reviews: Iterable[str]) -> pd.DataFrame:
    """Score an iterable of reviews.

    Args:
        reviews: Raw review strings.

    Returns:
        A DataFrame with ``review``, ``polarity`` and ``label`` columns.
        Empty input produces an empty DataFrame with those columns.
    """
    rows = []
    for raw in reviews:
        polarity = polarity_score(raw)
        rows.append(
            {
                "review": raw,
                "polarity": polarity,
                "label": classify_sentiment(polarity),
            }
        )
    return pd.DataFrame(rows, columns=["review", "polarity", "label"])


def sentiment_breakdown(reviews: Iterable[str]) -> dict[str, int]:
    """Return a ``{label: count}`` summary for an iterable of reviews."""
    labels = analyze(reviews)["label"]
    return labels.value_counts().to_dict()
