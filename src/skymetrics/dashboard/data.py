"""Pure data-prep functions backing the dashboard.

Kept separate from the Streamlit UI so every transformation is unit-tested
without launching a browser.
"""

from __future__ import annotations

import pandas as pd

from skymetrics.nlp.sentiment import analyze


def scored_reviews(reviews: pd.Series | list[str]) -> pd.DataFrame:
    """Return a sentiment-scored DataFrame for a column/list of reviews."""
    texts = reviews.tolist() if isinstance(reviews, pd.Series) else list(reviews)
    return analyze(texts)


def sentiment_summary(scored: pd.DataFrame) -> dict[str, float]:
    """Summarise a scored frame into counts and the positive share."""
    total = len(scored)
    positives = int((scored["label"] == "positive").sum())
    return {
        "total": total,
        "positive": positives,
        "negative_neutral": total - positives,
        "positive_pct": round(100 * positives / total, 1) if total else 0.0,
    }


def benchmark_airlines(df: pd.DataFrame) -> pd.DataFrame:
    """Compare mean polarity and positive share across airlines.

    Args:
        df: Frame with ``airline``, ``polarity`` and ``label`` columns.

    Returns:
        One row per airline, sorted by positive share descending.
    """
    required = {"airline", "polarity", "label"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"missing columns: {sorted(missing)}")

    grouped = df.groupby("airline").agg(
        reviews=("polarity", "size"),
        mean_polarity=("polarity", "mean"),
        positive_pct=("label", lambda s: round(100 * (s == "positive").mean(), 1)),
    )
    return grouped.reset_index().sort_values("positive_pct", ascending=False)
