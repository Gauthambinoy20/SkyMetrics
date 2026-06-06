"""Unit tests for the baseline sentiment module."""

import pandas as pd

from skymetrics.nlp import sentiment


class TestCleanReviewText:
    def test_strips_trip_verified_marker(self):
        assert sentiment.clean_review_text("✅ Trip Verified | Great flight") == "Great flight"

    def test_strips_not_verified_marker(self):
        assert sentiment.clean_review_text("Not Verified | Awful service") == "Awful service"

    def test_collapses_whitespace(self):
        assert sentiment.clean_review_text("good    seats\n\n  food") == "good seats food"

    def test_non_string_returns_empty(self):
        assert sentiment.clean_review_text(None) == ""
        assert sentiment.clean_review_text(3.14) == ""

    def test_plain_text_unchanged(self):
        assert sentiment.clean_review_text("comfortable seat") == "comfortable seat"


class TestPolarityAndClassify:
    def test_positive_review_scores_positive(self):
        assert sentiment.polarity_score("excellent wonderful amazing flight") > 0

    def test_negative_review_scores_negative(self):
        assert sentiment.polarity_score("terrible awful horrible delay") < 0

    def test_classify_positive(self):
        assert sentiment.classify_sentiment(0.5) == "positive"

    def test_classify_zero_is_negative_neutral(self):
        assert sentiment.classify_sentiment(0.0) == "negative/neutral"

    def test_classify_negative(self):
        assert sentiment.classify_sentiment(-0.3) == "negative/neutral"


class TestAnalyze:
    def test_returns_expected_columns(self):
        df = sentiment.analyze(["great", "awful"])
        assert list(df.columns) == ["review", "polarity", "label"]

    def test_empty_input_empty_frame(self):
        df = sentiment.analyze([])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert list(df.columns) == ["review", "polarity", "label"]

    def test_breakdown_counts_labels(self):
        breakdown = sentiment.sentiment_breakdown(
            ["wonderful amazing", "terrible horrible", "fantastic great"]
        )
        assert breakdown.get("positive") == 2
        assert breakdown.get("negative/neutral") == 1
