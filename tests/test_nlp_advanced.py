"""Unit tests for aspect sentiment, topic extraction and transformer wrapper."""

from skymetrics.nlp import aspects, topics
from skymetrics.nlp.transformer import TransformerSentiment


class TestAspects:
    def test_split_sentences(self):
        assert aspects.split_sentences("Great seat. Bad food! Late?") == [
            "Great seat",
            "Bad food",
            "Late",
        ]

    def test_detects_mentioned_aspects_only(self):
        result = aspects.aspect_sentiment("The crew were wonderful. The wifi never worked.")
        assert "staff" in result
        assert "wifi" in result
        assert "food" not in result

    def test_positive_and_negative_aspects_scored(self):
        result = aspects.aspect_sentiment("Excellent friendly crew. Terrible awful delay.")
        assert result["staff"] > 0
        assert result["delay"] < 0

    def test_empty_text_no_aspects(self):
        assert aspects.aspect_sentiment("") == {}


class TestTopics:
    def test_top_terms_ranks_frequent_distinctive_words(self):
        corpus = [
            "delayed flight delayed again",
            "delayed and late departure",
            "comfortable seat lovely crew",
        ]
        terms = [t for t, _ in topics.top_terms(corpus, n=3)]
        assert "delayed" in terms

    def test_empty_corpus_returns_empty(self):
        assert topics.top_terms([]) == []

    def test_all_stopwords_returns_empty(self):
        assert topics.top_terms(["the and of", "to a an"]) == []


class TestTransformerWrapper:
    def test_classify_uses_injected_pipeline(self):
        fake = lambda text: [{"label": "POSITIVE", "score": 0.99}]  # noqa: E731
        clf = TransformerSentiment(pipeline=fake)
        out = clf.classify("amazing flight")
        assert out == {"label": "positive", "score": 0.99}

    def test_empty_text_is_neutral_without_model(self):
        # pipeline=None would require a model download; empty text must skip it.
        clf = TransformerSentiment(pipeline=None)
        assert clf.classify("   ") == {"label": "neutral", "score": 0.0}
