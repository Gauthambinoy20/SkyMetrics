"""Tests for the RAG retriever and /chat endpoint — LLM mocked."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from skymetrics.api.app import app
from skymetrics.llm.client import OpenRouterClient
from skymetrics.llm.rag import ReviewRetriever, answer_question

client = TestClient(app)

_REVIEWS = [
    "The food was delicious and the meal service was excellent.",
    "Flight was delayed for hours with no explanation.",
    "Cabin crew were friendly and professional.",
    "The in-flight meal was cold and tasteless.",
    "Lost my luggage at the airport.",
]


def _fake_client(reply="ANSWER"):
    def poster(u, b, h):
        return {"choices": [{"message": {"content": reply}}]}

    return OpenRouterClient(api_key="k", poster=poster)


class TestRetriever:
    def test_retrieves_relevant_reviews(self):
        r = ReviewRetriever(_REVIEWS)
        hits = r.retrieve("how is the food and meals?", k=2)
        assert any("food" in h.lower() or "meal" in h.lower() for h in hits)

    def test_empty_corpus_raises(self):
        with pytest.raises(ValueError):
            ReviewRetriever(["", "  "])

    def test_irrelevant_query_returns_nothing(self):
        r = ReviewRetriever(_REVIEWS)
        assert r.retrieve("quantum chromodynamics submarine", k=3) == []


class TestAnswer:
    def test_answer_uses_context_and_client(self):
        r = ReviewRetriever(_REVIEWS)
        out = answer_question("food?", r, client=_fake_client("Customers like the food."))
        assert out["answer"] == "Customers like the food."
        assert out["sources"]

    def test_no_context_skips_llm(self):
        r = ReviewRetriever(_REVIEWS)
        out = answer_question("xyzzy plugh", r, client=_fake_client())
        assert out["sources"] == []


class TestChatEndpoint:
    def test_chat_ok(self):
        with patch(
            "skymetrics.api.app.answer_question",
            return_value={"answer": "A", "sources": ["s"]},
        ):
            resp = client.post("/chat", json={"question": "food?", "reviews": _REVIEWS})
        assert resp.status_code == 200
        assert resp.json()["answer"] == "A"
