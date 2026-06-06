"""Tests for the OpenRouter client and review summariser — no network."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import skymetrics.api.app as api
from skymetrics.api.app import app
from skymetrics.llm import summarize
from skymetrics.llm.client import LLMConfigError, OpenRouterClient

client = TestClient(app)


def _fake_poster(reply: str):
    def poster(url, body, headers):
        assert headers["Authorization"].startswith("Bearer ")
        return {"choices": [{"message": {"content": reply}}]}

    return poster


class TestClient_:
    def test_complete_returns_content(self):
        c = OpenRouterClient(api_key="k", poster=_fake_poster("hello"))
        assert c.complete("hi") == "hello"

    def test_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        c = OpenRouterClient(api_key="", poster=_fake_poster("x"))
        with pytest.raises(LLMConfigError):
            c.complete("hi")


class TestSummarize:
    def test_build_prompt_includes_reviews(self):
        prompt = summarize.build_prompt(["great crew", "late flight"])
        assert "great crew" in prompt and "complaints" in prompt

    def test_summarize_uses_client(self):
        c = OpenRouterClient(api_key="k", poster=_fake_poster("SUMMARY"))
        assert summarize.summarize_reviews(["good", "bad"], client=c) == "SUMMARY"

    def test_summarize_empty_raises(self):
        with pytest.raises(ValueError):
            summarize.summarize_reviews(["", "   "])


class TestSummaryEndpoint:
    def test_summary_ok(self):
        with patch.object(api, "summarize_reviews", return_value="SUMMARY"):
            resp = client.post("/summary", json={"reviews": ["good", "bad"]})
        assert resp.status_code == 200
        assert resp.json()["summary"] == "SUMMARY"

    def test_summary_503_without_key(self):
        with patch.object(api, "summarize_reviews", side_effect=LLMConfigError("no key")):
            resp = client.post("/summary", json={"reviews": ["good"]})
        assert resp.status_code == 503
