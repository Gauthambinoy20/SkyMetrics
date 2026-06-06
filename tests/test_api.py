"""Endpoint tests for the FastAPI app using TestClient."""

from fastapi.testclient import TestClient

from skymetrics.api.app import app

client = TestClient(app)


def test_health_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_sentiment_positive():
    resp = client.post("/sentiment", json={"text": "excellent wonderful flight"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] == "positive"
    assert body["polarity"] > 0


def test_sentiment_negative():
    resp = client.post("/sentiment", json={"text": "terrible awful delay"})
    assert resp.json()["label"] == "negative/neutral"


def test_sentiment_rejects_empty_text():
    resp = client.post("/sentiment", json={"text": ""})
    assert resp.status_code == 422  # pydantic min_length validation


def test_aspects_endpoint():
    resp = client.post("/aspects", json={"text": "The crew were great. The food was cold."})
    assert resp.status_code == 200
    data = resp.json()["aspects"]
    assert "staff" in data and "food" in data
