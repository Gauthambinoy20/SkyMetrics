"""FastAPI application exposing SkyMetrics NLP services.

Endpoints are intentionally lightweight and model-download-free: sentiment
uses the baseline TextBlob scorer and aspect breakdown uses the keyword
aspect model, so the API starts instantly and stays testable.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from skymetrics import __version__
from skymetrics.llm.client import LLMConfigError
from skymetrics.llm.summarize import summarize_reviews
from skymetrics.ml.persistence import load_model
from skymetrics.ml.predict import predict_one
from skymetrics.nlp import aspects
from skymetrics.nlp.sentiment import classify_sentiment, polarity_score
from skymetrics.scrapers.flights import live_ba_flights

MODEL_PATH = os.environ.get("SKYMETRICS_MODEL_PATH", "models/booking.joblib")


@lru_cache(maxsize=1)
def _get_model():
    """Lazily load the persisted booking model (cached)."""
    return load_model(MODEL_PATH)


app = FastAPI(title="SkyMetrics API", version=__version__)


class ReviewIn(BaseModel):
    """Request body carrying a single review text."""

    text: str = Field(..., min_length=1, description="Review text to analyse")


class SentimentOut(BaseModel):
    polarity: float
    label: str


class ReviewsIn(BaseModel):
    """Request body carrying a batch of reviews to summarise."""

    reviews: list[str] = Field(..., min_length=1, description="Reviews to summarise")


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok", "version": __version__}


@app.post("/sentiment", response_model=SentimentOut)
def sentiment(body: ReviewIn) -> SentimentOut:
    """Return baseline polarity and label for a review."""
    polarity = polarity_score(body.text)
    return SentimentOut(polarity=polarity, label=classify_sentiment(polarity))


@app.post("/aspects")
def aspect_breakdown(body: ReviewIn) -> dict[str, dict[str, float]]:
    """Return per-aspect sentiment for a review."""
    return {"aspects": aspects.aspect_sentiment(body.text)}


@app.get("/flights/live")
def flights_live() -> dict[str, object]:
    """Return airborne British Airways flights from the OpenSky Network."""
    flights = live_ba_flights()
    return {"count": len(flights), "flights": flights}


@app.post("/predict")
def predict(features: dict[str, Any]) -> dict[str, Any]:
    """Predict booking completion for a record of booking features.

    Returns a 503 if no trained model artifact is available.
    """
    try:
        model = _get_model()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"model not available at {MODEL_PATH}; run scripts/train_model.py",
        ) from exc
    return predict_one(model, features)


@app.post("/summary")
def summary(body: ReviewsIn) -> dict[str, str]:
    """Summarise a batch of reviews with the LLM (top complaints/praise).

    Returns a 503 if no LLM API key is configured.
    """
    try:
        return {"summary": summarize_reviews(body.reviews)}
    except LLMConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
