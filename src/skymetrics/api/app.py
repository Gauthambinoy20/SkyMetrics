"""FastAPI application exposing SkyMetrics NLP services.

Endpoints are intentionally lightweight and model-download-free: sentiment
uses the baseline TextBlob scorer and aspect breakdown uses the keyword
aspect model, so the API starts instantly and stays testable.
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from skymetrics import __version__
from skymetrics.nlp import aspects
from skymetrics.nlp.sentiment import classify_sentiment, polarity_score
from skymetrics.scrapers.flights import live_ba_flights

app = FastAPI(title="SkyMetrics API", version=__version__)


class ReviewIn(BaseModel):
    """Request body carrying a single review text."""

    text: str = Field(..., min_length=1, description="Review text to analyse")


class SentimentOut(BaseModel):
    polarity: float
    label: str


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
