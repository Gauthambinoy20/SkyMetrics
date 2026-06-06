"""Transformer-based sentiment with a lazily loaded HuggingFace pipeline.

The heavy model is only built on first use; an injectable ``pipeline`` makes
the wrapper unit-testable without downloading any weights.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

DEFAULT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"


class TransformerSentiment:
    """Wraps a text-classification pipeline returning normalised results."""

    def __init__(
        self,
        pipeline: Callable[[str], list[dict[str, Any]]] | None = None,
        model: str = DEFAULT_MODEL,
    ):
        self._pipeline = pipeline
        self._model = model

    def _ensure_pipeline(self) -> Callable[[str], list[dict[str, Any]]]:
        """Build the HuggingFace pipeline on first use (import kept lazy)."""
        if self._pipeline is None:
            from transformers import pipeline as hf_pipeline  # noqa: PLC0415

            self._pipeline = hf_pipeline("sentiment-analysis", model=self._model)
        return self._pipeline

    def classify(self, text: str) -> dict[str, Any]:
        """Classify one text into ``{"label": str, "score": float}``.

        The label is lower-cased; an empty string yields a neutral result
        without invoking the model.
        """
        if not text or not text.strip():
            return {"label": "neutral", "score": 0.0}
        result = self._ensure_pipeline()(text)[0]
        return {"label": str(result["label"]).lower(), "score": float(result["score"])}
