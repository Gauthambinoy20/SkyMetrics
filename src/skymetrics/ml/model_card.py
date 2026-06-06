"""Generate a Markdown model card for a trained booking model."""

from __future__ import annotations

from skymetrics.ml.booking import TrainedModel
from skymetrics.ml.importance import feature_importance


def model_card(trained: TrainedModel, metrics: dict, top_n: int = 5) -> str:
    """Render a Markdown model card from a trained model and its metrics."""
    accuracy = metrics.get("accuracy", float("nan"))
    n_samples = metrics.get("n_samples", "n/a")
    importances = feature_importance(trained)[:top_n]
    rows = "\n".join(f"| {name} | {score:.4f} |" for name, score in importances)
    return (
        "# Booking-Completion Model Card\n\n"
        "**Model:** RandomForestClassifier (balanced class weights)\n\n"
        "## Metrics\n"
        f"- Test accuracy: **{accuracy:.4f}**\n"
        f"- Evaluation samples: {n_samples}\n\n"
        "## Top features\n"
        "| Feature | Importance |\n"
        "| --- | --- |\n"
        f"{rows}\n"
    )
