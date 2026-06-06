"""Feature-importance helpers for the booking model."""

from __future__ import annotations

from skymetrics.ml.booking import TrainedModel


def feature_importance(trained: TrainedModel) -> list[tuple[str, float]]:
    """Return ``[(feature, importance), ...]`` sorted high to low.

    Uses the Random Forest's impurity-based importances aligned to the
    model's feature names.
    """
    importances = trained.model.feature_importances_
    paired = zip(trained.feature_names, (float(i) for i in importances), strict=True)
    return sorted(paired, key=lambda kv: kv[1], reverse=True)


def top_features(trained: TrainedModel, n: int = 5) -> list[str]:
    """Return the names of the ``n`` most important features."""
    return [name for name, _ in feature_importance(trained)[:n]]
