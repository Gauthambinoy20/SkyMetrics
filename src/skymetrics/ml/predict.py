"""Inference helpers for the booking-completion model."""

from __future__ import annotations

from typing import Any

import pandas as pd

from skymetrics.ml.booking import CATEGORICAL_COLUMNS, TrainedModel


def _encode_row(trained: TrainedModel, features: dict[str, Any]) -> pd.DataFrame:
    """Build a single-row feature frame, applying the saved label encoders.

    Unseen categorical values map to -1 so inference never crashes on a
    category that was absent at training time.
    """
    row = dict(features)
    for col in CATEGORICAL_COLUMNS:
        if col in row and col in trained.encoders:
            enc = trained.encoders[col]
            classes = set(enc.classes_)
            value = str(row[col])
            row[col] = int(enc.transform([value])[0]) if value in classes else -1
    frame = pd.DataFrame([row])
    # Align to the model's training columns, filling any missing with 0.
    return frame.reindex(columns=trained.feature_names, fill_value=0)


def predict_one(trained: TrainedModel, features: dict[str, Any]) -> dict[str, Any]:
    """Predict booking completion for one record.

    Returns ``{"booking_complete": int, "probability": float}`` where the
    probability is the model's confidence for the positive class.
    """
    X = _encode_row(trained, features)
    pred = int(trained.model.predict(X)[0])
    proba = trained.model.predict_proba(X)[0]
    positive_index = list(trained.model.classes_).index(1) if 1 in trained.model.classes_ else -1
    probability = float(proba[positive_index]) if positive_index >= 0 else 0.0
    return {"booking_complete": pred, "probability": round(probability, 4)}
