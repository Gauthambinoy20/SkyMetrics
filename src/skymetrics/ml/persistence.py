"""Persist and reload trained booking models."""

from __future__ import annotations

from pathlib import Path

import joblib

from skymetrics.ml.booking import TrainedModel


def save_model(trained: TrainedModel, path: str) -> str:
    """Serialise a :class:`TrainedModel` (model + encoders) to ``path``."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(trained, path)
    return path


def load_model(path: str) -> TrainedModel:
    """Load a :class:`TrainedModel` previously saved with :func:`save_model`."""
    if not Path(path).exists():
        raise FileNotFoundError(path)
    return joblib.load(path)
