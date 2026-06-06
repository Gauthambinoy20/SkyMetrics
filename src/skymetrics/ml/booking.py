"""Customer booking-completion prediction.

Extracted from the Task 2 modelling notebook into pure functions so the
pipeline can be unit-tested and later served behind an API. Defaults mirror
the notebook (Random Forest, balanced classes, 80/20 split, seed 42).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

TARGET = "booking_complete"

# Categorical columns that need label encoding before training.
CATEGORICAL_COLUMNS = ("sales_channel", "trip_type", "flight_day", "route", "booking_origin")

RANDOM_STATE = 42


@dataclass
class TrainedModel:
    """A fitted model plus the encoders needed to transform new rows."""

    model: RandomForestClassifier
    encoders: dict[str, LabelEncoder]
    feature_names: list[str]


def load_bookings(path: str, encoding: str = "ISO-8859-1") -> pd.DataFrame:
    """Load the customer booking dataset from CSV.

    The Forage dataset ships in Latin-1, hence the default encoding.
    """
    return pd.read_csv(path, encoding=encoding)


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict[str, LabelEncoder]]:
    """Split a booking frame into encoded features and the target.

    Args:
        df: Raw booking records including the ``booking_complete`` target.

    Returns:
        ``(X, y, encoders)`` where categorical columns present in the frame
        have been label-encoded and ``encoders`` maps column name to its
        fitted :class:`LabelEncoder` for reuse at inference time.

    Raises:
        KeyError: If the target column is missing.
    """
    if TARGET not in df.columns:
        raise KeyError(f"missing target column {TARGET!r}")

    features = df.drop(columns=[TARGET]).copy()
    encoders: dict[str, LabelEncoder] = {}
    for col in CATEGORICAL_COLUMNS:
        if col in features.columns:
            enc = LabelEncoder()
            features[col] = enc.fit_transform(features[col].astype(str))
            encoders[col] = enc
    return features, df[TARGET], encoders


def train_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int = 100,
) -> RandomForestClassifier:
    """Fit a balanced Random Forest classifier on the given features."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    model.fit(X, y)
    return model


def evaluate(model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Return ``{"accuracy": float, "n_samples": int}`` for a held-out set."""
    preds = model.predict(X_test)
    return {"accuracy": float(accuracy_score(y_test, preds)), "n_samples": int(len(y_test))}


def train_pipeline(df: pd.DataFrame, test_size: float = 0.2) -> tuple[TrainedModel, dict]:
    """Run the full prepare → split → train → evaluate pipeline.

    Returns the trained model (with encoders) and the evaluation metrics.
    """
    X, y, encoders = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE
    )
    model = train_random_forest(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)
    trained = TrainedModel(model=model, encoders=encoders, feature_names=list(X.columns))
    return trained, metrics
