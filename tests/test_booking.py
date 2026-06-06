"""Unit tests for the booking-prediction module."""

import pandas as pd
import pytest

from skymetrics.ml import booking


def _sample_frame(n: int = 60) -> pd.DataFrame:
    """Small synthetic booking frame with a learnable signal.

    ``purchase_lead`` is correlated with the target so a model can exceed
    chance accuracy on a tiny set without hitting the real dataset.
    """
    rows = []
    for i in range(n):
        complete = i % 2
        rows.append(
            {
                "num_passengers": 1 + (i % 3),
                "sales_channel": "Internet" if i % 2 else "Mobile",
                "trip_type": "RoundTrip" if i % 3 else "OneWay",
                "purchase_lead": 5 + complete * 100,
                "length_of_stay": 3 + (i % 10),
                "flight_hour": i % 24,
                "flight_day": ["Mon", "Tue", "Wed"][i % 3],
                "route": ["AKLDEL", "AKLKUL"][i % 2],
                "booking_origin": ["New Zealand", "India"][i % 2],
                "wants_extra_baggage": i % 2,
                "wants_preferred_seat": i % 2,
                "wants_in_flight_meals": i % 2,
                "flight_duration": 5.0 + (i % 5),
                "booking_complete": complete,
            }
        )
    return pd.DataFrame(rows)


class TestPrepareFeatures:
    def test_encodes_categoricals_to_numeric(self):
        X, y, encoders = booking.prepare_features(_sample_frame())
        for col in ("sales_channel", "trip_type", "route", "booking_origin", "flight_day"):
            assert pd.api.types.is_numeric_dtype(X[col])
        assert set(encoders) == set(booking.CATEGORICAL_COLUMNS)

    def test_target_excluded_from_features(self):
        X, y, _ = booking.prepare_features(_sample_frame())
        assert booking.TARGET not in X.columns
        assert y.name == booking.TARGET

    def test_missing_target_raises(self):
        df = _sample_frame().drop(columns=[booking.TARGET])
        with pytest.raises(KeyError):
            booking.prepare_features(df)


class TestTrainAndEvaluate:
    def test_train_returns_fitted_model(self):
        X, y, _ = booking.prepare_features(_sample_frame())
        model = booking.train_random_forest(X, y, n_estimators=10)
        assert len(model.predict(X)) == len(y)

    def test_evaluate_returns_accuracy_in_range(self):
        X, y, _ = booking.prepare_features(_sample_frame())
        model = booking.train_random_forest(X, y, n_estimators=10)
        metrics = booking.evaluate(model, X, y)
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert metrics["n_samples"] == len(y)

    def test_pipeline_end_to_end(self):
        trained, metrics = booking.train_pipeline(_sample_frame(), test_size=0.25)
        assert trained.feature_names
        assert set(trained.encoders) == set(booking.CATEGORICAL_COLUMNS)
        assert 0.0 <= metrics["accuracy"] <= 1.0
