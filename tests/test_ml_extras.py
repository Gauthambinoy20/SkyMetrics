"""Tests for model persistence, feature importance and the model card."""

import pandas as pd

from skymetrics.ml import booking, importance, model_card, persistence
from tests.test_booking import _sample_frame


def _trained():
    return booking.train_pipeline(_sample_frame(), test_size=0.25)


class TestPersistence:
    def test_round_trip_preserves_predictions(self, tmp_path):
        trained, _ = _trained()
        path = persistence.save_model(trained, str(tmp_path / "m.joblib"))
        loaded = persistence.load_model(path)
        X = pd.DataFrame([dict.fromkeys(trained.feature_names, 1)])
        assert list(loaded.model.predict(X)) == list(trained.model.predict(X))
        assert loaded.feature_names == trained.feature_names

    def test_load_missing_raises(self, tmp_path):
        try:
            persistence.load_model(str(tmp_path / "nope.joblib"))
            raise AssertionError("expected FileNotFoundError")
        except FileNotFoundError:
            pass


class TestImportance:
    def test_importances_sorted_and_complete(self):
        trained, _ = _trained()
        ranked = importance.feature_importance(trained)
        assert len(ranked) == len(trained.feature_names)
        scores = [s for _, s in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_top_features_count(self):
        trained, _ = _trained()
        assert len(importance.top_features(trained, n=3)) == 3


class TestModelCard:
    def test_card_contains_metrics_and_features(self):
        trained, metrics = _trained()
        card = model_card.model_card(trained, metrics)
        assert "Model Card" in card
        assert "Test accuracy" in card
        assert "| Feature | Importance |" in card
