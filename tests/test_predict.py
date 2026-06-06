"""Tests for booking inference helper and the /predict endpoint."""

from unittest.mock import patch

from fastapi.testclient import TestClient

import skymetrics.api.app as api
from skymetrics.api.app import app
from skymetrics.ml import booking, predict
from tests.test_booking import _sample_frame

client = TestClient(app)


def _trained():
    trained, _ = booking.train_pipeline(_sample_frame(), test_size=0.25)
    return trained


_RECORD = {
    "num_passengers": 2,
    "sales_channel": "Internet",
    "trip_type": "RoundTrip",
    "purchase_lead": 80,
    "length_of_stay": 5,
    "flight_hour": 9,
    "flight_day": "Mon",
    "route": "AKLDEL",
    "booking_origin": "India",
    "wants_extra_baggage": 1,
    "wants_preferred_seat": 1,
    "wants_in_flight_meals": 1,
    "flight_duration": 7.0,
}


class TestPredictOne:
    def test_returns_label_and_probability(self):
        result = predict.predict_one(_trained(), _RECORD)
        assert result["booking_complete"] in (0, 1)
        assert 0.0 <= result["probability"] <= 1.0

    def test_unseen_category_does_not_crash(self):
        record = {**_RECORD, "route": "ZZZUNKNOWN", "booking_origin": "Atlantis"}
        result = predict.predict_one(_trained(), record)
        assert "probability" in result


class TestPredictEndpoint:
    def test_predict_with_loaded_model(self):
        api._get_model.cache_clear()
        with patch.object(api, "load_model", return_value=_trained()):
            resp = client.post("/predict", json=_RECORD)
        api._get_model.cache_clear()
        assert resp.status_code == 200
        assert "booking_complete" in resp.json()

    def test_predict_returns_503_without_model(self):
        api._get_model.cache_clear()
        with patch.object(api, "load_model", side_effect=FileNotFoundError):
            resp = client.post("/predict", json=_RECORD)
        api._get_model.cache_clear()
        assert resp.status_code == 503
