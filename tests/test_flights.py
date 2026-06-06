"""Tests for the live-flights helper and endpoint — network mocked."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from skymetrics.api.app import app
from skymetrics.scrapers import flights

client = TestClient(app)

_PAYLOAD = {
    "states": [
        ["a", "BAW100  ", "UK", 0, 0, -0.45, 51.47, 11000],
        ["b", "BAW200  ", "UK", 0, 0, None, None, 9000],  # no coords -> dropped
        ["c", "RYR999  ", "IE", 0, 0, -6.2, 53.4, 9000],  # not BA -> dropped
    ]
}


class TestHelper:
    def test_filters_to_ba_with_coordinates(self):
        result = flights.live_ba_flights(fetcher=lambda: _PAYLOAD)
        assert len(result) == 1
        assert result[0]["callsign"] == "BAW100"

    def test_empty_payload(self):
        assert flights.live_ba_flights(fetcher=lambda: {"states": []}) == []


class TestEndpoint:
    def test_flights_live_endpoint(self):
        with patch("skymetrics.api.app.live_ba_flights", return_value=[{"callsign": "BAW100"}]):
            resp = client.get("/flights/live")
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 1
        assert body["flights"][0]["callsign"] == "BAW100"
