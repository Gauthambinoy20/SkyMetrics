"""Unit tests for external data sources — all network mocked."""

from unittest.mock import patch

import pytest

from skymetrics.scrapers import external


class TestOpenSky:
    def test_parse_filters_ba_callsigns(self):
        payload = {
            "states": [
                ["abc", "BAW123  ", "UK", 0, 0, -0.45, 51.47, 11000],
                ["def", "RYR456  ", "IE", 0, 0, -6.2, 53.4, 9000],
                ["ghi", None, "??", 0, 0, None, None, None],
            ]
        }
        flights = external.parse_ba_flights(payload)
        assert len(flights) == 1
        assert flights[0]["callsign"] == "BAW123"
        assert flights[0]["altitude"] == 11000

    def test_parse_handles_missing_states(self):
        assert external.parse_ba_flights({}) == []

    def test_fetch_is_mockable(self):
        with patch.object(external.requests, "get") as m:
            m.return_value.json.return_value = {"states": []}
            m.return_value.raise_for_status.return_value = None
            assert external.fetch_opensky_states() == {"states": []}


class TestIAGQuote:
    def test_parse_price(self):
        payload = {"chart": {"result": [{"meta": {"regularMarketPrice": 1.85}}]}}
        assert external.parse_iag_price(payload) == pytest.approx(1.85)

    def test_parse_price_missing_raises(self):
        with pytest.raises(ValueError):
            external.parse_iag_price({"chart": {"result": []}})
