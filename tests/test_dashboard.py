"""Tests for the dashboard data-prep functions."""

import pandas as pd
import pytest

from skymetrics.dashboard import data


class TestScoringAndSummary:
    def test_scored_reviews_from_series(self):
        scored = data.scored_reviews(pd.Series(["great flight", "awful delay"]))
        assert list(scored.columns) == ["review", "polarity", "label"]
        assert len(scored) == 2

    def test_summary_counts_and_percentage(self):
        scored = data.scored_reviews(["wonderful amazing", "terrible awful", "lovely great"])
        summary = data.sentiment_summary(scored)
        assert summary["total"] == 3
        assert summary["positive"] == 2
        assert summary["positive_pct"] == pytest.approx(66.7, abs=0.1)

    def test_summary_empty_is_zero(self):
        summary = data.sentiment_summary(data.scored_reviews([]))
        assert summary["total"] == 0
        assert summary["positive_pct"] == 0.0


class TestBenchmark:
    def test_benchmark_sorted_by_positive_share(self):
        df = pd.DataFrame(
            {
                "airline": ["BA", "BA", "Ryanair", "Ryanair"],
                "polarity": [0.5, 0.2, -0.4, -0.1],
                "label": ["positive", "positive", "negative/neutral", "negative/neutral"],
            }
        )
        result = data.benchmark_airlines(df)
        assert list(result["airline"]) == ["BA", "Ryanair"]
        assert result.iloc[0]["positive_pct"] == 100.0

    def test_benchmark_missing_columns_raises(self):
        with pytest.raises(KeyError):
            data.benchmark_airlines(pd.DataFrame({"airline": ["BA"]}))


class TestFlightsMap:
    def test_flights_to_map_df_keeps_coords(self):
        flights = [
            {"callsign": "BAW1", "latitude": 51.4, "longitude": -0.4},
            {"callsign": "BAW2", "latitude": None, "longitude": None},
        ]
        df = data.flights_to_map_df(flights)
        assert list(df.columns) == ["lat", "lon", "callsign"]
        assert len(df) == 1
        assert df.iloc[0]["callsign"] == "BAW1"

    def test_flights_to_map_df_empty(self):
        assert len(data.flights_to_map_df([])) == 0
