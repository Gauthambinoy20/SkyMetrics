"""Unit tests for the SQLite DB layer (in-memory, no disk)."""

import pandas as pd
import pytest

from skymetrics.db import loaders, schema


@pytest.fixture
def conn():
    c = schema.connect(":memory:")
    schema.init_db(c)
    yield c
    c.close()


class TestSchema:
    def test_tables_created(self, conn):
        names = {
            r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert {"reviews", "bookings", "metrics"} <= names

    def test_init_is_idempotent(self, conn):
        schema.init_db(conn)  # second call must not raise
        assert loaders.count_rows(conn, "reviews") == 0


class TestLoaders:
    def test_load_reviews_with_default_source(self, conn):
        df = pd.DataFrame({"review": ["great", "awful"], "polarity": [0.8, -0.5]})
        n = loaders.load_reviews(conn, df, source="trustpilot")
        assert n == 2
        row = conn.execute("SELECT source FROM reviews LIMIT 1").fetchone()
        assert row["source"] == "trustpilot"

    def test_load_bookings(self, conn):
        df = pd.DataFrame({"route": ["AKLDEL"], "purchase_lead": [30], "booking_complete": [1]})
        assert loaders.load_bookings(conn, df) == 1
        assert loaders.count_rows(conn, "bookings") == 1

    def test_insert_with_no_known_columns_raises(self, conn):
        with pytest.raises(ValueError):
            loaders.load_bookings(conn, pd.DataFrame({"unrelated": [1]}))

    def test_count_rows_rejects_bad_identifier(self, conn):
        with pytest.raises(ValueError):
            loaders.count_rows(conn, "reviews; DROP TABLE reviews")
