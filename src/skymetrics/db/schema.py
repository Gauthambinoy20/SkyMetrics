"""SQLite schema and connection helpers for SkyMetrics.

A thin, dependency-free persistence layer (stdlib ``sqlite3``) holding the
data the dashboard and API read from: customer reviews, booking records and
derived operational metrics.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS reviews (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source      TEXT NOT NULL,
    review      TEXT NOT NULL,
    rating      REAL,
    polarity    REAL,
    label       TEXT,
    review_date TEXT
);

CREATE TABLE IF NOT EXISTS bookings (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    route             TEXT,
    booking_origin    TEXT,
    purchase_lead     INTEGER,
    length_of_stay    INTEGER,
    flight_duration   REAL,
    booking_complete  INTEGER
);

CREATE TABLE IF NOT EXISTS metrics (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    name    TEXT NOT NULL,
    value   REAL NOT NULL,
    period  TEXT,
    source  TEXT
);

CREATE INDEX IF NOT EXISTS idx_reviews_source ON reviews(source);
CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(name);
"""


def connect(db_path: str = "skymetrics.db") -> sqlite3.Connection:
    """Open a SQLite connection with row access by column name.

    ``":memory:"`` is accepted for ephemeral test databases.
    """
    if db_path != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Create all tables and indexes if they do not already exist."""
    conn.executescript(SCHEMA)
    conn.commit()
