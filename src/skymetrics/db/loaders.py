"""Load DataFrames into the SkyMetrics SQLite tables."""

from __future__ import annotations

import sqlite3

import pandas as pd

_REVIEW_COLUMNS = ("source", "review", "rating", "polarity", "label", "review_date")
_BOOKING_COLUMNS = (
    "route",
    "booking_origin",
    "purchase_lead",
    "length_of_stay",
    "flight_duration",
    "booking_complete",
)
_ALLOWED_TABLES = frozenset({"reviews", "bookings", "metrics"})


def _insert(
    conn: sqlite3.Connection, table: str, columns: tuple[str, ...], df: pd.DataFrame
) -> int:
    """Insert the intersecting columns of ``df`` into ``table``; return row count."""
    if table not in _ALLOWED_TABLES:
        raise ValueError(f"unknown table {table!r}")
    present = [c for c in columns if c in df.columns]
    if not present:
        raise ValueError(f"none of the expected columns {columns} are in the frame")
    placeholders = ", ".join("?" for _ in present)
    # Table is allowlisted above and columns come from module-level constants
    # (never user input); values are bound via placeholders.
    sql = f"INSERT INTO {table} ({', '.join(present)}) VALUES ({placeholders})"  # nosec B608
    rows = df[present].itertuples(index=False, name=None)
    cur = conn.executemany(sql, rows)
    conn.commit()
    return cur.rowcount


def load_reviews(conn: sqlite3.Connection, df: pd.DataFrame, source: str = "skytrax") -> int:
    """Insert reviews, defaulting the ``source`` column when absent."""
    frame = df.copy()
    if "source" not in frame.columns:
        frame["source"] = source
    return _insert(conn, "reviews", _REVIEW_COLUMNS, frame)


def load_bookings(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """Insert booking records (only known columns are persisted)."""
    return _insert(conn, "bookings", _BOOKING_COLUMNS, df)


def count_rows(conn: sqlite3.Connection, table: str) -> int:
    """Return the number of rows in ``table``."""
    if not table.isidentifier():
        raise ValueError(f"invalid table name {table!r}")
    # table validated as a bare identifier immediately above.
    return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])  # nosec B608
