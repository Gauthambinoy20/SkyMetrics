"""Live British Airways flight helper built on the OpenSky source.

Wraps fetch + parse into a single call returning a tidy list of flights with
valid coordinates, suitable for mapping.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from skymetrics.scrapers.external import fetch_opensky_states, parse_ba_flights


def live_ba_flights(
    fetcher: Callable[[], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return airborne BA flights that have usable latitude/longitude.

    Args:
        fetcher: Injection point for tests; defaults to the live OpenSky fetch.

    Returns:
        Flight dicts with non-null ``latitude`` and ``longitude``.
    """
    fetch = fetcher or fetch_opensky_states
    flights = parse_ba_flights(fetch())
    return [f for f in flights if f["latitude"] is not None and f["longitude"] is not None]
