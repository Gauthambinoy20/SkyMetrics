"""External operational data sources.

Public APIs that enrich the customer-voice data: live flight states from the
OpenSky Network and IAG (British Airways' parent) share price. Each source
splits the HTTP call from response parsing so the parser is unit-tested and
the fetch is mocked.
"""

from __future__ import annotations

from typing import Any

import requests

OPENSKY_URL = "https://opensky-network.org/api/states/all"
# British Airways' ICAO operator code, used to filter OpenSky callsigns.
BA_CALLSIGN_PREFIX = "BAW"


def fetch_opensky_states(timeout: int = 20) -> dict[str, Any]:
    """Fetch the current OpenSky state vectors snapshot as JSON."""
    resp = requests.get(OPENSKY_URL, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def parse_ba_flights(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract airborne British Airways flights from an OpenSky payload.

    OpenSky returns ``states`` as positional arrays; index 1 is the callsign,
    5/6 are longitude/latitude, 7 is barometric altitude.
    """
    flights = []
    for state in payload.get("states") or []:
        callsign = (state[1] or "").strip()
        if callsign.startswith(BA_CALLSIGN_PREFIX):
            flights.append(
                {
                    "callsign": callsign,
                    "longitude": state[5],
                    "latitude": state[6],
                    "altitude": state[7],
                }
            )
    return flights


def fetch_iag_quote(symbol: str = "IAG.L", timeout: int = 20) -> dict[str, Any]:
    """Fetch a Yahoo Finance quote chart for the IAG share price."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    resp = requests.get(url, headers={"User-Agent": "skymetrics/0.1"}, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def parse_iag_price(payload: dict[str, Any]) -> float:
    """Pull the regular-market price out of a Yahoo Finance chart payload.

    Raises:
        ValueError: If the payload has no usable result.
    """
    try:
        return float(payload["chart"]["result"][0]["meta"]["regularMarketPrice"])
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("no price in payload") from exc
