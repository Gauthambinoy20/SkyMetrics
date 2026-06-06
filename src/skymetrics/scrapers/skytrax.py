"""Skytrax (airlinequality.com) review scraper.

Network I/O (``fetch_page``) is kept separate from parsing (``parse_reviews``)
so the parser can be unit-tested against fixture HTML and the fetch loop can
be mocked — tests never hit the live site.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.airlinequality.com/airline-reviews/british-airways"
DEFAULT_HEADERS = {"User-Agent": "skymetrics/0.1 (+research)"}


def page_url(page: int, page_size: int = 100) -> str:
    """Build the paginated reviews URL for a given page."""
    return f"{BASE_URL}/page/{page}/?sortby=post_date%3ADesc&pagesize={page_size}"


def fetch_page(page: int, page_size: int = 100, timeout: int = 20) -> str:
    """Fetch one page of reviews and return its raw HTML."""
    resp = requests.get(page_url(page, page_size), headers=DEFAULT_HEADERS, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def parse_reviews(html: str) -> list[str]:
    """Extract review bodies from a Skytrax reviews page.

    Reviews live in ``<div class="text_content">`` nodes.
    """
    soup = BeautifulSoup(html, "html.parser")
    return [node.get_text(strip=True) for node in soup.find_all("div", class_="text_content")]


def scrape(
    pages: int = 1,
    page_size: int = 100,
    delay: float = 1.0,
    fetcher: Callable[[int, int], str] | None = None,
) -> list[str]:
    """Scrape ``pages`` of reviews, politely sleeping between requests.

    Args:
        pages: Number of pages to fetch.
        page_size: Reviews per page.
        delay: Seconds to wait between page fetches (skipped after the last).
        fetcher: Injection point for tests; defaults to :func:`fetch_page`.

    Returns:
        A flat list of review strings across all pages.
    """
    fetch = fetcher or fetch_page
    reviews: list[str] = []
    for page in range(1, pages + 1):
        reviews.extend(parse_reviews(fetch(page, page_size)))
        if delay and page < pages:
            time.sleep(delay)
    return reviews
