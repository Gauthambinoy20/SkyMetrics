"""Unit tests for the Skytrax scraper — network fully mocked."""

from unittest.mock import patch

from skymetrics.scrapers import skytrax

_FIXTURE_HTML = """
<html><body>
  <article>
    <div class="text_content">✅ Trip Verified | Excellent crew and smooth flight.</div>
  </article>
  <article>
    <div class="text_content">Not Verified | Delayed three hours, no apology.</div>
  </article>
  <div class="other">ignore me</div>
</body></html>
"""


class TestParsing:
    def test_parse_extracts_review_bodies(self):
        reviews = skytrax.parse_reviews(_FIXTURE_HTML)
        assert len(reviews) == 2
        assert "Excellent crew" in reviews[0]
        assert "Delayed three hours" in reviews[1]

    def test_parse_empty_html_returns_empty(self):
        assert skytrax.parse_reviews("<html></html>") == []

    def test_page_url_contains_page_and_size(self):
        url = skytrax.page_url(3, page_size=50)
        assert "/page/3/" in url and "pagesize=50" in url


class TestScrapeLoop:
    def test_scrape_uses_injected_fetcher_no_network(self):
        calls = []

        def fake_fetch(page, page_size):
            calls.append(page)
            return _FIXTURE_HTML

        reviews = skytrax.scrape(pages=3, delay=0, fetcher=fake_fetch)
        assert calls == [1, 2, 3]
        assert len(reviews) == 6  # 2 reviews per page x 3 pages

    def test_scrape_default_fetcher_is_mockable(self):
        with patch.object(skytrax, "fetch_page", return_value=_FIXTURE_HTML) as m:
            reviews = skytrax.scrape(pages=1, delay=0)
        m.assert_called_once()
        assert len(reviews) == 2
