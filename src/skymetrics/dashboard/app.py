"""Streamlit dashboard for SkyMetrics.

Thin UI over skymetrics.dashboard.data and the NLP modules. Run with:
    streamlit run src/skymetrics/dashboard/app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from skymetrics.dashboard.data import flights_to_map_df, scored_reviews, sentiment_summary
from skymetrics.nlp.aspects import aspect_sentiment
from skymetrics.nlp.topics import top_terms
from skymetrics.scrapers.flights import live_ba_flights

DEFAULT_DATA = Path("notebooks/data_analysis/data/cleaned_BA_reviews.csv")


@st.cache_data
def _load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    col = "reviews" if "reviews" in df.columns else df.columns[-1]
    return scored_reviews(df[col].dropna().astype(str))


def main() -> None:  # pragma: no cover - exercised manually / via screenshots
    st.set_page_config(page_title="SkyMetrics", layout="wide")
    st.title("✈️ SkyMetrics — Airline Customer Intelligence")

    data_path = st.sidebar.text_input("Reviews CSV", str(DEFAULT_DATA))
    if not Path(data_path).exists():
        st.warning(f"Data file not found: {data_path}")
        return

    scored = _load(data_path)
    summary = sentiment_summary(scored)

    c1, c2, c3 = st.columns(3)
    c1.metric("Reviews", summary["total"])
    c2.metric("Positive", summary["positive"], f"{summary['positive_pct']}%")
    c3.metric("Negative / Neutral", summary["negative_neutral"])

    st.subheader("Sentiment distribution")
    st.bar_chart(scored["label"].value_counts())

    st.subheader("Top terms")
    terms = top_terms(scored["review"].tolist(), n=15)
    st.dataframe(pd.DataFrame(terms, columns=["term", "tf_idf"]))

    st.subheader("Aspect explorer")
    sample = st.text_area("Paste a review", "The crew were great but the food was cold.")
    if sample:
        st.json(aspect_sentiment(sample))

    st.subheader("Live British Airways flights")
    if st.button("Load live flights (OpenSky)"):
        try:
            flights = live_ba_flights()
            st.caption(f"{len(flights)} BA aircraft airborne now")
            st.map(flights_to_map_df(flights))
        except Exception as exc:  # pragma: no cover - network dependent
            st.warning(f"Could not load live flights: {exc}")


if __name__ == "__main__":  # pragma: no cover
    main()
