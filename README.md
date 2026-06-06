# SkyMetrics — Airline Customer Intelligence

[![CI](https://github.com/Gauthambinoy20/SkyMetrics/actions/workflows/ci.yml/badge.svg)](https://github.com/Gauthambinoy20/SkyMetrics/actions/workflows/ci.yml)
[![CodeQL](https://github.com/Gauthambinoy20/SkyMetrics/actions/workflows/codeql.yml/badge.svg)](https://github.com/Gauthambinoy20/SkyMetrics/actions/workflows/codeql.yml)
[![Python](https://img.shields.io/badge/python-3.12-blue?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> An airline-customer-intelligence platform for British Airways: review
> sentiment (baseline + transformer), aspect-based analysis, topic mining,
> booking-completion prediction, live flight and share-price enrichment — served
> through a FastAPI API and a Streamlit dashboard.

SkyMetrics began as the **British Airways Data Science Virtual Experience**
(Forage) and has been expanded into a fully tested, containerised product with
a proper package, API, dashboard, persistence layer and CI/CD pipeline.

---

## What it does

| Capability | Module |
| --- | --- |
| Baseline sentiment (TextBlob) | `nlp/sentiment.py` |
| Transformer sentiment (DistilBERT, lazy) | `nlp/transformer.py` |
| Aspect-based sentiment (seat, food, staff, delay…) | `nlp/aspects.py` |
| Topic / keyword mining (TF-IDF) | `nlp/topics.py` |
| Booking-completion model (Random Forest) | `ml/booking.py` |
| Feature importance + model card | `ml/importance.py`, `ml/model_card.py` |
| Model persistence | `ml/persistence.py` |
| Skytrax review scraper | `scrapers/skytrax.py` |
| Live flights + IAG share price | `scrapers/external.py` |
| SQLite persistence | `db/` |
| REST API | `api/app.py` |
| Dashboard | `dashboard/app.py` |

## Architecture

See [docs/architecture.md](docs/architecture.md) for the full set of diagrams
(architecture, data-flow, sequence, ER). High level:

```text
sources → scrapers → SQLite → nlp/ml → FastAPI + Streamlit
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
```

## Run

```bash
# API
uvicorn skymetrics.api.app:app --reload
# Dashboard
streamlit run src/skymetrics/dashboard/app.py
# Everything in containers
docker compose up --build
```

Example API call:

```bash
curl -X POST localhost:8000/sentiment -H 'Content-Type: application/json' \
  -d '{"text":"wonderful crew, smooth flight"}'
# {"polarity":0.9,"label":"positive"}
```

## Test

```bash
pytest --cov=skymetrics --cov-report=term-missing
```

59 tests · 91% coverage · all external calls mocked (no live network in tests).

## Results & screenshots

Generated from real runs (see `ANALYSIS_OUTPUT/`):

| Sentiment distribution | Top review terms |
| --- | --- |
| ![Sentiment](ANALYSIS_OUTPUT/SENTIMENT%20DISTRIBUTION.png) | ![Top words](ANALYSIS_OUTPUT/TOP%2010%20WORDS.png) |

| Random Forest vs XGBoost | Neural-net training |
| --- | --- |
| ![RF vs XGB](ANALYSIS_OUTPUT/RANDOM%20FOREST%20AND%20XGBOOST.png) | ![NN](ANALYSIS_OUTPUT/NEURAL%20NETWORK%20TRAINING%20ACCURACY.png) |

SHAP feature attribution:

![SHAP](ANALYSIS_OUTPUT/XGBOOST%20SHAP%20SUMMARY%20PLOT.png)

**Key findings:** 69% of 3,917 Skytrax reviews are positive; the Random Forest
predicts booking completion at ~85.6% test accuracy, with `purchase_lead`,
`flight_duration` and `length_of_stay` the strongest drivers.

## Roadmap

See [ROADMAP.md](ROADMAP.md). Tests green, CI green, fully containerised.

## ✍️ TODO: my words

> Why I built it this way, what I learned, and the trade-offs I'd defend in an
> interview — to be written in my own words.

## License

MIT — see [LICENSE](LICENSE).
