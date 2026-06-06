# British Airways Data Science Virtual Experience

[![Python](https://img.shields.io/badge/python-3.9+-blue?logo=python)](https://python.org)
[![Jupyter](https://img.shields.io/badge/jupyter-notebook-orange?logo=jupyter)](https://jupyter.org)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-blue?logo=scikit-learn)](https://scikit-learn.org)
[![Forage](https://img.shields.io/badge/Forage-Certified-brightgreen)](https://www.theforage.com/virtual-experience/NjynCWzGSaWXQCxSX/british-airways/data-science-yqoz)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> British Airways Data Science Virtual Experience Programme — NLP sentiment analysis of customer reviews and predictive ML modelling of customer buying behaviour.

---

## Overview

This project was completed as part of the **British Airways Data Science Virtual Experience Programme** on Forage. British Airways (BA) is the flag carrier airline of the United Kingdom, serving over 30 million passengers annually. The programme covers two real-world data science tasks that mirror the kind of work BA's data science team performs.

---

## Programme Tasks

### Task 1: Web Scraping & Customer Sentiment Analysis

Scraped **3,917 customer reviews** from [Skytrax](https://www.airlinequality.com/airline-reviews/british-airways) (60 pages × 300 reviews) to understand how customers feel about British Airways.

**Process:**
- Web scraping with `BeautifulSoup` + `requests`
- Text cleaning and preprocessing (removing noise, stopwords)
- Sentiment scoring with `TextBlob`
- Word frequency analysis and word cloud generation

**Key Findings:**

| Sentiment | Count | Percentage |
|---|---|---|
| Positive | 2,707 | 69% |
| Negative/Neutral | 1,210 | 31% |

**Visualisations produced:**
- Sentiment distribution bar chart
- Word cloud of most frequent review terms
- Top 10 most common words across all reviews
- Word count distribution

---

### Task 2: Predictive Modelling — Customer Buying Behaviour

Built machine learning models to predict whether a customer will **complete a flight booking**, using a dataset of **50,000 customer records** with 14 features.

**Dataset features:** `num_passengers`, `sales_channel`, `trip_type`, `purchase_lead`, `length_of_stay`, `flight_hour`, `flight_day`, `route`, `booking_origin`, `wants_extra_baggage`, `wants_preferred_seat`, `wants_in_flight_meals`, `flight_duration`, `booking_complete`

**Models trained:**

| Model | Test Accuracy | Notes |
|---|---|---|
| Random Forest | **85.6%** | 100 estimators, class_weight='balanced' |
| XGBoost | Compared | Gradient boosting baseline |
| Neural Network | Compared | Sequential Keras model, 20 epochs |

**Cross-validation (5-fold):** 56.3% ± 18.1%

**Feature Importance (SHAP analysis):**
The XGBoost SHAP values revealed that `purchase_lead`, `flight_duration`, and `length_of_stay` were the strongest predictors of booking completion.

**Visualisations produced:**
- Random Forest vs XGBoost accuracy comparison
- Neural network training accuracy curve
- XGBoost feature importance plot
- SHAP summary plot and SHAP importance rankings

---

## Analysis Output

Key visualisations are saved in `ANALYSIS_OUTPUT/`:

| File | Description |
|---|---|
| `SENTIMENT DISTRIBUTION.png` | Positive vs negative review breakdown |
| `WORD CLOUD – REVIEWS.png` | Most frequent words in customer reviews |
| `TOP 10 WORDS.png` | Bar chart of top 10 review terms |
| `WORD COUNT DISTRIBUTION.png` | Distribution of review lengths |
| `RANDOM FOREST AND XGBOOST.png` | Model accuracy comparison |
| `NEURAL NETWORK TRAINING ACCURACY.png` | Training curve over 20 epochs |
| `XGBOOST IMPORTANCE IN PREDICTING BOOKING COMPLETION.png` | Feature importances |
| `XGBOOST SHAP IMPORTANCE.png` | SHAP-based feature ranking |
| `XGBOOST SHAP SUMMARY PLOT.png` | SHAP beeswarm summary |

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.9+ |
| Data Analysis | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost, TensorFlow/Keras |
| Explainability | SHAP |
| NLP | NLTK, TextBlob |
| Visualisation | Matplotlib, Seaborn, WordCloud |
| Web Scraping | BeautifulSoup4, Requests |
| Notebook | Jupyter |

---

## Project Structure

```
british-airways-data-science/
├── Data Analysis/
│   ├── getting_started.ipynb       # Task 1: Scraping + sentiment analysis
│   └── data/
│       ├── BA_reviews.csv          # Raw scraped reviews
│       ├── cleaned_BA_reviews.csv  # Cleaned reviews
│       └── BA_reviews_tokenized.csv
├── AI Modelling/
│   ├── Getting Started - Task 2 Prediction Analysis.ipynb  # Task 2: ML modelling
│   └── data/
│       └── customer_booking.csv    # 50,000 booking records
├── ANALYSIS_OUTPUT/                # All generated charts and plots
├── CERTIFICATE/
│   └── British_Airways_AI_Certificate.pdf
├── requirements.txt
└── README.md
```

---

## Getting Started

```bash
git clone https://github.com/gauthambinoy/ba-insights-airline-analytics.git
cd ba-insights-airline-analytics

pip install -r requirements.txt

# Task 1 — Sentiment Analysis
jupyter notebook "Data Analysis/getting_started.ipynb"

# Task 2 — Predictive Modelling
jupyter notebook "AI Modelling/Getting Started - Task 2 Prediction Analysis.ipynb"
```

---

## Certificate

Completed and certified by [British Airways via Forage](https://www.theforage.com/virtual-experience/NjynCWzGSaWXQCxSX/british-airways/data-science-yqoz). Certificate available in `CERTIFICATE/`.

---

## License

[MIT](LICENSE) © Gautham Binoy
