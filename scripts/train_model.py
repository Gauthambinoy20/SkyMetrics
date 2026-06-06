"""Train the booking-completion model and persist it.

Usage:
    python scripts/train_model.py [csv_path] [out_path]
"""

from __future__ import annotations

import sys

from skymetrics.ml.booking import load_bookings, train_pipeline
from skymetrics.ml.model_card import model_card
from skymetrics.ml.persistence import save_model

DEFAULT_CSV = "notebooks/ai_modelling/data/customer_booking.csv"
DEFAULT_OUT = "models/booking.joblib"


def main(csv_path: str = DEFAULT_CSV, out_path: str = DEFAULT_OUT) -> None:
    df = load_bookings(csv_path)
    trained, metrics = train_pipeline(df)
    save_model(trained, out_path)
    print(f"Saved model to {out_path}")
    print(model_card(trained, metrics))


if __name__ == "__main__":
    main(*sys.argv[1:])
