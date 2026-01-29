"""
Training script for the weather prediction ML pipeline.

This script orchestrates the end‑to‑end process of reading cleaned
weather data, preparing features and targets, training a model and
persisting it to disk. It logs progress at key steps to aid
troubleshooting and reproducibility.

Example::

    python src/train.py --data data/weather_hourly.csv --model models/weather_rf.joblib

You can adjust the test split and hyperparameters via command line
arguments. See ``--help`` for full usage.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

try:
    # Prefer package-relative imports when executed via ``python -m``.
    from . import model  # type: ignore[import]
    from .utils import setup_logging  # type: ignore[import]
except ImportError:
    # Fall back to absolute imports when running this file directly.
    import model  # type: ignore[import]
    from utils import setup_logging  # type: ignore[import]


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for the training script."""
    parser = argparse.ArgumentParser(
        description="Train a machine learning model to predict future temperatures based on weather observations."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/weather_hourly.csv"),
        help="Path to the input CSV with weather observations",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/weather_model.joblib"),
        help="Output path for the trained model",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data reserved for testing (default: 0.2)",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="Number of trees in the RandomForest (default: 200)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        default="INFO",
        help="Logging verbosity level (default: INFO)",
    )
    return parser.parse_args(args)


def main() -> None:
    """Entry point for model training."""
    args = parse_args()
    setup_logging(getattr(logging, args.log_level))
    logger = logging.getLogger(__name__)

    # Load the dataset
    try:
        df = pd.read_csv(args.data, parse_dates=[0], index_col=0)
    except FileNotFoundError:
        logger.error("Input data file not found: %s", args.data)
        return
    except Exception as exc:
        logger.exception("Failed to read input CSV: %s", exc)
        return

    # Prepare features and target
    try:
        X, y = model.prepare_features(df)
    except Exception as exc:
        logger.exception("Failed to prepare features: %s", exc)
        return

    # Split data
    X_train, X_test, y_train, y_test = model.split_data(
        X, y, test_size=args.test_size, random_state=42
    )

    # Train model
    try:
        reg = model.train_random_forest(
            X_train,
            y_train,
            n_estimators=args.n_estimators,
            random_state=42,
        )
    except Exception as exc:
        logger.exception("Failed to train model: %s", exc)
        return

    # Evaluate model
    try:
        model.evaluate_model(reg, X_test, y_test)
    except Exception as exc:
        logger.exception("Failed to evaluate model: %s", exc)

    # Persist the trained model
    try:
        model.save_model(reg, args.model)
    except Exception as exc:
        logger.exception("Failed to save model: %s", exc)
        return


if __name__ == "__main__":
    main()
