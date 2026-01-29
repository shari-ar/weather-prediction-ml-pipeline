"""
Model utilities for the weather prediction ML pipeline.

This module defines helper functions to prepare feature matrices and
targets, train scikit‑learn models and persist them to disk. Keeping
model logic separate from the training script makes it easy to reuse
models from other modules (e.g. for inference or evaluation).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def prepare_features(df: pd.DataFrame, target_column: str = "temp") -> Tuple[pd.DataFrame, pd.Series]:
    """Generate feature matrix ``X`` and target vector ``y`` from the input data.

    The function selects a subset of meteorological features deemed
    predictive of the target variable (temperature by default). The
    target is shifted by one timestep (e.g. one hour ahead) so that we
    predict the next hour's temperature given the current conditions.

    Args:
        df: DataFrame indexed by datetime with weather variables as columns.
        target_column: Name of the column to predict. Defaults to ``"temp"``.

    Returns:
        A tuple ``(X, y)`` where ``X`` is the feature matrix and ``y`` is
        the target vector.
    """
    # Define which columns to use as predictors. Additional variables can
    # easily be added here. If a column is missing, we log a warning and
    # skip it.
    candidate_features = [
        "temp",
        "humidity",
        "pressure",
        "wind_speed",
        "wind_gust",
        "dew_point",
        "clouds",
    ]
    existing_features = [col for col in candidate_features if col in df.columns]
    missing = set(candidate_features) - set(existing_features)
    if missing:
        logger.warning("Missing feature columns: %s", ", ".join(sorted(missing)))
    X = df[existing_features].copy()
    # Shift the target by -1 (next timestep). dropna removes the last row,
    # which has no future value.
    y = df[target_column].shift(-1).dropna()
    X = X.iloc[:-1]  # align lengths after shifting
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the dataset into training and test subsets.

    Args:
        X: Feature matrix.
        y: Target vector.
        test_size: Fraction of the data to use for testing. Defaults to 0.2.
        random_state: Seed for the random number generator. Defaults to 42.

    Returns:
        ``(X_train, X_test, y_train, y_test)``
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    n_estimators: int = 200,
    random_state: int = 42,
    n_jobs: int = -1,
) -> RandomForestRegressor:
    """Train a RandomForestRegressor on the provided data.

    Args:
        X_train: Training features.
        y_train: Training targets.
        n_estimators: Number of trees in the forest. More trees can
            improve performance but increase training time. Defaults to 200.
        random_state: Seed for reproducibility. Defaults to 42.
        n_jobs: Number of parallel jobs. Use ``-1`` to use all cores.

    Returns:
        Trained RandomForestRegressor instance.
    """
    logger.info("Training RandomForest with %d estimators", n_estimators)
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    model.fit(X_train, y_train)
    logger.info("Model training complete")
    return model


def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    """Evaluate a regression model using the coefficient of determination (R²).

    Args:
        model: Trained regressor implementing ``score(X, y)``.
        X_test: Test feature matrix.
        y_test: Test target vector.

    Returns:
        The R² score on the test set. 1.0 represents a perfect fit.
    """
    score = model.score(X_test, y_test)
    logger.info("Model R² score: %.3f", score)
    return score


def save_model(model: Any, file_path: Path) -> None:
    """Persist a trained model to disk using joblib.

    Args:
        model: Trained scikit‑learn model or pipeline.
        file_path: Destination path for the serialized model. Parent
            directories will be created if needed.
    """
    file_path = file_path.expanduser().resolve()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, file_path)
    logger.info("Saved model to %s", file_path)


def load_model(file_path: Path) -> Any:
    """Load a serialized model from disk.

    Args:
        file_path: Path to the saved model.

    Returns:
        The deserialized model.
    """
    model = joblib.load(file_path)
    logger.info("Loaded model from %s", file_path)
    return model
