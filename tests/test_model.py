"""Tests for the model module."""

import numpy as np
import pandas as pd

from src import model


def _generate_df(num_rows: int = 10) -> pd.DataFrame:
    """Generate a synthetic DataFrame for testing."""
    index = pd.date_range("2025-01-01", periods=num_rows, freq="H", tz="UTC")
    data = {
        "temp": np.linspace(0, num_rows - 1, num_rows),
        "humidity": np.linspace(50, 50 + (num_rows - 1), num_rows),
        "pressure": np.linspace(1000, 1000 + (num_rows - 1), num_rows),
        "wind_speed": np.linspace(1, 1 + (num_rows - 1), num_rows),
        "wind_gust": np.linspace(2, 2 * num_rows, num_rows),
        "dew_point": np.linspace(-5, -5 + (num_rows - 1), num_rows),
        "clouds": np.linspace(0, 10 * (num_rows - 1), num_rows),
    }
    return pd.DataFrame(data, index=index)


def test_prepare_features_and_target():
    df = _generate_df(6)
    X, y = model.prepare_features(df)
    # X should have one row less than original DataFrame
    assert len(X) == len(df) - 1
    assert len(y) == len(df) - 1
    # The first value of y should match the temp of second row
    assert y.iloc[0] == df.iloc[1]["temp"]
    # The first value of X["temp"] should match the first row of df
    assert X.iloc[0]["temp"] == df.iloc[0]["temp"]


def test_split_data():
    df = _generate_df(10)
    X, y = model.prepare_features(df)
    X_train, X_test, y_train, y_test = model.split_data(X, y, test_size=0.3, random_state=0)
    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)


def test_prepare_features_drops_missing_values():
    df = _generate_df(8)
    df.loc[df.index[2], "wind_gust"] = np.nan
    X, y = model.prepare_features(df)
    assert not X.isna().any().any()
    assert not y.isna().any()
    # One row dropped for shift, one for NaN in features
    assert len(X) == len(df) - 2


def test_train_evaluate_and_persist(tmp_path):
    df = _generate_df(12)
    X, y = model.prepare_features(df)
    X_train, X_test, y_train, y_test = model.split_data(X, y)
    reg = model.train_random_forest(X_train, y_train, n_estimators=10, random_state=0)
    score = model.evaluate_model(reg, X_test, y_test)
    # Score should be a float (could be negative on synthetic data)
    assert isinstance(score, float)
    # Save and load the model
    model_path = tmp_path / "rf.joblib"
    model.save_model(reg, model_path)
    loaded_model = model.load_model(model_path)
    # Predictions from loaded model should match those from the original model
    np.testing.assert_array_almost_equal(
        reg.predict(X_test), loaded_model.predict(X_test)
    )
