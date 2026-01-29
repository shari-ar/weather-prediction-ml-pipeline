"""Tests for the data_ingest module."""

import pandas as pd
import pytest

from src.data_ingest import json_to_dataframe


def test_json_to_dataframe_basic():
    """json_to_dataframe should convert a simple JSON structure into a DataFrame."""
    ts = 1700000000  # sample unix timestamp
    json_data = {
        "hourly": [
            {"dt": ts, "temp": 10.0, "humidity": 50, "pressure": 1010, "wind_speed": 5.0},
            {"dt": ts + 3600, "temp": 11.0, "humidity": 55, "pressure": 1012, "wind_speed": 4.0},
        ]
    }
    df = json_to_dataframe(json_data)
    # The resulting DataFrame should have datetime index and correct columns
    assert isinstance(df.index, pd.DatetimeIndex)
    assert "temp" in df.columns
    assert len(df) == 2
    # Ensure the first row corresponds to the first timestamp
    assert df.iloc[0]["temp"] == 10.0


def test_json_to_dataframe_missing_hourly():
    """json_to_dataframe should raise when 'hourly' key is missing."""
    with pytest.raises(ValueError):
        json_to_dataframe({})
