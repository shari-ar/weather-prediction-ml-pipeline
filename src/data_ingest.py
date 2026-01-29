"""
Data ingestion for the weather prediction ML pipeline.

This module fetches weather data from the OpenWeatherMap API and
converts it into a tidy tabular form suitable for modelling. It uses
the ``utils`` module to load API credentials and configure logging.

Usage from the command line::

    python src/data_ingest.py --lat 52.52 --lon 13.405 --output weather.csv

Configuration:
    The script expects an ``OPENWEATHERMAP_API_KEY`` environment
    variable to be set or present in a ``.env`` file in the project
    root. See ``utils.load_api_key`` for details.

Example::

    export OPENWEATHERMAP_API_KEY=your_api_key_here
    python src/data_ingest.py --lat 37.7749 --lon -122.4194 --output sf_weather.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import requests

# Use relative import when part of a package (e.g. called via ``python -m``),
# fallback to absolute import when run as a script directly. This pattern
# improves developer ergonomics during local testing.
try:
    from .utils import load_api_key, setup_logging  # type: ignore[import]
except ImportError:
    from utils import load_api_key, setup_logging  # type: ignore[import]

# Base URL for the One Call API 3.0. Note: This endpoint may
# change over time; check the official docs for the most current version.
BASE_URL = "https://api.openweathermap.org/data/3.0/onecall"


def fetch_weather_data(lat: float, lon: float, api_key: str) -> Dict[str, Any]:
    """Fetch weather data for a location from OpenWeatherMap.

    Args:
        lat: Latitude of the location.
        lon: Longitude of the location.
        api_key: API key for authenticating with OpenWeatherMap.

    Returns:
        Raw JSON response as a dictionary.

    Raises:
        requests.HTTPError: If the HTTP request fails.
    """
    logger = logging.getLogger(__name__)
    params = {
        "lat": lat,
        "lon": lon,
        "appid": api_key,
        # Request hourly data. Additional fields can be specified via
        # "exclude" or "units" parameters. We use metric units by default.
        "units": "metric",
        "exclude": "minutely,daily,alerts,current",
    }
    logger.debug("Requesting weather data for (%s, %s)", lat, lon)
    response = requests.get(BASE_URL, params=params, timeout=10)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        logger.error("Failed to fetch weather data: %s", exc)
        raise
    logger.debug("Weather data fetched successfully")
    return response.json()


def json_to_dataframe(json_data: Dict[str, Any]) -> pd.DataFrame:
    """Convert the raw JSON response into a pandas DataFrame.

    The OpenWeatherMap One Call API returns nested JSON with an
    ``hourly`` list of observations. Each observation contains fields
    such as ``dt`` (timestamp), ``temp`` (temperature), etc. This
    function flattens the hourly records into a table and converts the
    timestamps into timezone-aware datetimes.

    Args:
        json_data: Parsed JSON data from OpenWeatherMap.

    Returns:
        A DataFrame containing one row per hour with all available
        meteorological variables.
    """
    logger = logging.getLogger(__name__)
    if "hourly" not in json_data:
        logger.error("JSON data does not contain 'hourly' records")
        raise ValueError("Missing 'hourly' key in API response")
    # Flatten the list of hourly observations.
    df = pd.json_normalize(json_data["hourly"])
    # Convert the Unix timestamp to a datetime in UTC.
    if "dt" in df.columns:
        df["datetime"] = pd.to_datetime(df["dt"], unit="s", utc=True)
        df.drop(columns=["dt"], inplace=True)
        df.set_index("datetime", inplace=True)
        df.sort_index(inplace=True)
    logger.debug("Converted JSON to DataFrame with %d rows", len(df))
    return df


def save_dataframe_to_csv(df: pd.DataFrame, file_path: Path) -> None:
    """Save a DataFrame to a CSV file.

    Args:
        df: The DataFrame to save.
        file_path: Destination path for the CSV file. Parent
            directories will be created if they do not exist.
    """
    logger = logging.getLogger(__name__)
    file_path = file_path.expanduser().resolve()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=True)
    logger.info("Saved weather data to %s", file_path)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        args: Optional list of arguments to parse. Defaults to None,
            which tells ``argparse`` to parse ``sys.argv``.

    Returns:
        Namespace with the parsed options.
    """
    parser = argparse.ArgumentParser(
        description="Fetch hourly weather data from OpenWeatherMap and save to CSV."
    )
    parser.add_argument(
        "--lat",
        type=float,
        required=True,
        help="Latitude of the location (e.g. 52.52)",
    )
    parser.add_argument(
        "--lon",
        type=float,
        required=True,
        help="Longitude of the location (e.g. 13.405)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/weather_hourly.csv"),
        help="Path to the output CSV file (default: data/weather_hourly.csv)",
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
    """Entry point for the data ingestion script."""
    args = parse_args()
    # Initialise logging
    setup_logging(getattr(logging, args.log_level))
    logger = logging.getLogger(__name__)

    try:
        api_key = load_api_key()
    except RuntimeError as exc:
        logger.error(str(exc))
        return

    try:
        raw_data = fetch_weather_data(args.lat, args.lon, api_key)
    except Exception as exc:
        logger.exception("Could not fetch weather data: %s", exc)
        return

    try:
        df = json_to_dataframe(raw_data)
    except Exception as exc:
        logger.exception("Could not convert API response to DataFrame: %s", exc)
        return

    save_dataframe_to_csv(df, args.output)


if __name__ == "__main__":
    main()
