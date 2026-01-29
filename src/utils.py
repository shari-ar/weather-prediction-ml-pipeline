"""
Utility functions for the weather prediction ML pipeline.

This module centralises common functionality such as logging setup and
configuration loading. Keeping this logic in a single place avoids
duplication (DRY principle) and simplifies the overall codebase.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

try:
    # python-dotenv is an optional dependency. It allows loading
    # environment variables from a .env file located in the project
    # root. If it's not installed, loading .env files will be skipped
    # gracefully.
    from dotenv import load_dotenv  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None  # type: ignore[assignment]


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger.

    Args:
        level: Minimum severity of messages to log. Defaults to
            ``logging.INFO``. Use ``logging.DEBUG`` for more
            verbose output.

    The logging configuration here prints logs to stderr with a
    timestamp, the log level and the logger name. Calling this function
    multiple times has no additional effect thanks to the `force` flag.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,  # override any prior configuration
    )


def load_api_key(env_var: str = "OPENWEATHERMAP_API_KEY", *, dotenv_path: Optional[str] = None) -> str:
    """Load the OpenWeatherMap API key from the environment.

    Args:
        env_var: Name of the environment variable to look up.
            Defaults to ``"OPENWEATHERMAP_API_KEY"``.
        dotenv_path: Optional path to a ``.env`` file containing
            ``KEY=value`` pairs. If provided and python-dotenv is
            available, this file will be loaded before reading the
            environment variable. If omitted, python-dotenv will
            attempt to load a ``.env`` file from the current working
            directory.

    Returns:
        The API key as a string.

    Raises:
        RuntimeError: If the API key cannot be found.

    Note:
        To avoid committing credentials to source control, store the key
        in a ``.env`` file or set it in the environment before running
        the script. See the README for details.
    """
    # If python-dotenv is installed, load environment variables from
    # .env. This is optional; if it's not present, environment
    # variables must be set externally.
    if load_dotenv is not None:
        load_dotenv(dotenv_path=dotenv_path)

    api_key = os.getenv(env_var)
    if not api_key:
        raise RuntimeError(
            f"Environment variable '{env_var}' not found. "
            "Please set the API key in your environment or create a .env file."
        )
    return api_key
