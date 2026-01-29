"""Weather prediction ML pipeline package.

This package contains modules for data ingestion, model training and
utility functions used throughout the project. See the individual
modules for detailed documentation.
"""

try:
    from importlib.metadata import version as _get_version
    __version__ = _get_version("weather-prediction-ml-pipeline")
except Exception:
    # Fallback to static version if package metadata is unavailable
    __version__ = "0.1.0"

__all__ = ["data_ingest", "model", "utils"]


# Expose high level functionality at the package level if desired.

from . import data_ingest, model, utils  # noqa: F401
