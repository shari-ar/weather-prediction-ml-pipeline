"""Tests for the utils module."""

import os
import pytest

from src.utils import load_api_key, setup_logging


def test_load_api_key_success(monkeypatch):
    """load_api_key should return the API key when it is set in the environment."""
    monkeypatch.setenv("OPENWEATHERMAP_API_KEY", "dummy-key")
    assert load_api_key() == "dummy-key"


def test_load_api_key_missing(monkeypatch):
    """load_api_key should raise RuntimeError when the key is missing."""
    # Ensure the environment variable is unset
    monkeypatch.delenv("OPENWEATHERMAP_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        load_api_key()


def test_setup_logging_does_not_error(caplog):
    """setup_logging should configure the logger without throwing exceptions."""
    # Just ensure it does not raise and logs can be emitted
    setup_logging()
    # After setup, sending a log message should be captured by caplog
    with caplog.at_level("INFO"):
        logger_name = "test_utils_logger"
        import logging

        logger = logging.getLogger(logger_name)
        logger.info("test message")
        assert any("test message" in message for message in caplog.messages)
