import unittest
from logging import Logger, LoggerAdapter, getLogger
from unittest.mock import Mock

from hoss.harmony_log_context import get_logger, set_logger


class TestLoggingContext(unittest.TestCase):
    """Test cases for logging context functions."""

    def setUp(self):
        """Reset the global logger before each test."""
        import hoss.harmony_log_context

        hoss.harmony_log_context._LOGGER = None

    def test_set_logger_with_logger(self):
        """Test setting a Logger instance."""
        logger = getLogger("test_logger")
        set_logger(logger)

        result = get_logger()
        self.assertIs(result, logger)

    def test_set_logger_with_adapter(self):
        """Test setting a LoggerAdapter instance."""
        base_logger = getLogger("test_logger")
        adapter = LoggerAdapter(base_logger, {"key": "value"})
        set_logger(adapter)

        result = get_logger()
        self.assertIs(result, adapter)

    def test_get_logger_without_set(self):
        """Test get_logger returns default logger when not set."""
        result = get_logger()

        self.assertIsInstance(result, Logger)
        self.assertEqual(result.name, "harmony-service")

    def test_get_logger_custom_default(self):
        """Test get_logger with custom default name."""
        result = get_logger("custom-service")

        self.assertIsInstance(result, Logger)
        self.assertEqual(result.name, "custom-service")

    def test_get_logger_after_set(self):
        """Test get_logger returns set logger, not default."""
        custom_logger = getLogger("custom")
        set_logger(custom_logger)

        result = get_logger("fallback")

        self.assertIs(result, custom_logger)
        self.assertNotEqual(result.name, "fallback")

    def test_set_logger_overwrites_previous(self):
        """Test that set_logger overwrites previously set logger."""
        logger1 = getLogger("logger1")
        logger2 = getLogger("logger2")

        set_logger(logger1)
        self.assertIs(get_logger(), logger1)

        set_logger(logger2)
        self.assertIs(get_logger(), logger2)

    def test_logger_persistence_across_calls(self):
        """Test that logger persists across multiple get_logger calls."""
        logger = getLogger("persistent")
        set_logger(logger)

        result1 = get_logger()
        result2 = get_logger()

        self.assertIs(result1, result2)
        self.assertIs(result1, logger)
