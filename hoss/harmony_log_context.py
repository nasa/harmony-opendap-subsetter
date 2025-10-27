"""Logging Context.

This module is used to capture the logging context from Harmony and allow easy
access to this context wrapped harmony logging adapter.

We use this to capture the Logger (LoggerAdapter) that the harmony service lib
has created and allows all of the modules in the service to access and use it
without having to pass a logging object in each function signature.

"""

from logging import Logger, LoggerAdapter, getLogger

_LOGGER = None


def set_logger(logger: Logger | LoggerAdapter) -> None:
    """Set the logger context for this Request's session.

    This should be set in your service's harmony service adapter's __init__
    after it calls the BaseHarmonyAdapter's __init__.

    """
    global _LOGGER
    _LOGGER = logger


def get_logger(default_name: str = "harmony-service") -> Logger | LoggerAdapter:
    """Get the context logger or fall back to module logger.

    Use this method to retrieve the harmony logger in your service.
    """
    return _LOGGER if _LOGGER else getLogger(default_name)
