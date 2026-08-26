import logging
import os

import structlog

# Configure structlog once at import time. newLogger() just creates loggers
# with this configuration; the level is fixed from LOG_LEVEL at import.
_level = os.environ.get("LOG_LEVEL", "INFO").upper()
_numeric_level = getattr(logging, _level, logging.INFO)

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(_numeric_level),
    context_class=dict,
    logger_factory=structlog.WriteLoggerFactory(),
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
    cache_logger_on_first_use=True,
)


def newLogger(name: str):
    """Create a structlog logger.

    Args:
        name: Logger name (typically __name__)

    Returns:
        structlog logger instance
    """
    return structlog.get_logger(name)