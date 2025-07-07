import logging
import sys

def get_logger(name: str, log_level: int = logging.INFO,
               logfile="rankfields.log") -> logging.Logger:
    """
    Creates and returns a logger with the specified name and log level.
    Logs to both console and file.

    Args:
        name (str): Name for the logger (e.g., __name__).
        log_level (int): Logging level, default INFO.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)

    # Remove any old handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(log_level)

    # Formatter
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Stream (console) handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # File handler
    file_handler = logging.FileHandler(logfile, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
