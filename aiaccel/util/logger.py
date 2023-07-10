import logging


def str_to_logging_level(s: str) -> int:
    """Convert from a input string to a logging level.

    Args:
        s (str): A string that specify a logging level.

    Returns:
        int: It returns a logging level.

    Raises:
        ValueError: Causes when an invalid argument s is given.
    """
    if "debug" in s.lower():
        return logging.DEBUG
    elif "info" in s.lower():
        return logging.INFO
    elif "warning" in s.lower():
        return logging.WARNING
    elif "warn" in s.lower():
        return logging.WARNING
    elif "error" in s.lower():
        return logging.ERROR
    elif "critical" in s.lower():
        return logging.CRITICAL
    else:
        raise ValueError(f"Invalid logging level: {s}, {type(s)}")
