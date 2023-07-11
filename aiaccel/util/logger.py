import logging


class Color:
    def __init__(self) -> None:
        self.black: str = "\033[30m"
        self.red: str = "\033[31m"
        self.green: str = "\033[32m"
        self.yellow: str = "\033[33m"
        self.blue: str = "\033[34m"
        self.magenta: str = "\033[35m"
        self.cyan: str = "\033[36m"
        self.lightgray: str = "\033[37m"
        self.default: str = "\033[39m"
        self.darkgray: str = "\033[90m"
        self.lightred: str = "\033[91m"
        self.lightgreen: str = "\033[92m"
        self.lightyellow: str = "\033[93m"
        self.lightblue: str = "\033[94m"
        self.lightmagenta: str = "\033[95m"
        self.lightcyan: str = "\033[96m"
        self.white: str = "\033[97m"
        self.reset: str = "\033[0m"


class ColoredHandler(logging.StreamHandler):  # type: ignore
    def emit(self, record) -> None:  # type: ignore
        if record.levelno == logging.DEBUG:
            color_start = Color().blue
        elif record.levelno == logging.INFO:
            color_start = Color().green
        elif record.levelno == logging.WARNING:
            color_start = Color().yellow
        elif record.levelno == logging.ERROR:
            color_start = Color().red
        elif record.levelno == logging.CRITICAL:
            color_start = Color().magenta
        else:
            color_start = Color().default
        color_end = Color().reset
        self.stream.write(f"{color_start}{self.format(record)}{color_end}\n")


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
    elif "notset" in s.lower():
        return logging.NOTSET
    else:
        raise ValueError(f"Invalid logging level: {s}, {type(s)}")
