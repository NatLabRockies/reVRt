"""Timing utilities for reVRt"""

import time
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@contextmanager
def log_runtime(message, log_level=logging.INFO):
    """Log the time taken to run a block of code

    Parameters
    ----------
    message : str
        Message to log with the time taken. The time taken will be
        appended to this message.
    log_level : int, default="INFO"
        Logging level to use for the message.
        By default, ``logging.INFO``.
    hours : bool, default=False
        If ``True``, log time in hours instead of minutes.
        By default, ``False``.
    """

    start_time = time.monotonic()
    try:
        yield
    finally:
        end_time = time.monotonic()
        elapsed_time = elapsed_time_as_str(end_time - start_time)
        msg = f"{message} took {elapsed_time} to run"
        logger.log(log_level, msg)


def elapsed_time_as_str(seconds_elapsed):
    """Format elapsed time into human readable string

    Parameters
    ----------
    seconds_elapsed : int
        Number of seconds that should be represented in string form.

    Returns
    -------
    str
        Human-readable string representing the number of elapsed
        seconds.
    """
    days, seconds = divmod(int(seconds_elapsed), 24 * 3600)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    time_str = f"{hours:d}:{minutes:02d}:{seconds:02d}"
    if days:
        time_str = f"{days:,d} day{'s' if abs(days) != 1 else ''}, {time_str}"
    return time_str
