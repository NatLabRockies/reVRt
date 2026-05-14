"""Timing utilities for reVRt"""

import time
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@contextmanager
def log_time(message, log_level=logging.INFO, hours=False):
    """Log the time taken to execute a block of code

    Parameters
    ----------
    message : str
        Message to log with the time taken. The time taken will be
        appended to this message.
    log_level : int, default=logging.INFO
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
        elapsed_time = (end_time - start_time) / 60
        time_unit = "minute(s)"
        if hours:
            elapsed_time /= 60
            time_unit = "hour(s)"

        msg = f"{message} completed in {elapsed_time:.2f} {time_unit}"
        logger.log(log_level, msg)
