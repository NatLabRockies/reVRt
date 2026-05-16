"""reVRt monitoring utilities"""

import time
import psutil
import logging
from pathlib import Path
from contextlib import nullcontext, contextmanager
from uuid import uuid4

import numpy as np
import xarray as xr
import dask.array as da
from dask.distributed import performance_report


logger = logging.getLogger(__name__)


def log_mem(log_level="DEBUG"):
    """Log the memory usage to the input logger object

    Parameters
    ----------
    log_level : str, default="DEBUG"
        Logging level to use. Can be any valid log level string, such
        as  DEBUG or INFO for different log levels for this log message.
        By default, ``"DEBUG"``.

    Returns
    -------
    msg : str
        Memory utilization log message string.
    """
    mem = psutil.virtual_memory()
    msg = (
        f"Memory utilization is {mem.used / (1024.0**3):.3f} GB "
        f"out of {mem.total / (1024.0**3):.3f} GB total "
        f"({mem.used / mem.total:.1%} used)"
    )
    log_level = logging.getLevelNamesMapping().get(log_level.upper(), "DEBUG")
    logger.log(log_level, msg)

    return msg


def log_array_backend(fname, data, kind):
    """Log backend information for layer data

    Parameters
    ----------
    fname : str or path-like
        Layer name or source identifier to include in the log message.
    data : xarray.DataArray, dask.array.Array, numpy.ndarray, or object
        Data object to inspect. For xarray inputs, the underlying array
        backend is checked to determine whether the data are NumPy- or
        Dask-backed. Other objects are logged using their concrete type
        name, along with any ``dtype`` and ``shape`` attributes if
        present.
    kind : str
        Short label describing the kind of the layer, such as
        ``"processed"`` or ``"lazy reload"``.
    """
    if isinstance(data, xr.DataArray):
        backend = data.data
        backend_name = type(backend).__name__
        if isinstance(backend, da.Array):
            storage = "Dask"
        elif isinstance(backend, np.ndarray):
            storage = "NumPy"
        else:
            storage = backend_name
    elif isinstance(data, da.Array):
        storage = "Dask"
        backend_name = type(data).__name__
    elif isinstance(data, np.ndarray):
        storage = "NumPy"
        backend_name = type(data).__name__
    else:
        storage = type(data).__name__
        backend_name = storage

    logger.debug(
        "%s layer %s is %s-backed (%s) with dtype %s, and shape %s",
        kind,
        fname,
        storage,
        backend_name,
        getattr(data, "dtype", None),
        getattr(data, "shape", None),
    )


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


def dask_performance_report(prefix, out_dir=None):
    """Context manager for generating a Dask performance report

    Parameters
    ----------
    prefix : str
        Prefix to use for the performance report filename, which will be
        saved as an HTML file in the specified output directory.
    out_dir : path-like, optional
        Path to the directory where the performance report should be
        saved. If not provided, the context manager will not generate a
        report and will simply yield without doing anything.
        By default, ``None``.

    Returns
    -------
    object
        A context manager that generates a Dask performance report when
        entered, and saves it to the specified output directory with a
        filename based on the provided prefix. A UUID is appended to the
        filename to avoid collisions across multiple invocations. If no
        output directory is specified, the context manager will do
        nothing and simply yield without generating a report.
    """
    if out_dir is None:
        return nullcontext()

    return performance_report(
        filename=Path(out_dir) / f"dask-report_{prefix}_{uuid4().hex}.html"
    )


def close_dask_client(client, timeout=600):
    """Close a Dask client without failing on shutdown timeouts

    Parameters
    ----------
    client : distributed.Client or None
        Dask client to close. If ``None``, this function does nothing.
    timeout : int, optional
        Number of seconds to wait for the client to close before timing
        out. If the client fails to close within this time, a warning
        is logged and the client is marked as closed anyway to allow
        cleanup to proceed without hanging. By default, ``600``.
    """
    if client is None:
        return

    try:
        client.close(timeout=timeout)
    except TimeoutError:
        logger.warning(
            "Timed out closing Dask client after %s seconds; "
            "continuing cleanup",
            timeout,
            exc_info=True,
        )
        client.status = "closed"
        vars(client)["_Client__loop"] = None
