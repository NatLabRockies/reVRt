"""revrt zonal stats command line interface (CLI)"""

import logging
from pathlib import Path

import rioxarray
import pandas as pd
import geopandas as gpd
from dask.distributed import Client
from gaps.config import load_config
from gaps.cli import CLICommandFromFunction

from revrt.spatial_characterization.zonal import ZonalStats
from revrt.utilities import buffer_routes, dask_performance_report, log_runtime


logger = logging.getLogger(__name__)


def buffered_route_characterizations(
    geotiff_fp,
    route_fp,
    row_widths=None,
    row_width_ranges=None,
    multiplier_scalar=1.0,
    prefix=None,
    copy_properties=None,
    parallel=False,
    row_width_key="voltage",
    chunks="auto",
    **kwargs,
):
    """Compute route characterizations/statistics

    Each route is buffered before computing statistics.

    Parameters
    ----------
    geotiff_fp : path-like
        Path to the raster file.
    route_fp : path-like
        Path to the vector file of routes. Must contain a "geometry"
        column and the `row_width_key` column (used to map to path ROW
        width).
    row_widths : dict, optional
        A dictionary specifying the row widths in the following format:
        ``{"row_width_id": row_width_meters}``. The ``row_width_id`` is
        a value used to match each route with a particular ROW width
        (this is typically a voltage). The value should be found under
        the ``row_width_key`` entry of the ``route_fp``.

        .. IMPORTANT::
            At least one of `row_widths` or `row_width_ranges` must be
            provided.

        By default, ``None``.
    row_width_ranges : list, optional
        Optional list of dictionaries, where each dictionary contains
        the keys "min", "max", and "width". This can be used to specify
        row widths based on ranges of values (e.g. voltage). For
        example, the following input::

            [
                {"min": 0, "max": 70, "width": 20},
                {"min": 70, "max": 150, "width": 30},
                {"min": 200, "max": 350, "width": 40},
                {"min": 400, "max": 500, "width": 50},
            ]

        would map voltages in the range ``0 <= volt < 70`` to a row
        width of 20 meters, ``70 <= volt < 150`` to a row width of 30
        meters, ``200 <= volt < 350`` to a row width of 40 meters,
        and so-on.

        .. IMPORTANT::
            Any values in the `row_widths` dict will take precedence
            over these ranges. So if a voltage of 138 kV is mapped to a
            row width of 25 meters in the `row_widths` dict, that value
            will be used instead of the 30 meter width specified by the
            ranges above.

        By default, ``None``.
    multiplier_scalar : float, optional
        Optional multiplier value to apply to layer before computing
        statistics. This is useful if you want to scale the values in
        the raster before computing statistics. By default, ``1.0``.
    prefix : str, optional
        A string representing a prefix to add to each stat name. If you
        wish to have the prefix separated by a delimiter, you must
        include it in this string (e.g. ``prefix="test_"``).
        By default, ``None``.
    copy_properties : iterable of str, optional
        Iterable of columns names to copy over from the zone feature.
        By default, ``None``.
    parallel : bool, optional
        Option to perform processing in parallel using dask.
        By default, ``False``.
    row_width_key : str, default="voltage"
        Name of column in vector file of routes used to map to the
        ROW widths. By default, ``"voltage"``.
    chunks : tuple or str, default="auto"
        ``chunks`` keyword argument to pass down to
        :func:`rioxarray.open_rasterio`. Use this to control the Dask
        chunk size. By default, ``"auto"``.

    Returns
    -------
    DataFrame
        Pandas DataFrame containing computed characteristics/stats.
    """
    rds = (
        rioxarray.open_rasterio(geotiff_fp, chunks=chunks) * multiplier_scalar
    )
    logger.debug("Tiff properties:\n%r", rds)
    logger.debug("Tiff chunksizes:\n%r", rds.chunksizes)  # cspell:disable-line

    routes = gpd.read_file(route_fp)
    routes = routes.to_crs(rds.rio.crs)

    routes = buffer_routes(
        routes,
        row_widths=row_widths,
        row_width_ranges=row_width_ranges,
        row_width_key=row_width_key,
    )

    logger.info("Initializing zonal stats with kwargs:\n%s", kwargs)
    zs = ZonalStats(**kwargs)
    logger.info("Computing stats...")
    stats = zs.from_array(
        zones=routes,
        raster_array=rds,
        affine_transform=rds.rio.transform(),
        prefix=prefix,
        copy_properties=copy_properties,
        parallel=parallel,
    )
    return pd.json_normalize(list(stats), sep="_")


def _route_characterizations_from_config(
    out_dir,
    _stat_kwargs,
    _row_widths=None,
    _row_width_ranges=None,
    _default_route_fp=None,
    _default_copy_properties=None,
    _default_row_width_key=None,
    _default_chunks=None,
    max_workers=1,
    tag=None,
    memory_limit_per_worker="auto",
    log_directory=None,
):
    """Compute route characterizations/statistics

    Parameters
    ----------
    max_workers : int, optional
        Number of parallel workers to use for computation. If ``None``
        or >1, processing is performed in parallel (using Dask). If your
        paths span large areas, keep this value low (~10) to avoid
        running into memory errors. By default, ``1``.
    memory_limit_per_worker : str, float, int, or None, default="auto"
        Sets the memory limit *per worker*. This only applies if
        ``max_workers != 1``. If ``None`` or ``0``, no limit is applied.
        If ``"auto"``, the total system memory is split evenly between
        the workers. If a float, that fraction of the system memory is
        used *per worker*. If a string giving a number  of bytes (like
        "1GiB"), that amount is used *per worker*. If an int, that
        number of bytes is used *per worker*. By default, ``"auto"``
    log_directory : path-like, optional
        Directory to save Dask performance reports in. If ``None``, Dask
        performance reports will not be generated. By default, ``None``.
    """
    tag = tag or ""
    _stat_kwargs.setdefault("route_fp", _default_route_fp)
    _stat_kwargs.setdefault("copy_properties", _default_copy_properties)
    _stat_kwargs.setdefault(
        "row_width_key", _default_row_width_key or "voltage"
    )
    _stat_kwargs.setdefault("chunks", _default_chunks or "auto")

    raster_name = _stat_kwargs.get("geotiff_fp")
    raster_name = f"_{Path(raster_name).stem}" if raster_name else ""
    route_name = _stat_kwargs.get("route_fp")
    route_name = f"_{Path(route_name).stem}" if route_name else ""
    out_fp = Path(out_dir) / f"characterized{raster_name}{route_name}{tag}.csv"

    logger.debug(
        "Running with max_workers=%r and memory_limit_per_worker=%r",
        max_workers,
        memory_limit_per_worker,
    )
    client = None
    parallel = False
    if max_workers != 1:
        parallel = True
        client = Client(
            n_workers=max_workers,
            memory_limit=memory_limit_per_worker,
            dashboard_address=None,
        )
        logger.info(
            "Dask client created with %s workers and %s memory limit per "
            "worker",
            max_workers,
            memory_limit_per_worker,
        )

    try:
        with (
            dask_performance_report(
                "route_characterizations",
                out_dir=log_directory if max_workers != 1 else None,
            ),
            log_runtime("Characterizing routes"),
        ):
            out_data = buffered_route_characterizations(
                row_widths=_row_widths,
                row_width_ranges=_row_width_ranges,
                parallel=parallel,
                **_stat_kwargs,
            )
            out_data.to_csv(out_fp, index=False)
    finally:
        if client is not None:
            client.close()

    return str(out_fp)


def _preprocess_stats_config(
    config,
    layers,
    default_route_fp=None,
    default_copy_properties=None,
    default_row_width_key=None,
    default_chunks=None,
    row_widths=None,
    row_width_ranges=None,
):
    """Preprocess user config

    Parameters
    ----------
    config : dict
        User configuration parsed as (nested) dict.
    layers : dict or list of dict
        A single dictionary or a list of dictionaries specifying the
        statistics to compute. Each dictionary should contain the
        following keys:

            - geotiff_fp: (REQUIRED) Path to the raster file.
            - route_fp: (REQUIRED) Path to the vector file of routes.
              Must contain a "geometry" column and the `row_width_key`
              column (used to map to path ROW width).
            - stats: (OPTIONAL) Names of all statistics to compute.
              Statistics must be one of the members of
              :class:`~revrt.spatial_characterization.stats.Stat` or
              :class:`~revrt.spatial_characterization.stats.FractionalStat`,
              or must start with the ``percentile_`` prefix and end with
              an int or float representing the percentile to compute
              (e.g. ``percentile_10.5``). If only one statistic is to be
              computed, you can provide it directly as a string.
              Otherwise, provide a list of statistic names or a string
              with the names separated by a space. You can also provide
              the string ``"ALL"`` or ``"*"`` to specify that all
              statistics should be computed (i.e. all options from
              *both*
              :class:`~revrt.spatial_characterization.stats.Stat` and
              :class:`~revrt.spatial_characterization.stats.FractionalStat`).
              If no input, empty input, or ``None`` is provided, then
              only the base stats ("count", "min", "max", "mean") are
              configured. To summarize, all of the following are valid
              inputs:

                - ``stats: "*"`` or ``stats="ALL"`` or ``stats="All"``
                - ``stats: "min"``
                - ``stats: "min max"``
                - ``stats: ["min"]``
                - ``stats: ["min", "max", "percentile_10.5"]``

            - nodata : (OPTIONAL) Value in the raster that represents
              `nodata`. This value will not show up in any statistics
              except for the `nodata` statistic itself, which computes
              the number of `nodata` values within the buffered routes.
              Note that this value is used **in addition to** any
              `NODATA` value in the raster's metadata.
            - all_touched : (OPTIONAL) Boolean flag indicating whether
              to include every raster cell touched by a geometry
              (``True``), or only those having a center point within the
              polygon (``False``). By default, ``True``.
            - category_map : (OPTIONAL) Dictionary mapping raster values
              to new names. If given, this mapping will be applied to
              the pixel count dictionary, so you can use it to map
              raster values to human-readable category names.
            - multiplier_scalar: (OPTIONAL) Optional multiplier value to
              apply to layer before computing statistics. This is useful
              if you want to scale the values in the raster before
              computing statistics. By default, ``1.0``.
            - prefix: (OPTIONAL) A string representing a prefix to add
              to each stat name. If you wish to have the prefix
              separated by a delimiter, you must include it in this
              string (e.g. ``prefix="test_"``).
            - copy_properties: (OPTIONAL) List of columns names to copy
              over from the vector file of routes.
            - row_width_key: (OPTIONAL) Name of column in vector file of
              routes used to map to the ROW widths.
              By default, ``"voltage"``.
            - chunks : (OPTIONAL) ``chunks`` keyword argument to pass
              down to :func:`rioxarray.open_rasterio`. Use this to
              control the Dask chunk size. By default, ``"auto"``.

    default_route_fp : path-like, optional
        Default path to the vector file of routes. This will be used
        *only if* no `route_fp` is provided in a layer's stats
        dictionary. Must contain a "geometry" column and the
        `row_width_key` column (used to map to path ROW width).
        By default, ``None``.
    default_copy_properties : iterable of str, optional
        Default iterable of columns names to copy over from the zone
        feature. This will be used *only if* no `copy_properties` is
        provided in a layer's stats dictionary. By default, ``None``.
    default_row_width_key : str, optional
        Default name of column in vector file of routes used to map to
        the ROW widths. This will be used *only if* no `row_width_key`
        is provided in a layer's stats dictionary. By default, ``None``.
    default_chunks : tuple or str, optional
        Default ``chunks`` keyword argument to pass down to
        :func:`rioxarray.open_rasterio`. This will be used *only if* no
        `chunks` is provided in a layer's stats dictionary. Use this to
        control the Dask chunk size. By default, ``None``, which uses
        ``"auto"`` as the final chunk input.
    row_widths : dict or path-like, optional
        A dictionary specifying the row widths in the following format:
        ``{"row_width_id": row_width_meters}``. The ``row_width_id`` is
        a value used to match each route with a particular ROW width
        (this is typically a voltage). The value should be found under
        the ``row_width_key`` entry of the ``route_fp``.

        .. IMPORTANT::
            At least one of `row_widths` or `row_width_ranges` must be
            provided.

        .. WARNING::
           Routes without a valid voltage in the `row_widths` or
           `row_width_ranges` input will not be characterized.

        If a path is provided, it should point to a JSON file containing
        the row width dictionary as specified above.
        By default, ``None``.
    row_width_ranges : list, optional
        Optional list of dictionaries, where each dictionary contains
        the keys "min", "max", and "width". This can be used to specify
        row widths based on ranges of values (e.g. voltage). For
        example, the following input::

            [
                {"min": 0, "max": 70, "width": 20},
                {"min": 70, "max": 150, "width": 30},
                {"min": 200, "max": 350, "width": 40},
                {"min": 400, "max": 500, "width": 50},
            ]

        would map voltages in the range ``0 <= volt < 70`` to a row
        width of 20 meters, ``70 <= volt < 150`` to a row width of 30
        meters, ``200 <= volt < 350`` to a row width of 40 meters,
        and so-on.

        .. IMPORTANT::
            Any values in the `row_widths` dict will take precedence
            over these ranges. So if a voltage of 138 kV is mapped to a
            row width of 25 meters in the `row_widths` dict, that value
            will be used instead of the 30 meter width specified by the
            ranges above.

        If a path is provided, it should point to a JSON file containing
        the list of dictionaries. By default, ``None``.
    """
    for key, user_input in (
        ("_row_widths", row_widths),
        ("_row_width_ranges", row_width_ranges),
    ):
        if isinstance(user_input, str):
            user_input = load_config(user_input)  # noqa: PLW2901

        config[key] = user_input

    if isinstance(layers, dict):
        layers = [layers]

    config["_stat_kwargs"] = layers
    config["_default_route_fp"] = default_route_fp
    config["_default_copy_properties"] = default_copy_properties
    config["_default_row_width_key"] = default_row_width_key
    config["_default_chunks"] = default_chunks
    return config


route_characterizations_command = CLICommandFromFunction(
    _route_characterizations_from_config,
    name="route-characterization",
    add_collect=False,
    split_keys=["_stat_kwargs"],
    config_preprocessor=_preprocess_stats_config,
)
