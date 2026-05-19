"""Base reVRt utilities"""

import shutil
import logging
import contextlib
from pathlib import Path
from warnings import warn

import rioxarray
import odc.geo.xr
import pandas as pd
import numpy as np
import xarray as xr
from pyproj import CRS, Transformer
from rasterio.warp import Resampling
from shapely.ops import transform as shapely_transform

from revrt.exceptions import (
    revrtFileNotFoundError,
    revrtProfileCheckError,
    revrtValueError,
)
from revrt.warn import revrtWarning


logger = logging.getLogger(__name__)
_NUM_GEOTIFF_DIMS = 3  # (band, y, x)
TRANSFORM_ATOL = 0.01
"""Tolerance in transform comparison when checking GeoTIFFs"""


def buffer_routes(
    routes, row_widths=None, row_width_ranges=None, row_width_key="voltage"
):
    """Buffer routes by specified row widths or row width ranges

    .. WARNING::
        Paths without a valid voltage in the `row_widths` or
        `row_width_ranges` input will be dropped from the output.

    Parameters
    ----------
    routes : geopandas.GeoDataFrame
        GeoDataFrame of routes to buffer. This dataframe must contain
        the route geometry as well as the `row_width_key` column.
    row_widths : dict, optional
        A dictionary specifying the row widths in the following format:
        ``{"row_width_id": row_width_meters}``. The ``row_width_id`` is
        a value used to match each route with a particular ROW width
        (this is typically a voltage). The value should be found under
        the ``row_width_key`` entry of the ``routes``.

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
    row_width_key : str, default="voltage"
        Name of column in vector file of routes used to map to the ROW
        widths. By default, ``"voltage"``.

    Returns
    -------
    geopandas.GeoDataFrame
        Route input with buffered paths (and without routes that are
        missing a voltage specification in the `row_widths` or
        `row_width_ranges` input).

    Raises
    ------
    revrtValueError
        If neither `row_widths` nor `row_width_ranges` are provided.
    """
    if not (row_widths or row_width_ranges):
        msg = "Must provide either `row_widths` or `row_width_ranges` input!"
        raise revrtValueError(msg)

    half_width = None
    if row_width_ranges:
        half_width = _compute_half_width_using_ranges(
            routes, row_width_ranges, row_width_key=row_width_key
        )

    if row_widths:
        hw_from_volts = _compute_half_width_using_voltages(
            routes, row_widths, row_width_key=row_width_key
        )
        if half_width is None:
            half_width = hw_from_volts
        else:
            half_width[hw_from_volts > 0] = hw_from_volts[hw_from_volts > 0]

    mask = half_width < 0
    if mask.any():
        msg = (
            f"{sum(mask):,d} route(s) will be dropped due to missing "
            "voltage-to-ROW-width mapping"
        )
        warn(msg, revrtWarning)
        routes = routes.loc[~mask].copy()
        half_width = half_width.loc[~mask]

    routes["geometry"] = routes.buffer(half_width, cap_style="flat")

    return routes


def delete_data_file(fp):
    """Delete data file (can be Zarr, which is a directory)

    Parameters
    ----------
    fp : path-like
        Path to data file (or directory in case of Zarr).
    """
    fp = Path(fp)
    if not fp.exists():
        return

    if fp.is_dir():
        shutil.rmtree(fp)
    else:
        fp.unlink()


def check_geotiff(layer_file_fp, geotiff, transform_atol=0.01):
    """Compare GeoTIFF with exclusion layer and raise errors if mismatch

    Parameters
    ----------
    layer_file_fp : path-like
        Path to data representing a
        :class:`~revrt.utilities.handlers.LayeredFile` instance.
    geotiff : path-like
        Path to GeoTIFF file.
    transform_atol : float, default=0.01
        Absolute tolerance parameter when comparing GeoTIFF transform
        data.

    Raises
    ------
    revrtProfileCheckError
        If shape, profile, or transform don't match between layered file
        and GeoTIFF file.
    """
    with (
        xr.open_dataset(
            layer_file_fp, consolidated=False, engine="zarr"
        ) as ds,
        rioxarray.open_rasterio(geotiff) as tif,
    ):
        if len(tif.band) > 1:
            msg = f"{geotiff} contains more than one band!"
            raise revrtProfileCheckError(msg)

        layered_file_shape = ds.sizes["band"], ds.sizes["y"], ds.sizes["x"]
        if layered_file_shape != tif.shape:
            msg = (
                f"Shape of layer data in {geotiff} and {layer_file_fp} "
                f"do not match!\n {tif.shape} !=\n {layered_file_shape}"
            )
            raise revrtProfileCheckError(msg)

        layered_file_crs = ds.rio.crs
        tif_crs = tif.rio.crs
        if not _crs_match(layered_file_crs, tif_crs):
            msg = (
                f'Geospatial "CRS" in {geotiff} and {layer_file_fp} do not '
                f"match!\n {tif_crs} !=\n {layered_file_crs}"
            )
            raise revrtProfileCheckError(msg)

        layered_file_transform = ds.rio.transform()
        tif_transform = tif.rio.transform()
        if not np.allclose(
            layered_file_transform, tif_transform, atol=transform_atol
        ):
            msg = (
                f'Geospatial "transform" in {geotiff} and {layer_file_fp} '
                f"do not match!\n {tif_transform} !=\n "
                f"{layered_file_transform}"
            )
            raise revrtProfileCheckError(msg)


def file_full_path(file_name, *layer_dirs):
    """Get full path to file, searching `layer_dirs` if needed

    Parameters
    ----------
    file_name : str
        File name to get full path for. If just the file name is
        provided, the class `layer_dir` attribute value is prepended
        to get the full path.
    *layer_dirs : path-like
        Directories to search for file if not found in current
        directory.

    Returns
    -------
    path-like
        Full path to file.

    Raises
    ------
    revrtFileNotFoundError
        If file cannot be found in either the current directory or any
        of the `layer_dirs` directories.
    """
    full_fname = Path(file_name)
    if full_fname.exists():
        return full_fname

    for layer_dir in layer_dirs:
        full_fname = Path(layer_dir) / file_name
        if full_fname.exists():
            return full_fname

    msg = f"Unable to find file {file_name}"
    raise revrtFileNotFoundError(msg)


def load_data_using_layer_file_profile(
    layer_fp, geotiff, tiff_chunks="file", layer_dirs=None, band_index=None
):
    """Load GeoTIFF data, reprojecting to LayeredFile CRS if needed

    Parameters
    ----------
    layer_fp : path-like
        Path to layered file on disk. This file must already exist.
    geotiff : path-like
        Path to GeoTIFF from which data should be read.
    tiff_chunks : int or str, default="file"
        Chunk size to use when reading the GeoTIFF file. This will be
        passed down as the ``chunks`` argument to
        :func:`rioxarray.open_rasterio`. By default, ``"file"``.
    layer_dirs : iterable of path-like, optional
        Directories to search for `geotiff` in, if not found in current
        directory. By default, ``None``, which means only the current
        directory is searched.
    band_index : int, optional
        Optional index of band to load from the GeoTIFF. If provided,
        only that band will be returned. By default, ``None``, which
        means all bands will be returned.

    Returns
    -------
    array-like
        Raster data.

    Raises
    ------
    revrtFileNotFoundError
        If `geotiff` cannot be found in either the current directory or
        the `layer_dir` directory.
    """
    if layer_dirs:
        geotiff = file_full_path(geotiff, *layer_dirs)

    with xr.open_dataset(layer_fp, consolidated=False, engine="zarr") as ds:
        crs = ds.rio.crs
        width, height = ds.rio.width, ds.rio.height
        transform = ds.rio.transform()
        if tiff_chunks == "file":
            tiff_chunks = ds.attrs.get("chunks", "auto")

    logger.debug(
        "Using the following chunks to open '%s': %r", geotiff, tiff_chunks
    )

    tif = rioxarray.open_rasterio(geotiff, chunks=tiff_chunks)
    try:
        check_geotiff(layer_fp, geotiff, transform_atol=TRANSFORM_ATOL)
    except revrtProfileCheckError:
        logger.info(
            "Profile of '%s' does not match template, reprojecting...",
            geotiff,
        )
        geo_box = odc.geo.geobox.GeoBox(
            shape=(height, width), affine=transform, crs=crs
        )
        tif = tif.odc.reproject(
            how=geo_box, resampling=Resampling.nearest, INIT_DEST=0
        )

    if band_index is not None:
        tif = tif.isel(band=band_index)

    return tif


def save_data_using_layer_file_profile(
    layer_fp,
    data,
    geotiff,
    nodata=None,
    lock=None,
    metadata=None,
    **profile_kwargs,
):
    """Write to GeoTIFF file

    Parameters
    ----------
    layer_fp : path-like
        Path to layered file on disk. This file must already exist.
    data : array-like
        Data to write to GeoTIFF using
        :class:`~revrt.utilities.handlers.LayeredFile` profile.
    geotiff : path-like
        Path to output GeoTIFF file.
    nodata : int or float, optional
        Optional nodata value for the raster layer. By default,
        ``None``, which does not add a "nodata" value.
    lock : bool or `dask.distributed.Lock`, optional
        Lock to use to write data using dask. If not supplied, a single
        process is used for writing data to the GeoTIFF.
        By default, ``None``.
    metadata : dict, optional
        Optional metadata tags to add to the output GeoTIFF. By
        default, ``None``.
    **profile_kwargs
        Additional keyword arguments to pass into writing the
        raster. The following attributes ar ignored (they are set
        using properties of the source
        :class:`~revrt.utilities.handlers.LayeredFile`):

            - nodata
            - transform
            - crs
            - count
            - width
            - height

    Raises
    ------
    revrtValueError
        If shape of provided data does not match shape of
        :class:`~revrt.utilities.handlers.LayeredFile`.
    """
    with xr.open_dataset(layer_fp, consolidated=False, engine="zarr") as ds:
        crs = ds.rio.crs
        width, height = ds.rio.width, ds.rio.height
        transform = ds.rio.transform()

    return save_data_using_custom_props(
        data=data,
        geotiff=geotiff,
        shape=(height, width),
        crs=crs,
        transform=transform,
        nodata=nodata,
        lock=lock,
        metadata=metadata,
        **profile_kwargs,
    )


def save_data_using_custom_props(
    data,
    geotiff,
    shape,
    crs,
    transform,
    nodata=None,
    lock=None,
    metadata=None,
    **profile_kwargs,
):
    """Write to GeoTIFF file

    Parameters
    ----------
    data : array-like
        Data to write to GeoTIFF using
        :class:`~revrt.utilities.handlers.LayeredFile` profile.
    geotiff : path-like
        Path to output GeoTIFF file.
    shape : tuple
        Shape of output raster (height, width).
    crs : str or dict
        Coordinate reference system of output raster.
    transform : affine.Affine
        Affine transform of output raster.
    nodata : int or float, optional
        Optional nodata value for the raster layer. By default,
        ``None``, which does not add a "nodata" value.
    lock : bool or `dask.distributed.Lock`, optional
        Lock to use to write data using dask. If not supplied, a single
        process is used for writing data to the GeoTIFF.
        By default, ``None``.
    metadata : dict, optional
        Optional metadata tags to add to the output GeoTIFF. By
        default, ``None``.
    **profile_kwargs
        Additional keyword arguments to pass into writing the
        raster. The following attributes ar ignored (they are set
        using properties of the source
        :class:`~revrt.utilities.handlers.LayeredFile`):

            - nodata
            - transform
            - crs
            - count
            - width
            - height

    Raises
    ------
    revrtValueError
        If shape of provided data does not match shape of
        :class:`~revrt.utilities.handlers.LayeredFile`.
    """
    data = expand_dim_if_needed(data)

    if data.shape[1:] != shape:
        msg = (
            f"Shape of provided data {data.shape[1:]} does "
            f"not match destination shape: {shape}"
        )
        raise revrtValueError(msg)

    if data.dtype.name == "bool":
        data = data.astype("uint8")

    da = xr.DataArray(data, dims=("band", "y", "x"))
    da.attrs["count"] = 1
    da = da.rio.write_crs(crs)
    da = da.rio.write_transform(transform)

    save_data_array_to_geotiff(
        da,
        geotiff,
        nodata=nodata,
        lock=lock,
        metadata=metadata,
        **profile_kwargs,
    )


def save_data_array_to_geotiff(
    da,
    geotiff,
    nodata=None,
    lock=None,
    metadata=None,
    **profile_kwargs,
):
    """Write xarray DataArray to GeoTIFF file

    Parameters
    ----------
    da : xarray.DataArray
        DataArray to write to GeoTIFF. This should have "y" and "x"
        dimensions, and optionally a "band" dimension (if not provided,
        one will be added).
    geotiff : path-like
        Path to output GeoTIFF file.
    nodata : int or float, optional
        Optional nodata value for the raster layer. By default,
        ``None``, which does not add a "nodata" value.
    lock : bool or `dask.distributed.Lock`, optional
        Lock to use to write data using dask. If not supplied, a single
        process is used for writing data to the GeoTIFF.
        By default, ``None``.
    metadata : dict, optional
        Optional metadata tags to add to the output GeoTIFF. By
        default, ``None``.
    **profile_kwargs
        Additional keyword arguments to pass into writing the
        raster. The following attributes ar ignored (they are set
        using properties of the source
        :class:`~revrt.utilities.handlers.LayeredFile`):

            - nodata
            - transform
            - crs
            - count
            - width
            - height
    """
    da = expand_dim_if_needed(da)

    if "y" not in da.dims or "x" not in da.dims:
        msg = "DataArray must have 'y' and 'x' dimensions!"
        raise revrtValueError(msg)

    if da.dtype.name == "bool":
        da = da.astype("uint8")

    if nodata is not None:
        nodata = da.dtype.type(nodata)
        da = da.rio.write_nodata(nodata)

    # TODO: Grab default profile from template when creating layer file
    # and use that instead
    pk = {
        "blockxsize": 256,
        "blockysize": 256,
        "tiled": True,
        "compress": "lzw",
        "interleave": "band",
        "driver": "GTiff",
    }
    pk.update(profile_kwargs)
    metadata = metadata or {}
    metadata = {str(k): str(v) for k, v in metadata.items()}
    logger.debug(
        "Saving TIFF with shape %r, dtype %r, and metadata %r to %s",
        da.shape,
        da.dtype,
        metadata,
        geotiff,
    )
    da.rio.to_raster(geotiff, lock=lock, tags=metadata, **pk)


def expand_dim_if_needed(values):
    """Expand data array dimensions if needed to ensure 3D (band, y, x)

    Parameters
    ----------
    values : array-like
        Array that is possibly missing a "band" dimension.

    Returns
    -------
    array-like
        Input array with a "band" dimension added if it was missing one.
    """
    if values.ndim >= _NUM_GEOTIFF_DIMS:
        return values

    try:
        values = values.expand_dims(dim={"band": 1})
    except AttributeError:
        values = np.expand_dims(values, 0)

    return values


def region_mapper(regions, region_identifier_column):
    """Generate a function to map points to a region

    The returned mapping function maps a point to a unique value from
    the `region_identifier_column` column in the input GeoDataFrame.

    Parameters
    ----------
    regions : geopandas.GeoDataFrame
        GeoDataFrame defining the regions. This table must
        have a `region_identifier_column` column which uniquely
        identifies the region, as well as a geometry for each region.
    """
    projected_regions, project_point = _get_point_transform(regions)

    def _map_region(point):
        """Find the region ID for the input point."""
        point = project_point(point)
        idx = projected_regions.distance(point).sort_values().index[0]
        return regions.loc[idx, region_identifier_column]

    return _map_region


def transform_xy(src_crs, dst_crs, x, y):
    """Transform coordinate iterables between coordinate systems

    Parameters
    ----------
    src_crs, dst_crs : str or dict
        Source and destination coordinate reference systems. These can
        be any valid CRS string or dictionary that can be parsed by
        pyproj.CRS.
    x, y : iterable
        Iterables of x and y coordinates to transform.

    Returns
    -------
    x, y : ndarray
        Numpy arrays of transformed x and y coordinates in the
        destination CRS.
    """
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    out_x, out_y = transformer.transform(x, y)
    return np.asarray(out_x), np.asarray(out_y)


def features_to_route_table(features):
    """Convert features GDF into route start/end point table

    This function builds a routing table to define start/end points for
    routes between all permutations of the given features. This is
    mostly for easy backward-compatibility, since the old routing code
    computed paths between all feature permutations by default.

    Parameters
    ----------
    features : geopandas.GeoDataFrame
        Geopandas DataFrame containing features to route between.

    Returns
    -------
    pandas.DataFrame
        Pandas DataFrame with the expected routing columns: `start_lat`,
        `start_lon`, `end_lat`, and `end_lon`. Also includes
        `start_index` and `index` columns for convenience. DataFrame
        index name is set to "rid".
    """
    coords = features["geometry"].centroid.to_crs("EPSG:4326")
    all_routes = []
    for start_ind, start_coord in enumerate(coords[:-1]):
        start_lat, start_lon = start_coord.y, start_coord.x
        end_coords = coords[start_ind + 1 :]
        end_idx = range(start_ind + 1, len(coords))
        new_routes = pd.DataFrame(
            {
                "end_lat": end_coords.y,
                "end_lon": end_coords.x,
                "index": end_idx,
            }
        )
        new_routes["start_lat"] = start_lat
        new_routes["start_lon"] = start_lon
        new_routes["start_index"] = start_ind
        all_routes.append(new_routes)

    all_routes = pd.concat(all_routes, axis=0).reset_index(drop=True)
    all_routes.index.name = "rid"
    return all_routes.reset_index(drop=False)


def strip_path_keys(config, keys_to_fix):
    """[NOT PUBLIC API] Strip whitespace from path-like keys"""
    for key in keys_to_fix:
        value = config.get(key)
        if isinstance(value, str):
            value = value.strip()
            for token in (r"\n", r"\r", r"\t"):
                while value.startswith(token):
                    value = value[len(token) :].lstrip()
                while value.endswith(token):
                    value = value[: -len(token)].rstrip()
            config[key] = value
    return config


def _crs_match(first_crs, second_crs):
    """Return whether two CRS definitions are semantically equivalent"""
    if first_crs is None or second_crs is None:
        return first_crs == second_crs

    with contextlib.suppress(AttributeError):
        if first_crs.equals(second_crs, ignore_axis_order=True):
            return True

    first_crs = CRS.from_user_input(first_crs)
    second_crs = CRS.from_user_input(second_crs)
    if first_crs.equals(second_crs, ignore_axis_order=True):
        return True

    first_axes = [
        (axis.name, axis.direction, axis.unit_name)
        for axis in first_crs.axis_info
    ]
    second_axes = [
        (axis.name, axis.direction, axis.unit_name)
        for axis in second_crs.axis_info
    ]
    return (
        first_crs.name == second_crs.name
        and first_axes == second_axes
        and first_crs.coordinate_system == second_crs.coordinate_system
        and {first_crs.is_projected, first_crs.is_engineering}
        == {second_crs.is_projected, second_crs.is_engineering}
    )


def _compute_half_width_using_ranges(
    routes, row_width_ranges, row_width_key="voltage"
):
    """Compute half-width for routes using row width ranges"""

    ranges = [(r["min"], r["max"], r["width"]) for r in row_width_ranges]

    def get_half_width(value):
        for min_val, max_val, width in ranges:
            if min_val <= value < max_val:
                return width / 2
        return -1

    return routes[row_width_key].map(get_half_width)


def _compute_half_width_using_voltages(
    routes, row_widths, row_width_key="voltage"
):
    """Compute half-width for routes using row width ranges"""
    row_widths = {float(k): v for k, v in row_widths.items()}

    def get_half_width(value):
        for voltage, width in row_widths.items():
            if np.isclose(value, voltage):
                return width / 2
        return -1

    return routes[row_width_key].map(get_half_width)


def _get_point_transform(regions):
    """Get function to transform points to region CRS if needed"""
    if regions.crs is None or not regions.crs.is_geographic:
        return regions, lambda point: point

    projected_crs = regions.estimate_utm_crs() or "EPSG:3857"
    projected_regions = regions.to_crs(projected_crs)
    transformer = Transformer.from_crs(
        regions.crs, projected_crs, always_xy=True
    )

    def project_point(point):
        return shapely_transform(transformer.transform, point)

    return projected_regions, project_point
