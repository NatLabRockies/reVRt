"""reVRt rasterization utilities"""

import logging
from math import ceil, hypot

import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import box

from revrt.constants import DEFAULT_DTYPE
from revrt.exceptions import revrtValueError
from revrt.utilities.monitoring import log_runtime


logger = logging.getLogger(__name__)
DEFAULT_RASTERIZE_TILE_SIZE = 2048
"""int: Default tile size to use for rasterization"""


def rasterize_shape_file(  # noqa: PLR0913, PLR0917
    fname,
    width,
    height,
    transform,
    tile_size=None,
    buffer_dist=None,
    all_touched=False,
    dest_crs=None,
    burn_value=1,
    boundary_only=False,
    simply_before_rasterize=False,
    dtype=DEFAULT_DTYPE,
    fill=0,
):
    """Rasterize a vector layer

    Parameters
    ----------
    fname : str
        Full path to GPKG or shp file.
    width : int
        Width of output raster.
    height : int
        Height of output raster.
    transform : affine.Affine
        Affine transform for output raster.
    tile_size : int, optional
        Size of tiles to use for rasterization. If None, uses a default
        tile size. By default, ``None``.
    buffer_dist : float, optional
        Distance to buffer features in fname by. Same units as the
        template raster. By default, ``None``.
    all_touched : bool, default=False
        Set all cells touched by vector to 1. False results in less
        cells being set to 1. By default, ``False``.
    reproject_vector : bool, default=True
        Reproject CRS of vector to match template raster if ``True``.
        By default, ``True``.
    burn_value : int, float, or str, default=1
        Value used to burn vectors into raster, or the name of a
        column containing per-feature values to burn.
        By default, ``1``.
    boundary_only : bool, default=False
        If ``True``, rasterize boundary of vector.
        By default, ``False``.
    simply_before_rasterize : bool, default=False
        If ``True``, simplify geometries to half of the raster cell size
        before rasterization. By default, ``False``.
    dtype : np.dtype, default="float32"
        Datatype to use. By default, ``float32``.
    fill : int, float, or None, default=0
        Value used to fill raster cells not burned by vector. If None,
        uses np.nan. By default, ``0``.

    Returns
    -------
    array-like
        Rasterized vector data
    """
    gdf = gpd.read_file(fname)

    if dest_crs is not None:
        logger.debug("Reprojecting vector")
        gdf = gdf.to_crs(crs=dest_crs)

    with log_runtime(f"Rasterization of {fname}"):
        return rasterize(
            gdf,
            width,
            height,
            transform,
            tile_size=tile_size,
            buffer_dist=buffer_dist,
            all_touched=all_touched,
            burn_value=burn_value,
            boundary_only=boundary_only,
            simply_before_rasterize=simply_before_rasterize,
            dtype=dtype,
            fill=fill,
        )


def rasterize(  # noqa: PLR0913, PLR0917
    gdf,
    width,
    height,
    transform,
    tile_size=None,
    buffer_dist=None,
    all_touched=False,
    burn_value=1,
    boundary_only=False,
    simply_before_rasterize=False,
    dtype=DEFAULT_DTYPE,
    fill=0,
):
    """Rasterize a vector layer

    Parameters
    ----------
    gdf : DataFrame
        Geopandas DataFrame contains shapes to rasterize.
    width : int
        Width of output raster.
    height : int
        Height of output raster.
    transform : affine.Affine
        Affine transform for output raster.
    tile_size : int, optional
        Size of tiles to use for rasterization. If None, uses a default
        tile size. By default, ``None``.
    buffer_dist : float, optional
        Distance to buffer features in fname by. Same units as the
        template raster. By default, ``None``.
    all_touched : bool, default=False
        Set all cells touched by vector to 1. False results in less
        cells being set to 1. By default, ``False``.
    burn_value : int, float, or str, default=1
        Value used to burn vectors into raster, or the name of a
        column containing per-feature values to burn.
        By default, ``1``.
    boundary_only : bool, default=False
        If ``True``, rasterize boundary of vector.
        By default, ``False``.
    simply_before_rasterize : bool, default=False
        If ``True``, simplify geometries to half of the raster cell size
        before rasterization. By default, ``False``.
    dtype : np.dtype, default="float32"
        Datatype to use. By default, ``float32``.
    fill : int, float, or None, default=0
        Value used to fill raster cells not burned by vector. If None,
        uses np.nan. By default, ``0``.

    Returns
    -------
    array-like
        Rasterized vector data
    """

    if simply_before_rasterize:
        gdf = simplify_shapes(gdf, transform)

    if buffer_dist is not None:
        gdf = gdf.copy()
        logger.debug("Buffering shapes by %s", buffer_dist)
        gdf.geometry = gdf.geometry.buffer(buffer_dist)
        logger.debug("Buffering done. %d features before cleaning.", len(gdf))
        gdf = gdf[~gdf.is_empty]  # Negative buffer may result in empty feats
        logger.debug("%d features after removing empty features.", len(gdf))

    logger.debug("Rasterizing %d shape(s)", len(gdf))
    with log_runtime("Rasterization of shape(s)"):
        return _tile_rasterize(
            gdf,
            width,
            height,
            transform,
            tile_size or DEFAULT_RASTERIZE_TILE_SIZE,
            all_touched=all_touched,
            burn_value=_resolve_burn_values(gdf, burn_value),
            boundary_only=boundary_only,
            dtype=dtype,
            fill=fill,
        )


def integer_dimension_window(bounds, transform):
    """Make valid-size window with integer dimensions

    This method guarantees the window dimensions are integers >= 1.
    Window offsets are also floored to the nearest integer.

    Parameters
    ----------
    bounds : tuple
        4-tuple defining bounding box (left, bottom, right, top).
    transform : affine.Affine
        Affine transform for raster for which to generate window.

    Returns
    -------
    rasterio.windows.Window
        Window with integer offsets and non-zero integer dimensions.
    """
    window = rasterio.windows.from_bounds(*bounds, transform=transform)
    return rasterio.windows.Window(
        ceil(window.col_off - 1),
        ceil(window.row_off - 1),
        max(1, ceil(window.width)) + 1,
        max(1, ceil(window.height)) + 1,
    )


def simplify_shapes(gdf, transform):
    """Simplify geometries to half of the raster cell size

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing the geometries to simplify.
    transform : affine.Affine
        Affine transform used to derive the raster cell size.

    Returns
    -------
    geopandas.GeoDataFrame
        Input ``gdf`` with simplified geometries and empty features
        removed.
    """
    tolerance = _simplify_tolerance(transform)
    logger.debug(
        "Simplifying %d shape(s) with tolerance %s", len(gdf), tolerance
    )
    gdf.geometry = gdf.geometry.simplify(tolerance, preserve_topology=True)
    gdf = gdf[~gdf.is_empty]
    logger.debug("%d shape(s) remain after simplification", len(gdf))
    return gdf


def _tile_rasterize(
    gdf,
    width,
    height,
    transform,
    tile_size,
    all_touched,
    burn_value,
    boundary_only,
    dtype,
    fill,
):
    """Rasterize shapes window by window into a preallocated array"""
    out = np.zeros((height, width), dtype=dtype)
    shapes = gdf.boundary if boundary_only else gdf.geometry
    use_feature_values = hasattr(burn_value, "loc")

    for window in _iter_tile_windows(width, height, tile_size=int(tile_size)):
        bounds = rasterio.windows.bounds(window, transform)
        tile_geom = box(*bounds)
        intersects = shapes.intersects(tile_geom)

        if not intersects.any():
            continue

        tile_shapes = shapes.loc[intersects].intersection(tile_geom)
        tile_shapes = tile_shapes[~tile_shapes.is_empty]
        if tile_shapes.empty:
            continue

        rasterize_kwargs = {
            "out_shape": (window.height, window.width),
            "fill": np.nan if fill is None else fill,
            "out": None,
            "transform": rasterio.windows.transform(window, transform),
            "all_touched": all_touched,
            "dtype": dtype,
        }
        if use_feature_values:
            tile_values = burn_value.loc[tile_shapes.index]
            rasterize_shapes = zip(tile_shapes, tile_values, strict=True)
        else:
            rasterize_shapes = tile_shapes
            rasterize_kwargs["default_value"] = burn_value

        out[*window.toslices()] = rasterio.features.rasterize(
            rasterize_shapes,
            **rasterize_kwargs,
        )

    return out


def _iter_tile_windows(width, height, tile_size):
    """Yield raster windows covering the full output extent"""
    for row_off in range(0, height, tile_size):
        win_height = min(tile_size, height - row_off)
        for col_off in range(0, width, tile_size):
            win_width = min(tile_size, width - col_off)
            yield rasterio.windows.Window(
                col_off, row_off, win_width, win_height
            )


def _simplify_tolerance(transform):
    """Return simplification tolerance from the raster cell size"""
    x_res = hypot(transform.a, transform.b)
    y_res = hypot(transform.d, transform.e)
    return max(x_res, y_res) * 0.5


def _resolve_burn_values(gdf, burn_value):
    """Resolve scalar or column-based burn values for rasterization"""
    if not isinstance(burn_value, str):
        return burn_value

    if burn_value not in gdf.columns:
        msg = (
            "Rasterize value column "
            f"{burn_value!r} was not found in vector data. "
            f"Available columns: {sorted(gdf.columns)!r}"
        )
        raise revrtValueError(msg)

    return gdf[burn_value]
