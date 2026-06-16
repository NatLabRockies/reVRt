"""reVRt build route table to features CLI command"""

import logging
from pathlib import Path
from warnings import warn

import geopandas as gpd
import xarray as xr
from gaps.cli import CLICommandFromFunction

from revrt.routing.cli.base import route_points_subset, split_routes
from revrt.routing.utilities import (
    PointToFeatureMapper,
    filter_points_outside_cost_domain,
    make_rev_sc_points,
    points_csv_to_geo_dataframe,
)
from revrt.utilities import strip_path_keys, log_runtime
from revrt.exceptions import revrtValueError
from revrt.warn import revrtWarning


logger = logging.getLogger(__name__)


def point_to_feature_route_table(  # noqa: PLR0913, PLR0917
    cost_fpath,
    features_fpath,
    out_dir,
    regions_fpath=None,
    clip_points_to_regions=False,
    resolution=None,
    radius=None,
    points_fpath=None,
    expand_radius=True,
    feature_out_fp="mapped_features.gpkg",
    route_table_out_fp="route_table.csv",
    region_identifier_column="rid",
    connection_identifier_column="end_feat_id",
    batch_size=500,
    max_workers=1,
    tag=None,
    _split_params=None,
):
    """Create a route table mapping points to nearest features

    This table is used as an input to the routing step, which computes
    the least cost path from each point to its mapped feature. The
    points can be defined by an input csv file or generated from a given
    reV supply curve resolution. The features are defined by an input
    vector file. The points can be optionally clipped to regions defined
    by an input vector file, which limits the features they can be
    mapped to. The output route table contains the row and column index
    of the point (in the cost raster) and the ID of the feature it maps
    to. The mapped features are also output as a separate file that
    contains the same feature IDs for linking to the route table.

    This command can be re-started from a partially completed state by
    providing the same output file paths. The command will check for
    existing outputs and skip any points that have already been mapped
    to features. If the output files are not provided or do not exist,
    the command will start from the beginning and create new output
    files.

    Parameters
    ----------
    cost_fpath : path-like
        Path to cost surface input file. This file will be used to
        define the CRS and transform for mapping points. If the
        `resolution` input is not ``None``, the cost surface grid shape
        will also be used to generate points.
    features_fpath : path-like
        Path to vector file containing features to map points to.
    out_dir : path-like
        Directory where routing outputs should be written.
    points_fpath : path-like, optional
        Path to csv file defining the start points to map to features.
        This input must contain "latitude" and "longitude" columns,
        which define the location of each starting point. This input may
        also have a radius column that defines a unique radius per
        point (see the `radius` input for details). At least one of
        `points_fpath` or `resolution` must be provided.
        By default, ``None``.
    regions_fpath : path-like, optional
        Optional path to file containing region boundaries that should
        be used to clip features. Specifically, points will only be
        mapped to features within their boundary. This can be used to
        limit connections by geography (i.e. within a state or county).
        At least one of `regions_fpath` or `radius` must be provided.
        By default, ``None``.
    clip_points_to_regions : bool, default=False
        If ``True``, points are clipped to the given `regions` before
        mapping to features. If the `regions` input is not set,
        this parameter has no effect. If ``False``, all points are
        used as-is, which means points outside of the regions domain
        are mapped to the closest region. By default, ``False``.
    resolution : int, optional
        reV supply curve point resolution used to generate the supply
        curve point grid. It is assumed that the `cost_fpath` input
        is of the same shape and resolution as the exclusion grid used
        to generate the supply curve points. If `points_fpath` is
        provided, this input is ignored. At least one of `points_fpath`
        or `resolution` must be provided. By default, ``None``.
    radius : str or float, optional
        Distance (in cost CRS units) used to limit radial connection
        length (i.e. a radius input of ``25_000`` in a "meter" CRS will
        limit all connections for the point to be within a circle of
        25 km radius). If this input is a string, it should be the
        column name in the input points that contains a unique radius
        value per point. This input can be combined with the
        `regions_fpath` input (to prevent connections from crossing
        state boundaries, for example). At least one of `regions_fpath`
        or `radius` must be provided. By default, ``None``.
    expand_radius : bool, default=True
        Option to expand the `radius` value for each point until at
        least one feature is found to connect to. This input has no
        effect if `radius` is ``None``. By default, ``True``.
    feature_out_fp : str, default="mapped_features.gpkg"
        Name of output file for mapped (and potentially clipped)
        features. This output file will contain an identifier column
        that can be linked back to the output route table. By default,
        ``"mapped_features.gpkg"``.
    route_table_out_fp : str, default="route_table.csv"
        Name of route table output file. This file will contain a start
        row and column index (into the cost raster) as well as a feature
        ID that maps into the `feature_out_fp` to route to.
        By default, ``"route_table.csv"``.
    region_identifier_column : str, default="rid"
        Column in the `regions_fpath` data used to uniquely identify
        each region. If a column name is given that does not exist in
        the data, it will created using the feature index as the ID.
        By default, ``"rid"``.
    connection_identifier_column : str, default="end_feat_id"
        Column name in the `feature_out_fp` and `route_table_out_fp`
        used to identify which features map to which points.
        By default, ``"end_feat_id"``.
    batch_size : int, default=500
        Number of features to process before writing to output feature
        file. This can be used to tune the tradeoff between performance
        and memory requirements. By default, ``500``.
    max_workers : int, optional
        Number of parallel workers to use for point-to-feature clipping.
        If ``None`` or >1, clipping is performed in parallel using Dask.
        By default, ``1``.

    Returns
    -------
    list of path-like
        Path to route table output file and mapped feature output file.
    """
    with log_runtime("Building point-to-feature routing table"):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        feature_out_fp, route_table_out_fp = _check_output_filepaths(
            out_dir, feature_out_fp, route_table_out_fp
        )
        feature_out_fp, route_table_out_fp = _tag_output_filepaths(
            feature_out_fp,
            route_table_out_fp,
            tag=tag,
            split_params=_split_params,
        )

        logger.debug("Cost input: %r", cost_fpath)
        logger.debug("Features input: %r", features_fpath)
        logger.debug("Output directory: %r", out_dir)

        regions = None
        if regions_fpath is not None:
            regions = gpd.read_file(regions_fpath)

        with xr.open_dataset(
            cost_fpath, consolidated=False, engine="zarr"
        ) as ds:
            crs = ds.rio.crs
            cost_shape = ds.rio.height, ds.rio.width
            transform = ds.rio.transform()

        points = _make_points(
            crs,
            transform,
            cost_shape,
            points_fpath=points_fpath,
            resolution=resolution,
        )
        points = route_points_subset(points, split_params=_split_params)

        mapper = PointToFeatureMapper(
            crs,
            features_fpath,
            regions,
            region_identifier_column=region_identifier_column,
            connection_identifier_column=connection_identifier_column,
            batch_size=batch_size,
            max_workers=max_workers,
        )
        mapper.map_points(
            points,
            feature_out_fp,
            route_table_out_fp=route_table_out_fp,
            radius=radius,
            expand_radius=expand_radius,
            clip_points_to_regions=clip_points_to_regions,
        )

    return [str(route_table_out_fp), str(feature_out_fp)]


def _make_points(
    crs, transform, cost_shape, points_fpath=None, resolution=None
):
    """Create routing points from file or grid"""
    if points_fpath is None and resolution is None:
        msg = (
            "Either `points_fpath` or `resolution` must be provided to "
            "create route table!"
        )
        raise revrtValueError(msg)

    if points_fpath is not None:
        points = points_csv_to_geo_dataframe(points_fpath, crs, transform)
    else:
        points = make_rev_sc_points(
            cost_shape[0], cost_shape[1], crs, transform, resolution=resolution
        )

    return filter_points_outside_cost_domain(points, cost_shape)


def _check_output_filepaths(out_dir, feature_out_fp, route_table_out_fp):
    """Check and fix output filepaths/extensions"""
    feature_out_fp = out_dir / feature_out_fp
    if feature_out_fp.suffix.lower() != ".gpkg":
        msg = (
            "The feature output file should have a '.gpkg' extension to "
            f"ensure proper format! Got input file: '{feature_out_fp}'. "
            "Adding '.gpkg' extension... "
        )
        warn(msg, revrtWarning)
        feature_out_fp = feature_out_fp.with_suffix(".gpkg")

    route_table_out_fp = out_dir / route_table_out_fp
    if route_table_out_fp.suffix.lower() != ".csv":
        msg = (
            "The route table output file should have a '.csv' extension to "
            f"ensure proper format! Got input file: '{route_table_out_fp}'. "
            "Adding '.csv' extension... "
        )
        warn(msg, revrtWarning)
        route_table_out_fp = route_table_out_fp.with_suffix(".csv")

    return feature_out_fp, route_table_out_fp


def _tag_output_filepaths(
    feature_out_fp, route_table_out_fp, tag=None, split_params=None
):
    """Tag node outputs so split runs do not clobber each other"""
    if not tag:
        return feature_out_fp, route_table_out_fp

    __, num_chunks = split_params or (0, 1)
    if num_chunks <= 1:
        return feature_out_fp, route_table_out_fp

    return (
        _tag_filepath(feature_out_fp, tag),
        _tag_filepath(route_table_out_fp, tag),
    )


def _tag_filepath(fp, tag):
    """Insert node tag before file suffix"""
    return fp.with_name(f"{fp.stem}{tag}{fp.suffix}")


def _preprocess_point_to_feature_route_table(config):
    """Preprocess config for point_to_feature_route_table command"""
    config = split_routes(config)
    return strip_path_keys(
        config,
        keys_to_fix={
            "cost_fpath",
            "features_fpath",
            "out_dir",
            "points_fpath",
            "regions_fpath",
        },
    )


build_point_to_feature_route_table_command = CLICommandFromFunction(
    point_to_feature_route_table,
    name="build-feature-route-table",
    add_collect=False,
    split_keys={"_split_params"},
    config_preprocessor=_preprocess_point_to_feature_route_table,
)
