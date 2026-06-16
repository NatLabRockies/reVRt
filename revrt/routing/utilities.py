"""reVRt routing module utilities"""

import contextlib
import logging
from pathlib import Path
from warnings import warn
from itertools import batched

import dask
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import Point

from revrt.exceptions import revrtValueError, revrtRuntimeError
from revrt.utilities.base import region_mapper, transform_xy
from revrt.utilities.handlers import IncrementalWriter
from revrt.warn import revrtWarning


logger = logging.getLogger(__name__)
_WHILE_LOOP_ITER_MAX = 10_000


class PointToFeatureMapper:
    """Map points to features within specified regions and/or radii"""

    def __init__(
        self,
        crs,
        features_fp,
        regions=None,
        region_identifier_column="rid",
        connection_identifier_column="end_feat_id",
        batch_size=500,
        max_workers=None,
    ):
        """

        Parameters
        ----------
        crs : str or pyproj.crs.CRS
            Coordinate reference system to use for all spatial data.
        features_fp : path-like
            File path to transmission features to map points to.
        regions : geopandas.GeoDataFrame or path-like, optional
            Regions to use for clipping features around points. Features
            will not extend beyond these regions, and point will only
            map to features within their own region. This input can be
            paired with the `radius` parameter (e.g. to forbid any
            connections to cross state boundaries, even if that
            connection would be shorter than some desired length). At
            least one of `regions` or `radius` must be provided;
            otherwise, an error is raised. By default, ``None``.
        region_identifier_column : str, default="rid"
            Column in `regions` data to use as the identifier for that
            region. If not given, a simple index will be put under the
            `rid` column. By default, ``"rid"``.
        connection_identifier_column : str, optional
            Column in output data (both features and points) that will
            be used to link the points to the features that should be
            routed to. By default, ``"end_feat_id"``.
        batch_size : int, default=500
            Number of points to process in each batch when writing
            clipped features to file. By default, ``500``.
        max_workers : int, optional
            Number of parallel workers to use for point-to-feature
            clipping. If ``None`` or >1, clipping is performed in
            parallel using Dask's process scheduler.
            By default, ``None``, which uses all CPU cores.
        """
        self._crs = crs
        self._features_fp = features_fp
        self._regions = None
        self._rid_column = region_identifier_column
        self._conn_id_column = connection_identifier_column
        self._set_regions(regions)
        self._batch_size = batch_size
        self._max_workers = max_workers

    def _set_regions(self, regions):
        """Set the regions GeoDataFrame"""
        if regions is None:
            return

        try:
            self._regions = regions.to_crs(self._crs)
        except AttributeError:
            self._regions = gpd.read_file(regions).to_crs(self._crs)

        if self._rid_column not in self._regions.columns:
            self._regions[self._rid_column] = range(len(self._regions))

    def map_points(
        self,
        points,
        feature_out_fp,
        route_table_out_fp=None,
        radius=None,
        expand_radius=True,
        clip_points_to_regions=False,
    ):
        """Map points to features within the point region

        Parameters
        ----------
        points : geopandas.GeoDataFrame
            Points to map to features.
        feature_out_fp : path-like
            File path to save clipped features to. This should end in
            '.gpkg' to ensure proper format. If not, the extension will
            be added automatically. This file will contain all features
            clipped to each point (with a feature ID column added to
            link back to the points).
        route_table_out_fp : path-like, optional
            Optional file path to incrementally persist mapped point
            assignments. If the route table and feature outputs already
            exist, the mapper resumes from the previously completed
            points. By default, ``None``.
        radius : float or str, optional
            Radius (in CRS units) around each point to clip features to.
            If `str`, the column in `points` to use for radius values.
            If ``None``, only regions are used for clipping.
            By default, ``None``.
        expand_radius : bool, optional
            If ``True``, the radius is expanded until at least one
            feature is found. By default, ``True``.
        clip_points_to_regions : bool, default=False
            If ``True``, points are clipped to the given regions before
            mapping to features. If the `regions` input is not set,
            this parameter has no effect. If ``False``, all points are
            used as-is, which means points outside of the regions domain
            are mapped to the closest region. By default, ``False``.

        Returns
        -------
        geopandas.GeoDataFrame
            Input points with added column giving mapped feature IDs.
        """
        if self._regions is None and radius is None:
            msg = (
                "Must provide either `regions` or a radius to map points "
                "to features!"
            )
            raise revrtValueError(msg)

        points = points.to_crs(self._crs).copy(deep=True)
        points[self._conn_id_column] = np.nan

        if self._regions is not None:
            points = self._map_points_to_nearest_region(
                points, clip_points_to_regions
            )

        points, next_conn_id = _resume_points(
            points,
            feature_out_fp,
            route_table_out_fp,
            self._conn_id_column,
            self._rid_column,
        )

        writer = _init_streaming_writer(feature_out_fp)
        route_writer = None
        if route_table_out_fp is not None:
            route_writer = IncrementalWriter(route_table_out_fp)

        pending_points = points[points[self._conn_id_column].isna()]
        if pending_points.empty:
            logger.info("All points already mapped; nothing to resume")

        total_pending_points = len(pending_points)
        if total_pending_points:
            logger.info(
                "Mapping %d point(s) to nearby features in batches of %d "
                "using %s workers",
                total_pending_points,
                self._batch_size,
                f"{self._max_workers:,d}"
                if self._max_workers in {None, 0}
                else "all",
            )

        processed_points = 0
        for point_batch in self._iter_point_batches(
            pending_points, start_conn_id=next_conn_id
        ):
            self._run_batch(
                points,
                point_batch,
                writer,
                radius,
                expand_radius,
                route_writer=route_writer,
            )
            processed_points += len(point_batch)
            logger.info(
                "%d/%d (%.2f%%) points processed",
                processed_points,
                total_pending_points,
                processed_points / total_pending_points * 100,
            )

        route_table = self._drop_unpaired_points(points)
        if route_table_out_fp is not None:
            route_table.drop(columns="geometry").to_csv(
                route_table_out_fp, index=False
            )
        return route_table

    def _iter_point_batches(self, points, start_conn_id=0):
        """Generate batches of points for processing"""
        jobs = (
            (ind, row_ind, point)
            for ind, (row_ind, point) in enumerate(
                points.iterrows(), start=start_conn_id
            )
        )
        yield from batched(jobs, self._batch_size)

    def _run_batch(
        self,
        points,
        point_batch,
        writer,
        radius,
        expand_radius,
        route_writer=None,
    ):
        """Process one point batch and flush complete feature chunks"""
        results = []
        routed_point_indices = []
        for row_ind, conn_id, clipped_features in self._compute_point_batch(
            point_batch, radius, expand_radius
        ):
            if clipped_features.empty:
                continue

            clipped_features[self._conn_id_column] = conn_id
            points.loc[row_ind, self._conn_id_column] = conn_id
            results.append(clipped_features)
            routed_point_indices.append(row_ind)

        if not results:
            return

        logger.debug("Dumping batch of %d features to file...", len(results))
        writer.save(pd.concat(results))

        if route_writer is None:
            return

        route_table_batch = points.loc[routed_point_indices].drop(
            columns="geometry"
        )
        route_writer.save(route_table_batch)

    def _compute_point_batch(self, point_batch, radius, expand_radius):
        """Compute clipped features for a batch of points"""
        if self._max_workers == 1:
            return [
                self._clip_point_job(job, radius, expand_radius)
                for job in point_batch
            ]

        num_workers = (
            None if self._max_workers in {None, 0} else self._max_workers
        )
        logger.debug(
            "Processing %d point(s) in parallel with Dask num_workers=%r",
            len(point_batch),
            num_workers,
        )
        tasks = [
            dask.delayed(self._clip_point_job)(job, radius, expand_radius)
            for job in point_batch
        ]
        return list(
            dask.compute(
                *tasks,
                scheduler="processes",
                num_workers=num_workers,
            )
        )

    def _clip_point_job(self, job, radius, expand_radius):
        """Clip features for one point and return point metadata"""
        conn_id, row_ind, point = job
        clipped_features = self._clip_to_point(point, radius, expand_radius)
        return row_ind, conn_id, clipped_features

    def _map_points_to_nearest_region(self, points, clip_points_to_regions):
        """Map points to nearest region; optionally clip to regions"""
        if clip_points_to_regions:
            logger.info("Clipping points to regions...")
            logger.debug("\t- Initial number of points: %d", len(points))
            points = gpd.sjoin(
                points,
                self._regions[["geometry"]],
                predicate="within",
                how="inner",
            ).drop(columns="index_right")
            logger.debug("\t- Final number of points: %d", len(points))

        map_func = region_mapper(self._regions, self._rid_column)
        points[self._rid_column] = points.geometry.apply(map_func)
        return points

    def _clip_to_point(self, point, radius=None, expand_radius=True):
        """Clip features to be within the point region"""
        logger.debug("Clipping features to point:\n%s", point)

        features = None
        if self._regions is not None:
            features = self._clip_to_region(point)

        if radius is not None:
            features = self._clip_to_radius(
                point, radius, features, expand_radius
            )

        return _filter_transmission_features(features)

    def _clip_to_region(self, point):
        """Clip features to region and record the region ID"""
        rid = point[self._rid_column]
        mask = self._regions[self._rid_column] == rid
        region = self._regions[mask]

        logger.debug("  - Clipping features to region: %s", rid)
        features = self._clipped_features(region.geometry, features=None)
        features[self._rid_column] = rid

        logger.debug(
            "%d transmission feature(s) found in region with id %s ",
            len(features),
            rid,
        )
        return features

    def _clip_to_radius(
        self, point, radius, input_features=None, expand_radius=True
    ):
        """Clip features to radius

        If no features are found within the initial radius, it is
        expanded (linearly by incrementally increasing the clipping
        buffer) until at least one connection feature is found.
        """
        if radius is None:
            return input_features

        if input_features is not None and len(input_features) == 0:
            return input_features

        if isinstance(radius, str):
            radius = point[radius]

        clipped_features = self._clipped_features(
            point.geometry.buffer(radius), features=input_features
        )

        clipping_buffer = 1
        ind = 0
        while expand_radius and len(clipped_features) <= 0:
            clipping_buffer += 0.05
            clipped_features = self._clipped_features(
                point.geometry.buffer(radius * clipping_buffer),
                input_features,
            )
            ind += 1
            if ind >= _WHILE_LOOP_ITER_MAX:  # pragma: no cover
                msg = "Maximum iterations reached when expanding radius!"
                raise revrtRuntimeError(msg)

        logger.info(
            "%d transmission features found in clipped area with radius %.2f",
            len(clipped_features),
            radius * clipping_buffer,
        )
        return clipped_features.copy(deep=True)

    def _clipped_features(self, region, features=None):
        """Clip features to region"""
        if features is None:
            features = gpd.read_file(self._features_fp, mask=region).to_crs(
                self._crs
            )

        return gpd.clip(features, region)

    def _drop_unpaired_points(self, points):
        """Drop points that did not map to any features"""
        feature_ids = points[self._conn_id_column]
        if not feature_ids.isna().any():
            return points

        empty_point_indices = feature_ids[feature_ids.isna()].index.tolist()
        msg = (
            f"No features found for {len(empty_point_indices)} point(s) "
            f"at the following indices: {empty_point_indices}"
        )
        warn(msg, revrtWarning)
        points = points[~feature_ids.isna()]
        return points.reset_index(drop=True)


def map_to_costs(route_points, crs, transform, shape):
    """Map route table to cost indices and drop out-of-bounds rows

    Parameters
    ----------
    route_points : pandas.DataFrame
        Route definitions table with at least `start_lat`, `start_lon`,
        `end_lat`, and `end_lon` coordinate columns.
    crs : str or pyproj.crs.CRS
        Coordinate reference system for the cost raster.
    transform : affine.Affine
        Rasterio affine transform giving pixel origin and resolution.
    shape : tuple
        Raster height and width for bounds checking.

    Returns
    -------
    pandas.DataFrame
        Updated route table filtered to routes within the cost domain.
    """
    logger.debug("Map %d routes to cost raster", len(route_points))
    logger.debug("First few routes:\n%s", route_points.head())
    logger.debug("Transform:\n%s", transform)

    route_points = convert_lat_lon_to_row_col(
        route_points,
        crs,
        transform,
        lat_col="start_lat",
        lon_col="start_lon",
        out_row_name="start_row",
        out_col_name="start_col",
    )

    route_points = convert_lat_lon_to_row_col(
        route_points,
        crs,
        transform,
        lat_col="end_lat",
        lon_col="end_lon",
        out_row_name="end_row",
        out_col_name="end_col",
    )

    logger.debug("Mapping done!")

    return filter_points_outside_cost_domain(route_points, shape)


def make_rev_sc_points(excl_rows, excl_cols, crs, transform, resolution=128):
    """Generate reV supply curve points for a given exclusion grid size

    Parameters
    ----------
    excl_rows, excl_cols : int
        Dimensions of the exclusion grid.
    crs : str or pyproj.crs.CRS
        Coordinate reference system for the exclusion raster.
    transform : affine.Affine
        Affine transform object representing the exclusion raster
        transformation.
    resolution : int, default=128
        Resolution of the supply curve grid w.r.t the exclusion grid
        (i.e. reV aggregation factor). By default, ``128``.

    Returns
    -------
    geopandas.GeoDataFrame
        Supply curve points with geometry and start row/column indices.
    """
    n_rows = int(np.ceil(excl_rows / resolution))
    n_cols = int(np.ceil(excl_cols / resolution))

    sc_col_ind, sc_row_ind = np.meshgrid(np.arange(n_cols), np.arange(n_rows))

    sc_points = pd.DataFrame(
        {
            "sc_row_ind": sc_row_ind.flatten().copy(),
            "sc_col_ind": sc_col_ind.flatten().copy(),
        }
    )

    sc_points.index.name = "gid"
    sc_points["sc_point_gid"] = sc_points.index.to_numpy()

    row = np.round(sc_points["sc_row_ind"] * resolution + resolution / 2)
    sc_points["start_row"] = row.astype(int)

    col = np.round(sc_points["sc_col_ind"] * resolution + resolution / 2)
    sc_points["start_col"] = col.astype(int)

    x, y = rasterio.transform.xy(
        transform, sc_points["start_row"].values, sc_points["start_col"].values
    )
    geo = [Point(xy) for xy in zip(x, y, strict=True)]
    sc_points = gpd.GeoDataFrame(sc_points, crs=crs, geometry=geo)
    lon, lat = transform_xy(crs, "EPSG:4326", x, y)
    sc_points["latitude"] = lat
    sc_points["longitude"] = lon
    return gpd.GeoDataFrame(sc_points, crs=crs, geometry=geo)


def points_csv_to_geo_dataframe(points, crs, transform):
    """Convert points CSV or DataFrame to GeoDataFrame

    Parameters
    ----------
    points : path-like or pandas.DataFrame
        Points CSV file path or DataFrame. If a file path is given, the
        file is read as a CSV. The CSV or DataFrame must have `latitude`
        and `longitude` columns.
    crs : str or pyproj.crs.CRS
        Coordinate reference system for the exclusion raster.
    transform : affine.Affine
        Affine transform object representing the exclusion raster
        transformation.

    Returns
    -------
    geopandas.GeoDataFrame
        Points as a GeoDataFrame in the given CRS. Will contain
        `start_row` and `start_col` columns giving the row and column
        indices in the cost raster.
    """
    with contextlib.suppress(TypeError, UnicodeDecodeError):
        points = pd.read_csv(points)

    points = convert_lat_lon_to_row_col(
        points,
        crs,
        transform,
        lat_col="latitude",
        lon_col="longitude",
        out_row_name="start_row",
        out_col_name="start_col",
    )
    return gpd.GeoDataFrame(
        points,
        geometry=gpd.points_from_xy(points.longitude, points.latitude),
        crs="EPSG:4326",
    ).to_crs(crs)


def convert_lat_lon_to_row_col(
    points,
    crs,
    transform,
    lat_col="latitude",
    lon_col="longitude",
    out_row_name="start_row",
    out_col_name="start_col",
):
    """Convert latitude and longitude columns to row and column indices

    Parameters
    ----------
    points : pandas.DataFrame or geopandas.GeoDataFrame
        Data with latitude and longitude columns that need to be
        converted to row and column indices.
    crs : str or pyproj.crs.CRS
        Coordinate reference system for the target grid.
    transform : affine.Affine
        Affine transform for the target grid.
    lat_col : str, default="latitude"
        Name of latitude column in input `point`.
        By default, ``"latitude"``.
    lon_col : str, default="longitude"
        Name of longitude column in input `point`.
        By default, ``"longitude"``.
    out_row_name : str, default="start_row"
        Name of output column for row indices.
        By default, ``"start_row"``.
    out_col_name : str, default="start_col"
        Name of output column for column indices.
        By default, ``"start_col"``.

    Returns
    -------
    pandas.DataFrame or geopandas.GeoDataFrame
        Input `points` with added row and column index columns.
    """
    start_lat = points[lat_col].astype("float32")
    start_lon = points[lon_col].astype("float32")
    start_row, start_col = _transform_lat_lon_to_row_col(
        transform, crs, start_lat, start_lon
    )
    points[out_row_name] = start_row.astype("int32")
    points[out_col_name] = start_col.astype("int32")
    return points


def filter_points_outside_cost_domain(route_table, shape):
    """Drop routes whose indices fall outside the cost domain

    Parameters
    ----------
    route_table : pandas.DataFrame or geopandas.GeoDataFrame
        Route definitions table with at least `start_row`, `start_col`.
        Can also optionally have `end_row`, and `end_col` columns.
    shape : tuple
        Raster height and width for bounds checking.

    Returns
    -------
    pandas.DataFrame or geopandas.GeoDataFrame
        Input `points` filtered to only those within the cost domain.
    """

    logger.debug("Filtering out points outside cost domain...")
    mask = route_table["start_row"] >= 0
    mask &= route_table["start_row"] < shape[0]
    mask &= route_table["start_col"] >= 0
    mask &= route_table["start_col"] < shape[1]
    if "end_row" in route_table.columns:
        mask &= route_table["end_row"] >= 0
        mask &= route_table["end_row"] < shape[0]
    if "end_col" in route_table.columns:
        mask &= route_table["end_col"] >= 0
        mask &= route_table["end_col"] < shape[1]

    logger.debug("Mask computed!")

    if any(~mask):
        msg = (
            f"The following {sum(~mask)} feature(s) are outside of the cost "
            f"domain and will be dropped:\n{route_table.loc[~mask]}"
        )
        warn(msg, revrtWarning)
        route_table = route_table.loc[mask].reset_index(drop=True)

    return route_table


def _transform_lat_lon_to_row_col(transform, cost_crs, lat, lon):
    """Convert WGS84 coordinates to cost grid row and column arrays"""
    x, y = transform_xy("EPSG:4326", cost_crs, lon, lat)
    row, col = rasterio.transform.rowcol(transform, x, y)
    row = np.array(row)
    col = np.array(col)
    return row, col


def _init_streaming_writer(feature_out_fp):
    """Initialize an incremental writer for clipped features"""
    feature_out_fp = Path(feature_out_fp)
    if feature_out_fp.suffix.lower() != ".gpkg":
        msg = (
            "Output feature file should have a '.gpkg' extension to "
            f"ensure proper format! Got input file: '{feature_out_fp}'. "
            "Adding '.gpkg' extension... "
        )
        warn(msg, revrtWarning)
        feature_out_fp = feature_out_fp.with_suffix(".gpkg")

    return IncrementalWriter(feature_out_fp)


def _filter_transmission_features(features):
    """Filter loaded transmission features"""
    features = features.drop(
        columns=["bgid", "egid", "cap_left"], errors="ignore"
    )
    features = features.rename(columns={"gid": "trans_gid"})
    if "category" in features.columns:
        features = _drop_empty_categories(features)

    return features.reset_index(drop=True)


def _drop_empty_categories(features):
    """Drop features with empty category field"""
    mask = features["category"].isna()
    if mask.any():
        msg = f"Dropping {mask.sum():,} feature(s) with NaN category!"
        warn(msg, revrtWarning)
        features = features[~mask].reset_index(drop=True)

    return features


def _resume_points(
    points, feature_out_fp, route_table_out_fp, conn_id_column, rid_column
):
    """Recover previously completed mappings from prior outputs"""
    if feature_out_fp is None or route_table_out_fp is None:
        return points, 0

    feature_out_fp = Path(feature_out_fp)
    route_table_out_fp = Path(route_table_out_fp)
    feature_output_missing = not feature_out_fp.exists()
    route_table_missing = not route_table_out_fp.exists()

    if feature_output_missing or route_table_missing:
        feature_out_fp.unlink(missing_ok=True)
        route_table_out_fp.unlink(missing_ok=True)
        logger.info(
            "Feature output missing: %r; route table missing: %r. "
            "Restarting point mapping...",
            feature_output_missing,
            route_table_missing,
        )
        return points, 0

    route_table = _read_route_table(route_table_out_fp)
    if route_table.empty:
        feature_out_fp.unlink(missing_ok=True)
        route_table_out_fp.unlink(missing_ok=True)
        logger.info("Existing route table is empty; restarting point mapping")
        return points, 0

    key_columns = _resume_key_columns(points, route_table, rid_column)
    existing_route_table = route_table[[*key_columns, conn_id_column]]
    existing_route_table = existing_route_table.drop_duplicates(
        subset=key_columns, keep="last"
    )
    conn_ids = pd.to_numeric(
        existing_route_table[conn_id_column], errors="coerce"
    )
    if conn_ids.isna().any():
        msg = (
            "Cannot resume point-to-feature mapping because the "
            f"existing route table contains invalid "
            f"'{conn_id_column}' values"
        )
        raise revrtValueError(msg)

    existing_lookup = existing_route_table.set_index(key_columns)[
        conn_id_column
    ]
    point_keys = pd.MultiIndex.from_frame(points[key_columns])
    points[conn_id_column] = existing_lookup.reindex(point_keys).to_numpy()

    completed = int(points[conn_id_column].notna().sum())
    logger.info(
        "Resuming point-to-feature mapping from %d completed point(s)",
        completed,
    )
    next_conn_id = int(conn_ids.max()) + 1
    return points, next_conn_id


def _resume_key_columns(points, route_table, rid_column):
    """Determine columns used to match existing route rows"""
    key_columns = ["start_row", "start_col"]
    missing_columns = [
        col
        for col in key_columns
        if col not in points.columns or col not in route_table.columns
    ]
    if missing_columns:
        msg = (
            "Cannot resume point-to-feature mapping because the "
            "existing route table is missing required columns: "
            f"{missing_columns}"
        )
        raise revrtValueError(msg)

    if rid_column in points.columns and rid_column in route_table.columns:
        key_columns.append(rid_column)

    return key_columns


def _read_route_table(route_table_out_fp):
    """Load a route table from disk, tolerating empty files"""
    try:
        return pd.read_csv(route_table_out_fp)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _add_voltage_polarity(route_table, voltages, polarities):
    """Add voltage and polarity columns to route table"""
    if voltages:
        out = []
        for voltage in voltages:
            new_data = route_table.copy(deep=True)
            new_data["voltage"] = voltage
            out.append(new_data)
        route_table = pd.concat(out, ignore_index=True)

    if polarities:
        out = []
        for polarity in polarities:
            new_data = route_table.copy(deep=True)
            new_data["polarity"] = polarity
            out.append(new_data)
        route_table = pd.concat(out, ignore_index=True)
    return route_table
