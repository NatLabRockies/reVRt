"""reVRt collection CLI command"""

import glob
import shutil
import logging
import warnings
from pathlib import Path
from functools import cached_property

import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from gaps.cli import CLICommandFromClass
from gaps.utilities import resolve_path

from revrt.constants import (
    SHORT_CUTOFF,
    MEDIUM_CUTOFF,
    SHORT_MULT,
    MEDIUM_MULT,
)
from revrt.utilities import (
    IncrementalWriter,
    LayeredFile,
    chunked_read_gpkg,
    gpkg_crs,
)
from revrt.utilities.raster import integer_dimension_window
from revrt.exceptions import revrtValueError, revrtFileNotFoundError

logger = logging.getLogger(__name__)


class RouteToFeatureMapper:
    """Class to map routes to transmission features"""

    def __init__(self, cost_fpath, transmission_feature_id_col="trans_gid"):
        """

        Parameters
        ----------
        cost_fpath : path-like
            Filepath to cost layer used for routing. This is used to
            obtain profile information for the raster layer in order to
            compute transmission feature row/column indices.
        transmission_feature_id_col : str, default="trans_gid"
            Name of column in features file uniquely identifying each
            transmission feature. By default, ``"trans_gid"``.
        """
        self._cost_lf = LayeredFile(cost_fpath)
        self._tid_col = transmission_feature_id_col

    def _buffered_endpoint(self, row):
        """Buffered endpoints of a route"""
        x, y = rasterio.transform.xy(
            self._cost_lf.profile["transform"], row["end_row"], row["end_col"]
        )
        return Point(x, y).buffer(self._cost_lf.cell_size * np.sqrt(2) * 1.1)

    def process(self, routes, transmission_feature_fp):
        """Map routes to transmission features

        Parameters
        ----------
        routes : pandas.DataFrame or geopandas.GeoDataFrame
            DataFrame containing routing results with 'end_row' and
            'end_col' columns representing the endpoint indices (into
            the cost raster) of each route.
        transmission_feature_fp : path-like
            Path to GeoPackage file containing transmission features.

        Returns
        -------
        pandas.DataFrame or geopandas.GeoDataFrame
            Input routes DataFrame with transmission feature columns
            merged in based on route endpoints.
        """
        features_crs = gpkg_crs(transmission_feature_fp)

        buffered_endpoints = routes.apply(self._buffered_endpoint, axis=1)
        buffered_endpoints = gpd.GeoDataFrame(
            geometry=buffered_endpoints, crs=self._cost_lf.profile["crs"]
        ).to_crs(features_crs)

        transmission_features = gpd.read_file(
            transmission_feature_fp, mask=buffered_endpoints.union_all()
        )
        transmission_features = self._de_duplicate_preserve_geo(
            transmission_features
        )

        clipped_for_rasterizing = gpd.clip(
            transmission_features, buffered_endpoints
        )

        endpoint_lookup = self._create_endpoint_lookup(
            clipped_for_rasterizing,
        )

        routes_with_features = routes.merge(
            endpoint_lookup, on=["end_row", "end_col"], how="left"
        )
        return routes_with_features.merge(
            transmission_features.drop(columns="geometry"),
            on=[self._tid_col],
            how="left",
            suffixes=("", "_transmission_feature"),
        )

    def _de_duplicate_preserve_geo(self, gdf):
        """De-duplicate GeoDataFrame while preserving geometries"""
        return gpd.GeoDataFrame(
            (
                {
                    **{k: v for k, v in g.iloc[0].items() if k != "geometry"},
                    "geometry": g.union_all(),
                }
                for __, g in gdf.groupby(self._tid_col)
            ),
            crs=gdf.crs,
        )

    def _create_endpoint_lookup(self, transmission_features):
        """Create endpoint lookup DataFrame from transmission feats"""
        transmission_features = transmission_features.to_crs(
            self._cost_lf.profile["crs"]
        )

        endpoint_lookup = []
        for __, feat in transmission_features.iterrows():
            rows, cols = self._feature_to_row_col(feat)
            tid = feat[self._tid_col]
            endpoint_lookup += [
                {
                    "end_row": int(r),
                    "end_col": int(c),
                    self._tid_col: tid,
                }
                for r, c in zip(rows, cols, strict=True)
                if 0 <= r < self._cost_lf.profile["height"]
                and 0 <= c < self._cost_lf.profile["width"]
            ]
        return pd.DataFrame(endpoint_lookup)

    def _feature_to_row_col(self, feat):
        """Get row/col indices for a transmission feature geometry"""
        window = integer_dimension_window(
            feat.geometry.bounds,
            transform=self._cost_lf.profile["transform"],
        )
        window_transform = rasterio.windows.transform(
            window=window, transform=self._cost_lf.profile["transform"]
        )
        mask = rasterio.features.geometry_mask(
            [feat.geometry],
            out_shape=(window.height, window.width),
            transform=window_transform,
            invert=True,
        )
        rows, cols = np.where(mask)
        return rows + window.row_off, cols + window.col_off


class RoutePostProcessor:
    """Class to finalize routing outputs"""

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        collect_pattern,
        project_dir,
        job_name,
        out_dir=None,
        min_line_length=0,
        length_mult_kind=None,
        simplify_geo_tolerance=None,
        cost_fpath=None,
        features_fpath=None,
        transmission_feature_id_col="trans_gid",
        purge_chunks=False,
        chunk_size=10_000,
    ):
        """

        Parameters
        ----------
        collect_pattern : str
            Unix-style ``/filepath/pattern*.gpkg`` representing the
            file(s) to be collected into a single output file. Can also
            just be a path to a single file if data is already in one
            place.
        project_dir : path-like
            Path to project directory. This path is used to resolve the
            out filepath input from the user.
        job_name : str
            Label used to name the generated output file.
        out_dir : path-like, optional
            Directory where finalized routing file should be written.
            If not given, defaults to ``project_dir``.
            By default, ``None``.
        min_line_length : int or float, default=0
            Minimum line length in km. If a line length is below this
            value, its length and cost are adjust to meet this minimum.
            Costs are scaled up linearly. By default, ``0``.
        length_mult_kind : {None, "step", "linear"}, optional
            Type of length multiplier calculation to apply. Length
            multipliers can be used to increase the cost of short lines
            (similar to an economies-of-scale adjustment). Note that
            your input files must contain a ``length_km`` column for
            this to work. The valid options for this input are:
            ``"step"``, which computes length multipliers using a step
            function, ``"linear"``, which computes the length multiplier
            using a linear interpolation between 0 and 10 mile spur-line
            lengths, or ``None``, which indicates no length multipliers
            should be applied. The default for reV runs prior to 2024
            was ``"step"``, after which it was updated to default to
            ``"linear"``. Length multiplier data source:
            https://www.wecc.org/Administrative/TEPPC_TransCapCostCalculator_E3_2019_Update.xlsx
            By default, ``None``.
        simplify_geo_tolerance : float, optional
            Option to simplify geometries before saving to output. Note
            that this changes the path geometry and therefore create a
            mismatch between the geometry and any associated attributes
            (e.g., length, cost, etc). This also means the paths should
            not be used for characterization. If provided, this value
            will be used as the tolerance parameter in the
            `geopandas.GeoSeries.simplify` method. Specifically, all
            parts of a simplified geometry will be no more than
            `tolerance` distance from the original. This value has the
            same units as the coordinate reference system of the
            GeoSeries. Only works for GeoPackage outputs (errors
            otherwise). By default, ``None``.
        cost_fpath : path-like, optional
            Filepath to cost layer used for routing. If provided along
            with `features_fpath`, the routes will be merged with the
            transmission features that they connected to. To skip this
            functionality, leave out **both** `cost_fpath` and
            `features_fpath`. This is used to obtain profile information
            for the cost raster layer in order to compute transmission
            feature row/column indices. By default, ``None``.
        features_fpath : path-like, optional
            Filepath to transmission features GeoPackage. If provided
            along with `cost_fpath`, the routes will be merged with the
            transmission features that they connected to. The features
            in this file must contain a column matching
            `transmission_feature_id_col` that uniquely identifies each
            transmission feature. These features will be merged into the
            routes based on which features the routes connected to. To
            skip this functionality, leave out **both** `cost_fpath` and
            `features_fpath`. By default, ``None``.
        transmission_feature_id_col : str, default="trans_gid"
            Name of column in features file uniquely identifying each
            transmission feature. This is only used if both `cost_fpath`
            and `features_fpath` are provided.
            By default, ``"trans_gid"``.
        purge_chunks : bool, default=False
            Option to delete single-node input files after the
            collection step. By default, ``False``.
        chunk_size : int, default=10_000
            Number of features to read into memory at a time when
            merging files. This helps limit memory usage when merging
            large files. By default, ``10_000``.
        """
        self.collect_pattern = collect_pattern
        self.project_dir = project_dir
        self.job_name = job_name
        self.out_dir = Path(out_dir or project_dir)
        self.min_line_length = min_line_length
        self.length_mult_kind = length_mult_kind
        self.simplify_geo_tolerance = simplify_geo_tolerance
        self.cost_fpath = cost_fpath
        self.features_fpath = features_fpath
        self.transmission_feature_id_col = transmission_feature_id_col
        self.purge_chunks = purge_chunks
        self.chunk_size = chunk_size

        if simplify_geo_tolerance:
            logger.info(
                "Simplifying geometries using a tolerance of %r",
                simplify_geo_tolerance,
            )

    @cached_property
    def files_to_collect(self):
        """list: List of files to collect"""
        collect_pattern = resolve_path(
            self.collect_pattern
            if self.collect_pattern.startswith("/")
            else f"./{self.collect_pattern}",
            self.project_dir,
        )

        files_to_collect = list(glob.glob(str(collect_pattern)))  # noqa
        if not files_to_collect:
            msg = f"No files found using collect pattern: {collect_pattern}"
            raise revrtFileNotFoundError(msg)

        return files_to_collect

    @cached_property
    def file_suffix(self):
        """str: Output file suffix"""
        file_types = {Path(fp).suffix.lower() for fp in self.files_to_collect}
        if len(file_types) > 1:
            msg = (
                "Multiple file types found to collect! All files must be of "
                f"the same type. Found: {file_types}"
            )
            raise revrtValueError(msg)

        return file_types.pop()

    @cached_property
    def out_fp(self):
        """pathlib.Path: Output filepath"""
        return self.out_dir / f"{self.job_name}{self.file_suffix}"

    @cached_property
    def chunk_dir(self):
        """pathlib.Path: Directory for chunk files"""
        return self.out_dir / "chunk_files"

    @cached_property
    def writer(self):
        """revrt.utilities.handlers.IncrementalWriter: Output writer"""
        return IncrementalWriter(self.out_fp)

    @cached_property
    def rtf_mapper(self):
        """RouteToFeatureMapper: Shared route-to-feature mapper obj"""
        return RouteToFeatureMapper(
            cost_fpath=self.cost_fpath,
            transmission_feature_id_col=self.transmission_feature_id_col,
        )

    def _next_file_to_process(self):
        """Generator yielding files to process"""
        num_files = len(self.files_to_collect)
        if not self.purge_chunks:
            logger.debug("Creating chunk dir: %s", self.chunk_dir)
            self.chunk_dir.mkdir(parents=True, exist_ok=True)

        for i, data_fp in enumerate(self.files_to_collect, start=1):
            logger.info("Loading %s (%i/%i)", data_fp, i, num_files)
            logger.debug(
                "\t- Processing file in chunks of %d", self.chunk_size
            )
            yield data_fp
            self._handle_chunk_file(data_fp)

    def process(self):
        """Merge and post-process routes files into a single file

        This function collects together multiple LCP routing output
        files into a single file and applies any specified
        post-processing steps, which include simplifying geometries,
        applying length multipliers, applying a minimum length floor,
        and merging in transmission feature attributes.

        Raises
        ------
        revrtFileNotFoundError
            Raised when no files are found matching the provided
            pattern.
        revrtValueError
            Raised when multiple file types are found matching the
            pattern or if the length multiplier kind is invalid.
        """
        logger.info("Collecting routing outputs to: %s", self.out_fp)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if self.file_suffix == ".gpkg":
                self._collect_geo_files()
            else:
                self._collect_csv_files()

        return str(self.out_fp)

    def _collect_geo_files(self):
        """Collect GeoPackage files into a single output file"""
        for data_fp in self._next_file_to_process():
            for df in chunked_read_gpkg(data_fp, self.chunk_size):
                if self.simplify_geo_tolerance:
                    df.geometry = df.geometry.simplify(
                        self.simplify_geo_tolerance
                    )

                if self.length_mult_kind:
                    df = _apply_length_mult(df, self.length_mult_kind)

                if self.min_line_length > 0:
                    df = _apply_min_length_floor(df, self.min_line_length)

                if self.cost_fpath or self.features_fpath:
                    df = self._merge_transmission_features(df)

                self.writer.save(df)

    def _collect_csv_files(self):
        """Collect CSV files into a single output file"""
        for data_fp in self._next_file_to_process():
            with pd.read_csv(
                data_fp,
                chunksize=self.chunk_size,  # cspell:disable-line
            ) as reader:
                for chunk_idx, df in enumerate(reader):
                    logger.debug("\t\t- Processing CSV chunk %d", chunk_idx)
                    if len(df) == 0:
                        continue

                    if self.length_mult_kind:
                        df = _apply_length_mult(df, self.length_mult_kind)

                    if self.min_line_length > 0:
                        df = _apply_min_length_floor(df, self.min_line_length)

                    if self.cost_fpath or self.features_fpath:
                        df = self._merge_transmission_features(df)

                    self.writer.save(df)

    def _handle_chunk_file(self, chunk_fp):
        """Handle chunk file after collection step"""
        chunk_fp = Path(chunk_fp)
        if self.purge_chunks:
            logger.info("Purging chunk file: %s", chunk_fp)
            chunk_fp.unlink()
        else:
            logger.debug("Retaining chunk file: %s", chunk_fp)
            shutil.move(chunk_fp, self.chunk_dir / chunk_fp.name)

    def _merge_transmission_features(self, routes):
        """Merge transmission feature attributes into routes"""
        if self.cost_fpath is None or self.features_fpath is None:
            msg = (
                "Both `cost_fpath` and `features_fpath` must be provided "
                "to merge transmission feature attributes into routes!"
            )
            raise revrtValueError(msg)

        return self.rtf_mapper.process(routes, self.features_fpath)


def _apply_min_length_floor(features, min_line_length):
    """Apply minimum line length floor to features"""
    if "length_km" not in features.columns:
        msg = "Cannot set minimum line length without 'length_km' column!"
        raise revrtValueError(msg)

    mask = features["length_km"] < min_line_length
    if mask.any():
        msg = (
            "Route length will be increased to the minimum allowed length "
            f"({min_line_length}) for {mask.sum()} route(s)"
        )
        logger.info(msg)
        lengths = features.loc[mask, "length_km"]
        cost_multipliers = np.ones(len(lengths))
        cost_multipliers[lengths > 0] = min_line_length / lengths
        features.loc[mask, "length_km"] = min_line_length
        features.loc[mask, "cost"] *= cost_multipliers

    return features


def _apply_length_mult(features, kind):
    """Apply length multipliers to features based on distance"""
    if "length_km" not in features.columns:
        msg = "Cannot compute length multipliers without 'length_km' column!"
        raise revrtValueError(msg)

    features = _compute_length_mult(features, kind)
    features["raw_cost"] = features["cost"].copy()
    features["cost"] *= features["length_mult"]
    return features


def _compute_length_mult(features, kind="linear"):
    """Compute length multipliers based on user input"""
    if kind.casefold() == "step":
        return _compute_step_wise_lm(features)

    if kind.casefold() == "linear":
        return _compute_linear_lm(features)

    msg = f"Unknown length computation kind: {kind}"
    raise revrtValueError(msg)


def _compute_step_wise_lm(features):
    """Compute length multipliers using step function"""
    features["length_mult"] = 1.0

    # Handle medium cutoff **FIRST**
    mask = features["length_km"] <= MEDIUM_CUTOFF
    features.loc[mask, "length_mult"] = MEDIUM_MULT

    # Short cutoff is second so it can overwrite medium where applicable
    mask = features["length_km"] < SHORT_CUTOFF
    features.loc[mask, "length_mult"] = SHORT_MULT
    return features


def _compute_linear_lm(features):
    """Compute length multipliers using linear interpolation"""

    features["length_mult"] = 1.0
    slope = (1 - SHORT_MULT) / (MEDIUM_CUTOFF - SHORT_CUTOFF / 2)

    mask = features["length_km"] <= MEDIUM_CUTOFF
    features.loc[mask, "length_mult"] = (
        slope * (features.loc[mask, "length_km"] - MEDIUM_CUTOFF) + 1
    )

    return features


finalize_routes_command = CLICommandFromClass(
    init=RoutePostProcessor,
    method="process",
    name="finalize-routes",
    add_collect=False,
)
