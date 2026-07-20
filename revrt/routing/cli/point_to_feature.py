"""reVRt point-to-feature routing CLI command"""

import logging
import re
from collections import abc
from pathlib import Path
from warnings import warn

import rasterio
import numpy as np
import geopandas as gpd
from gaps.cli import CLICommandFromFunction
from gaps.utilities import TAG
from gaps.pipeline import parse_previous_status

from revrt.routing.cli.base import (
    run_lcp,
    route_points_subset,
    split_routes,
    RouteToDefinitionConverter,
)
from revrt.utilities import strip_path_keys, strip_path, log_runtime
from revrt.routing.utilities import map_to_costs
from revrt.costs.config import parse_config
from revrt.utilities.raster import integer_dimension_window
from revrt.exceptions import revrtConfigurationError
from revrt.warn import revrtWarning


logger = logging.getLogger(__name__)


class PointToFeatureRouteDefinitionConverter(RouteToDefinitionConverter):
    """Convert route points DataFrame to route definition for Rust"""

    def __init__(
        self,
        cost_fpath,
        route_points,
        features_fpath,
        out_fp,
        routing_options,
        transmission_config=None,
        drivers=None,
        transition_costs=None,
        connection_identifier_column="end_feat_id",
    ):
        """

        Parameters
        ----------
        cost_fpath : path-like
            Path to layered Zarr file containing cost and other required
            routing layers.
        route_points : pandas.DataFrame
            DataFrame defining the points to be routed. This DataFrame
            should contain route definitions to be transformed and
            passed down to the Rust routing algorithm.
        out_fp : path-like
            Path to output file where computed routes will be saved.
            This file will be checked for existing routes to avoid
            recomputation.
        routing_options : dict
            Mapping of routing-option names to dictionaries describing
            the cost, friction, barrier, and option-level multiplier
            inputs for each option. See
            :class:`~revrt.models.routing.RoutingOptionConfig`.
        transmission_config : path-like or dict, optional
            Dictionary of transmission cost configuration values, or
            path to JSON/JSON5 file containing this dictionary. See the
            description of
            :func:`revrt.routing.cli.point_to_point.compute_lcp_routes`
            for more details.
        """
        super().__init__(
            cost_fpath=cost_fpath,
            route_points=route_points,
            out_fp=out_fp,
            routing_options=routing_options,
            transmission_config=transmission_config,
            drivers=drivers,
            transition_costs=transition_costs,
        )
        self.features_fpath = features_fpath
        self.connection_identifier_column = connection_identifier_column

    def _rp_with_expected_cols(self):
        """Ensure route points has required columns"""

        if (
            "start_row" not in self._input_route_points.columns
            or "start_col" not in self._input_route_points.columns
        ):
            logger.info("Mapping route start points to cost grid...")
            self._input_route_points = map_to_costs(
                self._input_route_points,
                crs=self.cost_metadata["crs"],
                transform=self.cost_metadata["transform"],
                shape=self.cost_metadata["shape"],
            )

        return super()._rp_with_expected_cols()

    def _route_as_tuple(self, row):
        """Convert route row to a tuple for existing route checking"""
        return (
            int(row["start_row"]),
            int(row["start_col"]),
            str(row.get("start_option", self._routing_options.default)),
            str(row[self.connection_identifier_column]),
            *self._route_value_signature(row),
        )

    def _convert_to_route_definitions(self, routes):
        """Convert route DataFrame to route definitions format"""
        start_point_cols = ["start_row", "start_col"]
        start_option = "start_option"

        route_definitions = []
        route_attrs = {}
        cost_height, cost_width = self.cost_metadata["shape"]
        for route_id, ((feat_id, eo), sub_routes) in enumerate(
            routes.groupby([self.connection_identifier_column, "end_option"])
        ):
            end_feats = gpd.read_file(
                self.features_fpath,
                where=f"{self.connection_identifier_column} == {feat_id}",
            )
            if end_feats.empty:
                msg = (
                    f"No features found with "
                    f"{self.connection_identifier_column} == {feat_id}!"
                )
                warn(msg, revrtWarning)
                continue

            rows, cols = self._end_feats_to_row_col(end_feats)

            start_points = []
            for __, info in sub_routes.iterrows():
                start_idx = (
                    *info[start_point_cols].astype("int32"),
                    info[start_option],
                )
                route_attrs[(route_id, start_idx)] = info.to_dict()
                start_points.append(start_idx)

            route_definitions.append(
                (
                    route_id,
                    start_points,
                    [
                        (int(r), int(c), str(eo))
                        for r, c in zip(rows, cols, strict=True)
                        if 0 <= r < cost_height and 0 <= c < cost_width
                    ],
                )
            )

        return route_definitions, route_attrs

    def _end_feats_to_row_col(self, end_feats):
        """Convert end features to row/col indices in cost grid"""
        window = integer_dimension_window(
            bounds=end_feats.total_bounds,
            transform=self.cost_metadata["transform"],
        )
        window_transform = rasterio.windows.transform(
            window=window, transform=self.cost_metadata["transform"]
        )

        mask = rasterio.features.geometry_mask(
            [end_feats.union_all()],
            out_shape=(window.height, window.width),
            transform=window_transform,
            invert=True,
        )

        rows, cols = np.where(mask)
        rows += window.row_off
        cols += window.col_off
        return rows, cols


def compute_lcp_routes(  # ruff:ignore[too-many-arguments, too-many-positional-arguments]
    cost_fpath,
    out_dir,
    job_name,
    routing_options,
    route_table_fpath="PIPELINE",
    features_fpath="PIPELINE",
    drivers=None,
    transition_costs=None,
    tracked_layers=None,
    transmission_config=None,
    save_paths=False,
    save_routing_layer=False,
    invalid_costs_block_routing=False,
    connection_identifier_column="end_feat_id",
    algorithm="bidirectional_long_range_dijkstra",
    memory_utilization_limit=0.9,
    system_mem_limit_gb=5,
    _split_params=None,
):
    r"""Run least-cost path routing for points mapped to features

    Given a table that defines each route as a start point (via latitude
    and longitude input or preferably a row/column index into the data)
    and a feature ID representing the feature to connect to, compute the
    least-cost paths (LCPs) for each route using the routing layers
    defined in `routing_options`.

    Parameters
    ----------
    cost_fpath : path-like
        Path to layered Zarr file containing cost and other required
        routing layers.
    out_dir : path-like
        Directory where routing outputs should be written.
    job_name : str
        Label used to name the generated output file.
    route_table_fpath : path-like, str, or list, default="PIPELINE"
        Route-table input defining all route start points and end
        features.

        If set to ``"PIPELINE"``, then ``features_fpath`` must also
        be ``"PIPELINE"`` and the route tables will be pulled from
        the previous pipeline step. Pipeline inputs require previous
        outputs containing matching CSV route tables and GPKG
        feature files.

        Otherwise, provide either a single CSV path or a list of
        CSV paths. When a list is provided, ``features_fpath`` must
        also be a list with the same length, and each route table
        is paired with the feature file at the same list index.

        Each route table must have the following columns:

            - "start_lat": Stating point latitude (can alternatively use
              "start_col" to define the start point column index in the
              cost raster).
            - "start_lon": Stating point longitude (can alternatively
              use "start_row" to define the start point row index in the
              cost raster).
            - `connection_identifier_column`: ID of the feature that
              should be mapped to. This ID should match at least one of
              the feature IDs in the `features_fpath` input; otherwise,
              no route will be computed for that point.

              You can also specify `polarity` and `voltage` columns
              which apply to every routing option. If you want to
              provide per-option polarity and voltage, use
              `polarity_<option>` and `voltage_<option>`. Options that
              are omitted will use `polarity` and `voltage` column
              values.

    features_fpath : path-like, str, or list, default="PIPELINE"
        Feature input containing the vector geometries to map points to.

        If set to ``"PIPELINE"``, then ``route_table_fpath`` must
        also be ``"PIPELINE"`` and the feature files will be pulled
        from the previous pipeline step.

        Otherwise, provide either a single vector path or a list of
        vector paths. When a list is provided,
        ``route_table_fpath`` must also be a list with the same
        length, and each feature file is paired with the route table
        at the same list index.

        Each feature file must have a column matching the
        `connection_identifier_column` parameter that maps each
        feature back to the starting points defined in the
        `route_table`.
    routing_options : dict
        Mapping of routing-option names to dictionaries describing the
        cost, friction, barrier, and option-level multiplier inputs for
        each option. See
        :class:`~revrt.models.routing.RoutingOptionConfig` for details.
    drivers : dict, optional
        Optional driver-rule configuration keyed by routing option. See
        :class:`~revrt.models.routing.DriverConfig` for details.
    transition_costs : dict, optional
        Optional transition-cost configuration between routing
        options. See
        :class:`~revrt.models.routing.TransitionCostsConfig` for
        details.
    tracked_layers : list, optional
        List of dictionaries defining route-characterization layers.
        These layers do not influence the routing objective and are
        only summarized for output characterization. See
        :class:`~revrt.models.routing.TrackedLayer` for details.
    transmission_config : path-like or dict, optional
        Dictionary of transmission cost configuration values, or
        path to JSON/JSON5 file containing this dictionary. The
        dictionary should have a subset of the following keys:

            - base_line_costs
            - iso_lookup
            - iso_multipliers
            - land_use_classes
            - new_substation_costs
            - power_classes
            - power_to_voltage
            - transformer_costs
            - upgrade_substation_costs
            - voltage_polarity_mult
            - row_width

        Each of these keys should point to another dictionary or
        path to JSON/JSON5 file containing a dictionary of
        configurations for each section. For the expected contents
        of each dictionary, see the default config. If ``None``,
        values from the default config are used.
        By default, ``None``.
    save_paths : bool, default=False
        Save outputs as a GeoPackage with path geometries when ``True``.
        Defaults to ``False``.
    save_routing_layer : bool, default=False
        Save Rust routing layer outputs to ``out_dir/extra_outputs``
        when ``True``. Defaults to ``False``.
    invalid_costs_block_routing : bool, optional
        Optional flag to treat any invalid cost values (<= 0) as
        impassable (i.e. no paths can ever cross this). If ``False``,
        invalid cost values (<= 0) are set to a large value to simulate
        a strong but permeable "quasi-barrier". By default, ``False``.
    connection_identifier_column : str, default="end_feat_id"
        Column in the `features_fpath` data used to uniquely identify
        each feature. This column is also expected to be in the
        `route_table` input to map points to features. If a column name
        is given that does not exist in the data, an error will be
        raised. By default, ``"end_feat_id"``.
    algorithm : str, default="bidirectional_long_range_dijkstra"
        Routing algorithm implementation to use. Supported values
        are ``"astar"``, ``"long_range_astar"``,
        ``"long_range_dijkstra"``,
        ``"bidirectional_long_range_dijkstra"``, and
        ``"dijkstra"``. ``"astar"`` and ``"dijkstra"`` are
        in-memory implementations that do not respect the memory
        limit. Prefer a long-range option unless you know for a fact
        that your route computations will not need much memory and
        speed is very important to you.
        By default, ``"bidirectional_long_range_dijkstra"``.
    memory_utilization_limit : float, default=0.9
        Fraction of `system_mem_limit_gb` to utilize for routing. Should
        be a value between 0 and 1. By default, ``0.9``.
    system_mem_limit_gb : int or float, default=5
        Maximum amount of system memory (in GB) to utilize for routing.
        This is used in conjunction with `memory_utilization_limit` to
        determine the memory limit for routing. By default, ``5`` GB.

    Returns
    -------
    str or None
        Path to the output table if any routes were computed.

    See Also
    --------
    revrt.routing.cli.point_to_point.compute_lcp_routes
        Compute LCP routes between pairs of points.
    revrt.routing.cli.build_route_table.point_to_feature_route_table
        Helper function to build a routing table for points mapped to
        features.
    """
    with log_runtime("LCP processing"):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.debug("Tracked layers input: %r", tracked_layers)
        logger.debug("Transmission config input: %r", transmission_config)

        transmission_config = parse_config(config=transmission_config)

        route_points = route_points_subset(
            route_table_fpath, split_params=_split_params
        )
        if len(route_points) == 0:
            logger.info("No routes to process!")
            return None

        out_fp = (
            out_dir / f"{job_name}.gpkg"
            if save_paths
            else out_dir / f"{job_name}.csv"
        )

        routes_to_compute = PointToFeatureRouteDefinitionConverter(
            cost_fpath=cost_fpath,
            route_points=route_points,
            features_fpath=features_fpath,
            out_fp=out_fp,
            routing_options=routing_options,
            transmission_config=transmission_config,
            drivers=drivers,
            transition_costs=transition_costs,
            connection_identifier_column=connection_identifier_column,
        )

        run_lcp(
            cost_fpath,
            out_fp=out_fp,
            routes_to_compute=routes_to_compute,
            job_name=job_name,
            tracked_layers=tracked_layers,
            invalid_costs_block_routing=invalid_costs_block_routing,
            user_mem_limit_gb=memory_utilization_limit * system_mem_limit_gb,
            save_routing_layer=save_routing_layer,
            algorithm=algorithm,
        )

    return str(out_fp)


def _prep_config(config, nodes, project_dir, command_name):
    """Pre-process config inputs for point-to-feature routing"""
    config = split_routes(config, nodes)
    config = _handle_route_table_and_features_input(
        config, project_dir, command_name
    )
    return strip_path_keys(config, keys_to_fix={"cost_fpath", "out_dir"})


def _handle_route_table_and_features_input(config, project_dir, command_name):
    """Handle route table and features input from user"""

    rt_is_pipeline = config["route_table_fpath"] == "PIPELINE"
    f_is_pipeline = config["features_fpath"] == "PIPELINE"

    if rt_is_pipeline != f_is_pipeline:
        msg = (
            "Both `route_table_fpath` and `features_fpath` must be set "
            "to 'PIPELINE' for pipeline runs."
        )
        raise revrtConfigurationError(msg)

    if not rt_is_pipeline:
        return _handle_non_pipeline_input(config)

    files = [
        strip_path(fp)
        for fp in parse_previous_status(project_dir, command_name)
    ]
    route_tables, feature_files = _split_pipeline_route_inputs(files)
    config["route_table_fpath"] = route_tables
    config["features_fpath"] = feature_files
    return config


def _split_pipeline_route_inputs(files):
    """Split and align pipeline route-table and feature outputs"""
    route_tables = [fp for fp in files if Path(fp).suffix.lower() == ".csv"]
    feature_files = [fp for fp in files if Path(fp).suffix.lower() == ".gpkg"]

    if not route_tables or not feature_files:
        msg = (
            "Pipeline route-features input requires previous outputs with "
            "both CSV route tables and GPKG feature files."
        )
        raise revrtConfigurationError(msg)

    if len(route_tables) != len(feature_files):
        msg = (
            "Pipeline route-features input requires the same number of "
            "CSV route tables and GPKG feature files."
        )
        raise revrtConfigurationError(msg)

    if len(route_tables) == 1:
        return route_tables, feature_files

    feature_map = _match_pipeline_feature_files(route_tables, feature_files)
    return route_tables, [feature_map[fp] for fp in route_tables]


def _match_pipeline_feature_files(route_tables, feature_files):
    """Match route tables to feature files using the GAPs split tag"""
    route_tags = {_pipeline_file_tag(fp): fp for fp in route_tables}
    feature_tags = {_pipeline_file_tag(fp): fp for fp in feature_files}

    if None in route_tags or None in feature_tags:
        msg = "Pipeline route-features inputs are ambiguously tagged."
        raise revrtConfigurationError(msg)

    if len(route_tags) != len(route_tables) or len(feature_tags) != len(
        feature_files
    ):
        msg = "Pipeline route-features inputs are ambiguously tagged."
        raise revrtConfigurationError(msg)

    if set(route_tags) != set(feature_tags):
        msg = (
            "Could not align pipeline route-table CSV outputs with "
            "mapped-feature GPKG outputs using the GAPs split tag."
        )
        raise revrtConfigurationError(msg)

    return {route_tags[tag]: feature_tags[tag] for tag in sorted(route_tags)}


def _pipeline_file_tag(fp):
    """Extract the GAPs split tag from a pipeline output filepath"""
    match = re.search(rf"({re.escape(TAG)}\d+)$", Path(fp).stem)
    if match is None:
        return None
    return match.group(1)


def _handle_non_pipeline_input(config):
    """Handle non-pipeline input from user"""
    rt_fp = config["route_table_fpath"]
    f_fp = config["features_fpath"]
    rt_is_sequence = isinstance(rt_fp, abc.Sequence) and not isinstance(
        rt_fp, str
    )
    f_is_sequence = isinstance(f_fp, abc.Sequence) and not isinstance(
        f_fp, str
    )

    if rt_is_sequence or f_is_sequence:
        if not (rt_is_sequence and f_is_sequence):
            msg = (
                "`route_table_fpath` and `features_fpath` must both "
                "be sequences or both be strings."
            )
            raise revrtConfigurationError(msg)

        if len(rt_fp) != len(f_fp):
            msg = (
                "`route_table_fpath` and `features_fpath` sequences "
                "must be the same length."
            )
            raise revrtConfigurationError(msg)

        config["route_table_fpath"] = [strip_path(p) for p in rt_fp]
        config["features_fpath"] = [strip_path(p) for p in f_fp]
        return config

    if not isinstance(rt_fp, str) or not isinstance(f_fp, str):
        msg = (
            "`route_table_fpath` and `features_fpath` must both be "
            "sequences or both be strings."
        )
        raise revrtConfigurationError(msg)

    config["route_table_fpath"] = [strip_path(rt_fp)]
    config["features_fpath"] = [strip_path(f_fp)]
    return config


route_features_command = CLICommandFromFunction(
    compute_lcp_routes,
    name="route-features",
    add_collect=False,
    split_keys=[("route_table_fpath", "features_fpath"), "_split_params"],
    config_preprocessor=_prep_config,
    skip_doc_params=["system_mem_limit_gb"],
)
