"""reVRt point-to-feature routing CLI command"""

import time
import logging
from pathlib import Path
from warnings import warn

import rasterio
import numpy as np
import geopandas as gpd
from gaps.cli import CLICommandFromFunction

from revrt.routing.cli.base import (
    run_lcp,
    route_points_subset,
    split_routes,
    RouteToDefinitionConverter,
)
from revrt.routing.utilities import map_to_costs
from revrt.costs.config import parse_config
from revrt.utilities.raster import integer_dimension_window
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
        cost_layers,
        friction_layers=None,
        barrier_layers=None,
        transmission_config=None,
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
        cost_layers : list
            List of dictionaries defining the layers that are summed to
            determine total costs raster used for routing. Each layer is
            pre-processed before summation according to the user input.
            See the description of
            :func:`revrt.routing.cli.point_to_point.compute_lcp_routes`
            for more details.
        friction_layers : list
            Layers to be multiplied onto the aggregated cost layer to
            influence routing but NOT be reported in final cost
            (i.e. friction, barriers, etc.). See the description of
            :func:`revrt.routing.cli.point_to_point.compute_lcp_routes`
            for more details.
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
            cost_layers=cost_layers,
            friction_layers=friction_layers,
            barrier_layers=barrier_layers,
            transmission_config=transmission_config,
        )
        self.features_fpath = features_fpath
        self.connection_identifier_column = connection_identifier_column

    def _validate_route_points(self):
        """Ensure route points has required columns"""

        if (
            "start_row" not in self.route_points.columns
            or "start_col" not in self.route_points.columns
        ):
            logger.info("Mapping route start points to cost grid...")
            self.route_points = map_to_costs(
                self.route_points,
                crs=self.cost_metadata["crs"],
                transform=self.cost_metadata["transform"],
                shape=self.cost_metadata["shape"],
            )

        super()._validate_route_points()

    def _route_as_tuple(self, row):
        """Convert route row to a tuple for existing route checking"""
        return (
            int(row["start_row"]),
            int(row["start_col"]),
            str(row[self.connection_identifier_column]),
            str(row.get("polarity", "unknown")),
            str(row.get("voltage", "unknown")),
        )

    def _convert_to_route_definitions(self, routes):
        """Convert route DataFrame to route definitions format"""
        start_point_cols = ["start_row", "start_col"]

        route_definitions = []
        route_attrs = {}
        cost_height, cost_width = self.cost_metadata["shape"]
        for route_id, (feat_id, sub_routes) in enumerate(
            routes.groupby(self.connection_identifier_column)
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
                start_idx = tuple(info[start_point_cols].astype("int32"))
                route_attrs[(route_id, start_idx)] = info.to_dict()
                start_points.append(start_idx)

            route_definitions.append(
                (
                    route_id,
                    start_points,
                    [
                        (int(r), int(c))
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


def compute_lcp_routes(  # noqa: PLR0913, PLR0917
    cost_fpath,
    route_table_fpath,
    features_fpath,
    cost_layers,
    out_dir,
    job_name,
    friction_layers=None,
    barrier_layers=None,
    tracked_layers=None,
    cost_multiplier_layer=None,
    cost_multiplier_scalar=1,
    transmission_config=None,
    save_paths=False,
    save_routing_layer=False,
    ignore_invalid_costs=False,
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
    least-cost paths (LCPs) for each route using the cost layers defined
    in the `cost_layers` parameter.

    Parameters
    ----------
    cost_fpath : path-like
        Path to layered Zarr file containing cost and other required
        routing layers.
    route_table_fpath : path-like
        Path to CSV file defining the start points and end features of
        all routes. Must have the following columns:

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

    features_fpath : path-like
        Path to vector file containing features to map points to. This
        file must have a column matching the
        `connection_identifier_column` parameter that maps each feature
        back to the starting points defined in the `route_table`.
    cost_layers : list
        List of dictionaries defining the layers that are summed to
        determine total costs raster used for routing. Each layer is
        pre-processed before summation according to the user input.
        Each dict in the list should have the following keys:

            - "layer_name": (REQUIRED) Name of layer in layered file
              containing cost data.
            - "multiplier_layer": (OPTIONAL) Name of layer in layered
              file containing spatially explicit multiplier values to
              apply to this cost layer before summing it with the
              others. Default is ``None``.
            - "multiplier_scalar": (OPTIONAL) Scalar value to multiply
              this layer by before summing it with the others. Default
              is ``1``.
            - "is_invariant": (OPTIONAL) Boolean flag indicating whether
              this layer is length invariant (i.e. should NOT be
              multiplied by path length; values should be $). Default is
              ``False``.
            - "include_in_final_cost": (OPTIONAL) Boolean flag
              indicating whether this layer should contribute to the
              final cost output for each route in the LCP table.
              Default is ``True``.
            - "include_in_report": (OPTIONAL) Boolean flag indicating
              whether the costs and distances for this layer should be
              output in the final LCP table. Default is ``True``.
            - "apply_row_mult": (OPTIONAL) Boolean flag indicating
              whether the right-of-way width multiplier should be
              applied for this layer. If ``True``, then the transmission
              config should have a "row_width" dictionary that maps
              voltages to right-of-way width multipliers. Also, the
              routing table input should have a "voltage" entry for
              every route. Every "voltage" value in the routing table
              must be given in the "row_width" dictionary in the
              transmission config, otherwise an error will be thrown.
              Default is ``False``.
            - "apply_polarity_mult": (OPTIONAL) Boolean flag indicating
              whether the polarity multiplier should be applied for this
              layer. If ``True``, then the transmission config should
              have a "voltage_polarity_mult" dictionary that maps
              voltages to a new dictionary, the latter mapping
              polarities to multipliers. For example, a valid
              "voltage_polarity_mult" dictionary might be
              ``{"138": {"ac": 1.15, "dc": 2}}``.
              In addition, the routing table input should have a
              "voltage" **and** a "polarity" entry for every route.
              Every "voltage" + "polarity" combination in the routing
              table must be given in the "voltage_polarity_mult"
              dictionary in the transmission config, otherwise an error
              will be thrown.

              .. IMPORTANT::
                 The multiplier in this config is assumed to be in units
                 of "million $ per mile" and will be converted to
                 "$ per pixel" before being applied to the layer!

              Default is ``False``.

        The summed layers define the cost routing surface, which
        determines the cost output for each route. Specifically, the
        cost at each pixel is multiplied by the length that the route
        takes through the pixel, and all of these values are summed for
        each route to determine the final cost.

        .. IMPORTANT::
           If a pixel has a final cost of :math:`\leq 0`, it is treated
           as a barrier (i.e. no paths can ever cross this pixel).

    out_dir : path-like
        Directory where routing outputs should be written.
    job_name : str
        Label used to name the generated output file.
    friction_layers : list, optional
        Layers to be multiplied onto the aggregated cost layer to
        influence routing but NOT be reported in final cost
        (i.e. friction, barriers, etc.). These layers are first
        aggregated, and then the aggregated friction layer is applied
        to the aggregated cost. The cost at each pixel is therefore
        computed as:

        .. math::

            C = (\sum_{i} c_i) * (1 + \sum_{j} f_j)

        where :math:`C` is the final cost at each pixel, :math:`c_i` are
        the individual cost layers, and :math:`f_j` are the individual
        friction layers.

        .. NOTE:: :math:`\sum_{j} f_j` is always clamped to be
           :math:`\gt -1` to prevent zero or negative routing costs.
           In other words, :math:`(1 + \sum_{j} f_j) > 0` always holds.
           This means friction can scale costs to/away from zero but
           never cause the sign of the cost layer to flip (even if
           friction values themselves are negative). This means all
           "barrier" pixels (i.e. cost value :math:`\leq 0`) will remain
           barriers after friction is applied.

        Each item in this list should be a dictionary containing the
        following keys:

            - "multiplier_layer" or "mask": (REQUIRED) Name of layer in
              layered file containing the spatial friction multipliers
              or mask that will be turned into the friction multipliers
              by applying the `multiplier_scalar`.
            - "multiplier_scalar": (OPTIONAL) Scalar value to multiply
              the spatial friction layer by before using it as a
              multiplier on the aggregated costs. Default is ``1``.
            - "include_in_report": (OPTIONAL) Boolean flag indicating
              whether the routing and distances for this layer should be
              output in the final LCP table. Default is ``False``.
            - "apply_row_mult": (OPTIONAL) Boolean flag indicating
              whether the right-of-way width multiplier should be
              applied for this layer. If ``True``, then the transmission
              config should have a "row_width" dictionary that maps
              voltages to right-of-way width multipliers. Also, the
              routing table input should have a "voltage" entry for
              every route. Every "voltage" value in the routing table
              must be given in the "row_width" dictionary in the
              transmission config, otherwise an error will be thrown.
              Default is ``False``.
            - "apply_polarity_mult": (OPTIONAL) Boolean flag indicating
              whether the polarity multiplier should be applied for this
              layer. If ``True``, then the transmission config should
              have a "voltage_polarity_mult" dictionary that maps
              voltages to a new dictionary, the latter mapping
              polarities to multipliers. For example, a valid
              "voltage_polarity_mult" dictionary might be
              ``{"138": {"ac": 1.15, "dc": 2}}``.
              In addition, the routing table input should have a
              "voltage" **and** a "polarity" entry for every route.
              Every "voltage" + "polarity" combination in the routing
              table must be given in the "voltage_polarity_mult"
              dictionary in the transmission config, otherwise an error
              will be thrown.

              .. IMPORTANT::
                 The multiplier in this config is assumed to be in units
                 of "million $ per mile" and will be converted to
                 "$ per pixel" before being applied to the layer!

              Default is ``False``.

        By default, ``None``.
    barrier_layers : list, optional
        Layers defining explicit routing barriers that routes should
        not cross. Unlike `friction_layers`, barrier layers do not add
        a penalty to the routing surface. Instead, any pixel matching a
        barrier definition is treated as blocked during routing.

        Each item in this list should be a dictionary containing the
        following keys:

                - ``"layer_name"``: (REQUIRED) Name of layer in the
                    layered file containing the values to test for
                    barrier cells.
                - ``"barrier_values"``: (REQUIRED) Comparison expression
                    defining which pixel values act as barriers.
                    Supported operators are ``">"``, ``">="``, ``"<"``,
                    ``"<="``, and ``"=="``, followed by a numeric
                    threshold. For example, ``">=15"`` marks all pixels
                    with values greater than or equal to ``15`` as
                    barriers, and ``"==1"`` marks pixels equal to ``1``
                    as barriers.
                - ``"barrier_importance"``: (OPTIONAL) Positive integer
                    ranking used to define a soft barrier. When a route
                    cannot be found, reVRt will iteratively drop the
                    lowest-ranked soft barrier and retry routing until a
                    route is found or all ranked barriers have been
                    removed.

        If ``"barrier_importance"`` is omitted, the barrier is treated
        as a hard barrier and is never relaxed. This allows hard and
        soft barriers to be combined in the same routing run. Multiple
        entries may reference the same layer with different
        ``"barrier_values"`` definitions. By default, ``None``.
    tracked_layers : dict, optional
        Dictionary mapping layer names to strings, where the strings are
        dask aggregation methods (similar to what numpy has) that
        should be applied to the layer along the LCP to be included as a
        characterization column in the output. By default, ``None``.
    cost_multiplier_layer : str, optional
        Name of the spatial multiplier layer applied to final costs.
        By default, ``None``.
    cost_multiplier_scalar : int, default=1
        Scalar multiplier applied to the final cost surface.
        By default, ``1``.
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
    ignore_invalid_costs : bool, optional
        Optional flag to treat any cost values <= 0 as impassable
        (i.e. no paths can ever cross this). If ``False``, cost values
        of <= 0 are set to a large value to simulate a strong but
        permeable "quasi-barrier". By default, ``False``.
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

    start_time = time.time()

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
        cost_layers=cost_layers,
        friction_layers=friction_layers,
        barrier_layers=barrier_layers,
        transmission_config=transmission_config,
        connection_identifier_column=connection_identifier_column,
    )

    run_lcp(
        cost_fpath,
        out_fp=out_fp,
        routes_to_compute=routes_to_compute,
        job_name=job_name,
        cost_multiplier_layer=cost_multiplier_layer,
        cost_multiplier_scalar=cost_multiplier_scalar,
        tracked_layers=tracked_layers,
        ignore_invalid_costs=ignore_invalid_costs,
        user_mem_limit_gb=memory_utilization_limit * system_mem_limit_gb,
        save_routing_layer=save_routing_layer,
        algorithm=algorithm,
    )

    elapsed_time = (time.time() - start_time) / 60
    logger.info("Processing took %.2f minutes", elapsed_time)

    return str(out_fp)


route_features_command = CLICommandFromFunction(
    compute_lcp_routes,
    name="route-features",
    add_collect=False,
    split_keys={"_split_params"},
    config_preprocessor=split_routes,
    skip_doc_params=["system_mem_limit_gb"],
)
