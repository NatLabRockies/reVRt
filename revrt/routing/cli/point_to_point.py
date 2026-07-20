"""reVRt point-to-point routing CLI command"""

import logging
from pathlib import Path

from gaps.cli import CLICommandFromFunction

from revrt.routing.cli.base import (
    run_lcp,
    route_points_subset,
    split_routes,
    RouteToDefinitionConverter,
)
from revrt.utilities import strip_path_keys, log_runtime
from revrt.routing.utilities import map_to_costs
from revrt.costs.config import parse_config


logger = logging.getLogger(__name__)


class PointToPointRouteDefinitionConverter(RouteToDefinitionConverter):
    """Convert route points DataFrame to route definition for Rust"""

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
            int(row["end_row"]),
            int(row["end_col"]),
            str(row.get("end_option", self._routing_options.default)),
            *self._route_value_signature(row),
        )

    def _convert_to_route_definitions(self, routes):  # ruff:ignore[no-self-use]
        """Convert route DataFrame to route definitions format"""
        start_point_cols = ["start_row", "start_col"]
        end_point_cols = ["end_row", "end_col"]
        start_option, end_option = "start_option", "end_option"
        num_unique_start_points = len(routes.groupby(start_point_cols))
        num_unique_end_points = len(routes.groupby(end_point_cols))
        if num_unique_end_points > num_unique_start_points:
            logger.info(
                "Less unique starting points detected! Swapping start and "
                "end point set for optimal routing performance"
            )
            start_point_cols, end_point_cols = end_point_cols, start_point_cols
            start_option, end_option = end_option, start_option

        route_definitions = []
        route_attrs = {}
        for route_id, (end_idx, sub_routes) in enumerate(
            routes.groupby([*end_point_cols, end_option])
        ):
            start_points = []
            for __, info in sub_routes.iterrows():
                start_idx = (
                    *info[start_point_cols].astype("int32"),
                    info[start_option],
                )
                route_attrs[(route_id, start_idx)] = info.to_dict()
                start_points.append(start_idx)

            er, ec, eo = end_idx
            route_definitions.append(
                (
                    route_id,
                    start_points,
                    [(int(er), int(ec), str(eo))],
                )
            )

        return route_definitions, route_attrs


def compute_lcp_routes(  # ruff:ignore[too-many-arguments, too-many-positional-arguments]
    cost_fpath,
    route_table_fpath,
    out_dir,
    job_name,
    routing_options,
    drivers=None,
    transition_costs=None,
    tracked_layers=None,
    transmission_config=None,
    save_paths=False,
    save_routing_layer=False,
    invalid_costs_block_routing=False,
    memory_utilization_limit=0.9,
    system_mem_limit_gb=5,
    _split_params=None,
    algorithm="bidirectional_long_range_dijkstra",
):
    r"""Run least-cost path routing for pairs of points

    Given a table that defines start and end points (via latitude and
    longitude inputs; see the `route_table` parameter), compute the
    least-cost paths (LCPs) between each pair of points using the
    routing layers defined in `routing_options`.

    Parameters
    ----------
    cost_fpath : path-like
        Path to layered Zarr file containing cost and other required
        routing layers.
    route_table_fpath : path-like
        Path to CSV file defining the start and
        end points of all routes. Must have the following columns:

            - "start_lat": Stating point latitude
            - "start_lon": Stating point longitude
            - "end_lat": Ending point latitude
            - "end_lon": Ending point longitude

        You can also specify `polarity` and `voltage` columns which
        apply to every routing option. If you want to provide per-option
        polarity and voltage, use `polarity_<option>` and
        `voltage_<option>`. Options that are omitted will use `polarity`
        and `voltage` column values.
    out_dir : path-like
        Directory where routing outputs should be written.
    job_name : str
        Label used to name the generated output file.
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
    revrt.routing.cli.point_to_feature.compute_lcp_routes
        Compute LCP routes between points and features.
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

        routes_to_compute = PointToPointRouteDefinitionConverter(
            cost_fpath=cost_fpath,
            route_points=route_points,
            out_fp=out_fp,
            routing_options=routing_options,
            transmission_config=transmission_config,
            drivers=drivers,
            transition_costs=transition_costs,
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


def _prep_config(config, nodes):
    """Pre-process config inputs for point-to-point routing"""
    config = split_routes(config, nodes)
    return strip_path_keys(
        config, keys_to_fix={"cost_fpath", "route_table_fpath", "out_dir"}
    )


route_points_command = CLICommandFromFunction(
    compute_lcp_routes,
    name="route-points",
    add_collect=False,
    split_keys={"_split_params"},
    config_preprocessor=_prep_config,
    skip_doc_params=["system_mem_limit_gb"],
)
