"""reVRt routing CLI functions and helpers"""

import logging
import contextlib
from math import ceil
from pathlib import Path
from copy import deepcopy
from abc import ABC, abstractmethod
from functools import cached_property

import pandas as pd
import rioxarray  # noqa: F401
import geopandas as gpd
import xarray as xr

from revrt.routing.cli.utilities import routing_layer_mover
from revrt.routing.base import RoutingScenario
from revrt.routing.processing import BatchRouteProcessor
from revrt.utilities.monitoring import log_runtime
from revrt.exceptions import revrtKeyError


logger = logging.getLogger(__name__)
_MILLION_USD_PER_MILE_TO_USD_PER_PIXEL = 55923.40730136006
"""Conversion from million dollars/mile to $/pixel

1,000,000 [$/million dollars]
* 90 [meters/pixel]
/ 1609.344 [meters/mile]
= 55923.40730136006 [$/pixel]
"""
_RUST_MEM_FRACTION = 0.75
"""Fraction of user-input memory to use for Rust computations

The remaining memory is assumed to be for the Python process.
"""
_POLARITY = "polarity"
_VOLTAGE = "voltage"


class RouteToDefinitionConverter(ABC):
    """Abstract base class for route definition converters"""

    def __init__(
        self,
        cost_fpath,
        route_points,
        out_fp,
        routing_options,
        transmission_config=None,
        drivers=None,
        transition_costs=None,
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
        barrier_layers : list
            Layers defining explicit hard or soft routing barriers. See
            the description of
            :func:`revrt.routing.cli.point_to_point.compute_lcp_routes`
            for more details.
        transmission_config : path-like or dict, optional
            Dictionary of transmission cost configuration values, or
            path to JSON/JSON5 file containing this dictionary. See the
            description of
            :func:`revrt.routing.cli.point_to_point.compute_lcp_routes`
            for more details.
        """
        self.cost_fpath = cost_fpath
        self.route_points = route_points
        self.out_fp = Path(out_fp)
        self.transmission_config = transmission_config
        self.routing_options = routing_options
        self.drivers = drivers
        self.transition_costs = transition_costs

    @property
    def num_routes(self):
        """int: Number of routes to be computed"""
        return len(self.route_points)

    @cached_property
    def cost_metadata(self):
        """dict: Metadata from cost file (CRS, transform, shape)"""
        with xr.open_dataset(
            self.cost_fpath, consolidated=False, engine="zarr"
        ) as ds:
            return {
                "crs": ds.rio.crs,
                "transform": ds.rio.transform(),
                "shape": ds.rio.shape,
            }

    @cached_property
    def existing_routes(self):
        """set: Already computed routes in the output file"""
        if self.out_fp is None or not self.out_fp.exists():
            return set()

        if self.out_fp.suffix.lower() == ".gpkg":
            existing_df = gpd.read_file(self.out_fp)
        else:
            existing_df = pd.read_csv(self.out_fp)

        return {
            self._route_as_tuple(row) for __, row in existing_df.iterrows()
        }

    @cached_property
    def _default_routing_option(self):
        """str: Default routing option to use if omitted from points"""
        if "default" in self.routing_options:
            return "default"
        return next(iter(self.routing_options))

    @cached_property
    def _group_cols_by_option_value(self):
        """dict: Explicit per-option columns used for batching"""
        return {
            option: {
                value_name: f"{value_name}_{option}"
                for value_name in (_POLARITY, _VOLTAGE)
            }
            for option in self.routing_options
        }

    @cached_property
    def _group_cols(self):
        """list: Explicit per-option columns used for batching"""
        return [
            c
            for cols in self._group_cols_by_option_value.values()
            for c in cols.values()
        ]

    def __iter__(self):
        if self.num_routes == 0:
            return

        for pv_by_option, routes in self._paths_to_compute:
            logger.info(
                "Computing routes for %d points with option values: %s",
                len(routes),
                _format_pv_by_option(pv_by_option),
            )
            route_options = update_route_options(
                self.routing_options,
                pv_by_option=pv_by_option,
                transmission_config=self.transmission_config,
            )
            route_definitions, route_attrs = (
                self._convert_to_route_definitions(routes)
            )
            yield route_options, route_definitions, route_attrs

    @property
    def _paths_to_compute(self):
        """Generator that yields route groups to be computed"""
        for __, routes in self.route_points.groupby(self._group_cols):
            pv_by_option = self._pv_by_option_for_row(routes.iloc[0])
            if self.existing_routes:
                mask = routes.apply(
                    lambda row: (
                        self._route_as_tuple(row) not in self.existing_routes
                    ),
                    axis=1,
                )
                routes = routes[mask]  # noqa: PLW2901

            if routes.empty:
                continue

            yield pv_by_option, routes

    def _pv_by_option_for_row(self, row):
        """dict: Per-option polarity and voltage values for one row"""
        return {
            option: {value_name: row[c] for value_name, c in values.items()}
            for option, values in self._group_cols_by_option_value.items()
        }

    def _route_value_signature(self, row):
        """object: Explicit per-option route values for one route row"""
        return (str(row[c]) for c in self._group_cols)

    @abstractmethod
    def _route_as_tuple(self, row):
        """Convert route row to a tuple for existing route checking"""
        raise NotImplementedError

    @abstractmethod
    def _convert_to_route_definitions(self, routes):
        """Convert route DataFrame to route definitions format"""
        raise NotImplementedError


def run_lcp(
    cost_fpath,
    out_fp,
    routes_to_compute,
    job_name="routes",
    tracked_layers=None,
    invalid_costs_block_routing=True,
    user_mem_limit_gb=4,
    save_routing_layer=False,
    algorithm="long_range_dijkstra",
):
    """[NOT PUBLIC API] Run LCP routing and save to output file"""

    logger.info(
        "Computing best routes for %d point pairs",
        routes_to_compute.num_routes,
    )

    with log_runtime(f"Routing for {routes_to_compute.num_routes:,d} points"):
        _run_all_lcp_batches(
            cost_fpath=cost_fpath,
            out_fp=out_fp,
            routes_to_compute=routes_to_compute,
            job_name=job_name,
            tracked_layers=tracked_layers,
            invalid_costs_block_routing=invalid_costs_block_routing,
            user_mem_limit_gb=user_mem_limit_gb,
            save_routing_layer=save_routing_layer,
            algorithm=algorithm,
        )


def _run_all_lcp_batches(
    cost_fpath,
    out_fp,
    routes_to_compute,
    job_name,
    tracked_layers,
    invalid_costs_block_routing,
    user_mem_limit_gb,
    save_routing_layer,
    algorithm,
):
    """Run LCP routing for all batches of routes and save results"""
    out_fp = Path(out_fp)
    save_paths = out_fp.suffix.lower() == ".gpkg"
    for route_options, route_definitions, route_attrs in routes_to_compute:
        scenario = RoutingScenario(
            cost_fpath=cost_fpath,
            routing_options=route_options,
            drivers=routes_to_compute.drivers,
            transition_costs=routes_to_compute.transition_costs,
            tracked_layers=tracked_layers,
            invalid_costs_block_routing=invalid_costs_block_routing,
            algorithm=algorithm,
        )

        route_computer = BatchRouteProcessor(
            routing_scenario=scenario,
            route_definitions=route_definitions,
            route_attrs=route_attrs,
            mem_limit_gb=user_mem_limit_gb * _RUST_MEM_FRACTION,
        )

        rl_mover = routing_layer_mover(
            save=save_routing_layer,
            cost_fpath=cost_fpath,
            out_fp=out_fp,
            route_attrs=route_attrs,
            job_name=job_name,
            routing_options=route_options,
        )

        with rl_mover as routing_layer_out_fp:
            route_computer.process(
                out_fp=out_fp,
                save_paths=save_paths,
                routing_layer_out_fp=routing_layer_out_fp,
            )


def route_points_subset(route_points, split_params):
    """[NOT PUBLIC API] Get indices of points sorted by location"""

    with contextlib.suppress(TypeError, UnicodeDecodeError):
        route_points = pd.read_csv(route_points)

    sort_cols = ["start_lat", "start_lon"]
    if not set(sort_cols).issubset(route_points.columns):
        sort_cols = ["start_row", "start_col"]

    route_points = route_points.sort_values(sort_cols).reset_index(drop=True)

    start_ind, n_chunks = split_params or (0, 1)
    chunk_size = ceil(len(route_points) / n_chunks)
    return route_points.iloc[
        start_ind * chunk_size : (start_ind + 1) * chunk_size
    ]


def split_routes(config, nodes):
    """[NOT PUBLIC API] Compute route split params inside of config"""
    exec_control = config.get("execution_control", {})
    if mem_limit_gb := exec_control.get("memory"):
        config["system_mem_limit_gb"] = float(mem_limit_gb)

    config["_split_params"] = [(i, nodes) for i in range(nodes)]
    return config


def update_multipliers(layers, polarity, voltage, transmission_config):
    """[NOT PUBLIC API] Update layer multipliers based on user input"""
    output_layers = deepcopy(layers)
    unknowns = {None, "None", "unknown"}
    polarity = "unknown" if polarity in unknowns else str(polarity)
    voltage = "unknown" if voltage in unknowns else str(int(voltage))

    for layer in output_layers:
        if layer.pop("apply_row_mult", False):
            row_multiplier = _get_row_multiplier(transmission_config, voltage)
            layer["multiplier_scalar"] = (
                layer.get("multiplier_scalar", 1) * row_multiplier
            )

        if layer.pop("apply_polarity_mult", False):
            polarity_multiplier = _get_polarity_multiplier(
                transmission_config, voltage, polarity
            )
            layer["multiplier_scalar"] = (
                layer.get("multiplier_scalar", 1)
                * polarity_multiplier
                * _MILLION_USD_PER_MILE_TO_USD_PER_PIXEL
            )

    return output_layers


def update_route_options(
    routing_options, pv_by_option, transmission_config=None
):
    """[NOT PUBLIC API] Update multipliers for multi-option routing"""
    updated_options = deepcopy(routing_options)
    for option_name, option_config in updated_options.items():
        option_polarity = pv_by_option.get(option_name, {}).get("polarity")
        option_voltage = pv_by_option.get(option_name, {}).get("voltage")
        option_config["cost_layers"] = update_multipliers(
            option_config.get("cost_layers", []),
            option_polarity,
            option_voltage,
            transmission_config,
        )
        option_config["friction_layers"] = update_multipliers(
            option_config.get("friction_layers", []),
            option_polarity,
            option_voltage,
            transmission_config,
        )
        option_config["barrier_layers"] = deepcopy(
            option_config.get("barrier_layers", [])
        )

    return updated_options


def _get_row_multiplier(transmission_config, voltage):
    """Get right-of-way width multiplier for a given voltage"""
    try:
        row_widths = transmission_config["row_width"]
    except KeyError as e:
        msg = (
            "`apply_row_mult` was set to `True`, but 'row_width' "
            "not found in transmission config!"
        )
        raise revrtKeyError(msg) from e

    try:
        row_multiplier = row_widths[voltage]
    except KeyError as e:
        msg = (
            "`apply_row_mult` was set to `True`, but voltage '"
            f"{voltage}' not found in transmission config "
            "'row_width' settings. Available voltages: "
            f"{list(row_widths)}"
        )
        raise revrtKeyError(msg) from e

    return row_multiplier


def _get_polarity_multiplier(transmission_config, voltage, polarity):
    """Get multiplier for a given voltage and polarity"""
    try:
        polarity_config = transmission_config["voltage_polarity_mult"]
    except KeyError as e:
        msg = (
            "`apply_polarity_mult` was set to `True`, but "
            "'voltage_polarity_mult' not found in transmission config!"
        )
        raise revrtKeyError(msg) from e

    try:
        polarity_voltages = polarity_config[voltage]
    except KeyError as e:
        msg = (
            "`apply_polarity_mult` was set to `True`, but voltage '"
            f"{voltage}' not found in polarity config. Available voltages: "
            f"{list(polarity_config)}"
        )
        raise revrtKeyError(msg) from e

    try:
        polarity_multiplier = polarity_voltages[polarity]
    except KeyError as e:
        msg = (
            "`apply_polarity_mult` was set to `True`, but polarity ' "
            f"{polarity}' not found in voltage config. Available polarities: "
            f"{list(polarity_voltages)}"
        )
        raise revrtKeyError(msg) from e

    return polarity_multiplier


def _format_pv_by_option(pv_by_option):
    """str: Human-readable per-option route values"""
    formatted_values = []
    for option, values in pv_by_option.items():
        formatted_values.append(
            f"{option}(polarity={values['polarity']!r}, "
            f"voltage={values['voltage']!r})"
        )

    return ", ".join(formatted_values)
