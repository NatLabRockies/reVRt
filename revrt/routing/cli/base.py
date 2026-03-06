"""reVRt routing CLI functions and helpers"""

import time
import logging
import contextlib
from math import ceil
from pathlib import Path
from copy import deepcopy
from abc import ABC, abstractmethod
from functools import cached_property

import pandas as pd
import geopandas as gpd
import xarray as xr

from revrt.routing.base import BatchRouteProcessor, RoutingScenario
from revrt.exceptions import revrtKeyError


logger = logging.getLogger(__name__)
_MILLION_USD_PER_MILE_TO_USD_PER_PIXEL = 55923.40730136006
"""Conversion from million dollars/mile to $/pixel

1,000,000 [$/million dollars]
* 90 [meters/pixel]
/ 1609.344 [meters/mile]
= 55923.40730136006 [$/pixel]
"""
_CACHE_MEM_FRACTION = 0.5
"""Fraction of system memory to use for Zarr caching in Rust"""


class RouteToDefinitionConverter(ABC):
    """Abstract base class for route definition converters"""

    _GROUP_COLS = ["polarity", "voltage"]

    def __init__(
        self,
        cost_fpath,
        route_points,
        out_fp,
        cost_layers,
        friction_layers=None,
        transmission_config=None,
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
        self.cost_fpath = cost_fpath
        self.route_points = route_points
        self.out_fp = Path(out_fp)
        self.cost_layers = cost_layers
        self.friction_layers = friction_layers or []
        self.transmission_config = transmission_config

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

    def __iter__(self):
        if self.num_routes == 0:
            return

        for polarity, voltage, routes in self._paths_to_compute:
            logger.info(
                "Computing routes for %d points with polarity: %r and "
                "voltage: %r",
                len(routes),
                polarity,
                voltage,
            )
            route_cl = self._update_cl(polarity, voltage)
            route_fl = self._update_fl(polarity, voltage)
            route_definitions, route_attrs = (
                self._convert_to_route_definitions(routes)
            )
            yield route_cl, route_fl, route_definitions, route_attrs

    @property
    def _paths_to_compute(self):
        """Generator that yields route groups to be computed"""
        self._validate_route_points()

        for group_info, routes in self.route_points.groupby(self._GROUP_COLS):
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

            yield *group_info, routes

    def _validate_route_points(self):
        """Ensure route points has required columns"""
        for check_col in self._GROUP_COLS:
            if check_col not in self.route_points.columns:
                self.route_points[check_col] = "unknown"

    def _update_cl(self, polarity, voltage):
        """Update multipliers for cost layers"""
        return update_multipliers(
            self.cost_layers, polarity, voltage, self.transmission_config
        )

    def _update_fl(self, polarity, voltage):
        """Update multipliers for friction layers"""
        return update_multipliers(
            self.friction_layers, polarity, voltage, self.transmission_config
        )

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
    cost_multiplier_layer=None,
    cost_multiplier_scalar=1,
    tracked_layers=None,
    ignore_invalid_costs=True,
    user_mem_limit_gb=4,
):
    """[NOT PUBLIC API] Run LCP routing and save to output file"""

    ts = time.monotonic()
    out_fp = Path(out_fp)
    save_paths = out_fp.suffix.lower() == ".gpkg"

    logger.info(
        "Computing best routes for %d point pairs",
        routes_to_compute.num_routes,
    )
    for route_batch in routes_to_compute:
        route_cl, route_fl, route_definitions, route_attrs = route_batch
        scenario = RoutingScenario(
            cost_fpath=cost_fpath,
            cost_layers=route_cl,
            friction_layers=route_fl,
            tracked_layers=tracked_layers,
            cost_multiplier_layer=cost_multiplier_layer,
            cost_multiplier_scalar=cost_multiplier_scalar,
            ignore_invalid_costs=ignore_invalid_costs,
        )

        route_computer = BatchRouteProcessor(
            routing_scenario=scenario,
            route_definitions=route_definitions,
            route_attrs=route_attrs,
            mem_limit_gb=user_mem_limit_gb * _CACHE_MEM_FRACTION,
        )
        route_computer.process(out_fp=out_fp, save_paths=save_paths)

    time_elapsed = f"{(time.monotonic() - ts) / 3600:.4f} hour(s)"
    logger.info(
        "Routing for %d points completed in %s",
        routes_to_compute.num_routes,
        time_elapsed,
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


def split_routes(config):
    """[NOT PUBLIC API] Compute route split params inside of config"""
    exec_control = config.get("execution_control", {})
    if exec_control.get("option") == "local":
        num_nodes = 1
    else:
        num_nodes = exec_control.pop("nodes", 1)

    if mem_limit_gb := exec_control.get("memory"):
        config["system_mem_limit_gb"] = float(mem_limit_gb)

    config["_split_params"] = [(i, num_nodes) for i in range(num_nodes)]
    return config


def update_multipliers(layers, polarity, voltage, transmission_config):
    """[NOT PUBLIC API] Update layer multipliers based on user input"""
    output_layers = deepcopy(layers)
    polarity = "unknown" if polarity in {None, "unknown"} else str(polarity)
    voltage = "unknown" if voltage in {None, "unknown"} else str(int(voltage))

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
            "`apply_row_mult` was set to `True`, but voltage ' "
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
            "`apply_polarity_mult` was set to `True`, but voltage ' "
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
