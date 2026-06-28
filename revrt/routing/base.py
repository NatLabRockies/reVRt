"""reVRt base routing functionality"""

import json
import logging
from warnings import warn
from functools import cached_property
from itertools import pairwise

import rasterio
import numpy as np
import xarray as xr
import dask.array as da
from shapely.geometry import Point
from shapely.geometry.linestring import LineString

from revrt import simplify_using_slopes
from revrt.models.cost_layers import BarrierLayer
from revrt.models.routing import (
    validate_driver_configs,
    validate_routing_options,
    validate_transition_cost_configs,
)
from revrt.routing.utilities import compute_lens
from revrt.utilities.parsing import parse_comparison_values
from revrt.exceptions import revrtKeyError
from revrt.warn import revrtWarning, revrtDeprecationWarning


logger = logging.getLogger(__name__)
LCP_AGG_COST_LAYER_NAME = "lcp_agg_costs"
"""Special name reserved for internally-built cost layer"""

ROUTE_SOLUTION_LEN = 3
MIN_ROUTE_POINTS_FOR_TRANSITION_COST = 2


class RoutingScenario:
    """Container for routing scenario configuration"""

    def __init__(
        self,
        cost_fpath,
        routing_options,
        tracked_layers=None,
        drivers=None,
        transition_costs=None,
        invalid_costs_block_routing=True,
        algorithm="bidirectional_long_range_dijkstra",
    ):
        """

        Parameters
        ----------
        cost_fpath : path-like
            Path to the cost layer Zarr store used for routing.
        tracked_layers : list, optional
            List of dictionaries defining layers to summarize along the
            route after applying optional multiplier inputs. See
            :class:`~revrt.models.routing.TrackedLayer` for the
            canonical schema.
        routing_options : dict
            Mapping of routing-option names to dictionaries containing
            cost, friction, and barrier definitions. See
            :class:`~revrt.models.routing.RoutingOptionConfig` for the
            canonical schema that is serialized for the Rust routing
            core.
        drivers : dict, optional
            Optional driver-rule configuration keyed by routing option.
            See :class:`~revrt.models.routing.DriverConfig` and
            :class:`~revrt.models.routing.DriverZoneConfig`.
        transition_costs : dict, optional
            Optional transition-cost configuration between routing
            options. See
            :class:`~revrt.models.routing.TransitionCostsConfig` and
            :class:`~revrt.models.routing.TransitionCostRule`.
        invalid_costs_block_routing : bool, optional
            Flag indicating whether non-positive costs block traversal.
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
        """
        self.cost_fpath = cost_fpath
        self.routing_options = validate_routing_options(routing_options)
        self.drivers = validate_driver_configs(drivers, self.routing_options)
        self.transition_costs = validate_transition_cost_configs(
            transition_costs, self.routing_options
        )
        self.tracked_layers = tracked_layers or []
        self.invalid_costs_block_routing = invalid_costs_block_routing
        self.algorithm = algorithm

    def __repr__(self):
        return (
            "RoutingScenario:"
            f"\n\t- routing_options: {self.routing_options}"
            f"\n\t- drivers: {self.drivers}"
            f"\n\t- transition_costs: {self.transition_costs}"
            f"\n\t- algorithm: {self.algorithm}"
        )

    @property
    def routing_option_names(self):
        """list: Routing option names in solver index order"""
        return list(self.routing_options)

    @cached_property
    def cost_function_json(self):
        """str: JSON string describing configured cost layers"""
        payload = {
            "invalid_costs_block_routing": self.invalid_costs_block_routing,
            "routing_options": self._routing_options_for_rust(),
        }
        if self.drivers is not None:
            payload["drivers"] = _drivers_for_rust(self.drivers)
        if self.transition_costs is not None:
            payload["transition_costs"] = self.transition_costs
        return json.dumps(payload)

    def _routing_options_for_rust(self):
        """dict: Routing options formatted for Rust ingestion"""
        return {
            option_name: {
                "cost_layers": list(
                    _cost_layers_for_rust(option_config.get("cost_layers", []))
                ),
                "friction_layers": list(
                    _friction_layers_for_rust(
                        option_config.get("friction_layers", [])
                    )
                ),
                "barrier_layers": list(
                    _barrier_layers_for_rust(
                        option_config.get("barrier_layers", [])
                    )
                ),
                "cost_multiplier_layer": option_config.get(
                    "cost_multiplier_layer"
                ),
            }
            for option_name, option_config in self.routing_options.items()
        }


class RoutingLayerManager:
    """Class to build routing layers from user input"""

    def __init__(self, routing_scenario, chunks="auto"):
        """

        Parameters
        ----------
        routing_scenario : RoutingScenario
            Scenario containing cost, friction, and tracking metadata.
        chunks : str or mapping, default="auto"
            Chunk specification forwarded to ``xarray.open_dataset``.
            By default, ``"auto"``.
        """
        self.routing_scenario = routing_scenario
        self._layer_fh = xr.open_dataset(
            self.routing_scenario.cost_fpath,
            chunks=chunks,
            consolidated=False,
            engine="zarr",
        )
        self.tracked_layers = []

        self.transform = self._layer_fh.rio.transform()
        self.full_shape = self._layer_fh.rio.shape
        self.cost_crs = self._layer_fh.rio.crs
        self._layers = set(self._layer_fh.variables)

        self.costs = {}
        self.li_costs = {}
        self.final_routing_layers = {}

    def __repr__(self):
        return f"RoutingLayerManager for {self.routing_scenario!r}"

    @property
    def latitudes(self):
        """xarray.DataArray: Latitude coordinates for the cost grid"""
        return self._layer_fh["latitude"]

    @property
    def longitudes(self):
        """xarray.DataArray: Longitude coordinates for the cost grid"""
        return self._layer_fh["longitude"]

    def _verify_layer_exists(self, layer_name):
        """Verify that layer exists in cost file"""
        if layer_name not in self._layers:
            msg = (
                f"Did not find layer {layer_name!r} in cost "
                f"file {str(self.routing_scenario.cost_fpath)!r}"
            )
            raise revrtKeyError(msg)

    def close(self):
        """Close the underlying xarray file handle"""
        self._layer_fh.close()

    def build(self):
        """Build lazy routing arrays from layered file"""

        logger.debug("Building %r", self)
        self._build_cost_layers()
        self._build_tracked_layers()

        return self

    def _build_cost_layers(self):
        """Build a coarse validation layer across routing options"""
        if self.costs:
            return

        for option, config in self.routing_scenario.routing_options.items():
            self._build_cost_layer_from_option(option, config)

    def _empty_cost_layer_data_array(self):
        """xarray.DataArray: Empty routing-cost layer template"""
        template = self.latitudes.astype(np.float32).copy(deep=False)
        template.values = da.zeros(self.full_shape, dtype=np.float32)
        return template

    def _build_cost_layer_from_option(self, option, config):
        """Build a single routing option's cost layer"""
        option_cost = self._empty_cost_layer_data_array()
        option_li_cost = self._empty_cost_layer_data_array()
        option_untracked_cost = self._empty_cost_layer_data_array()
        for layer_info in config.get("cost_layers", []):
            cost = self._extract_and_scale_layer(layer_info)
            cost.values = da.where(cost > 0, cost, 0)
            is_li = layer_info.get("is_invariant", False)
            if layer_info.get("include_in_final_cost", True):
                if is_li:
                    option_li_cost += cost
                else:
                    option_cost += cost
            else:
                option_untracked_cost += cost

            if layer_info.get("include_in_report", True):
                layer_name = f"{layer_info['layer_name']}_{option}"
                self.tracked_layers.append(
                    CharacterizedLayer(
                        layer_name,
                        cost,
                        option=option,
                        is_length_invariant=is_li,
                    )
                )

        mult = config.get("cost_multiplier_scalar", 1) or 1
        option_cost *= mult
        option_li_cost *= mult
        option_untracked_cost *= mult

        multiplier_layer_name = config.get("cost_multiplier_layer")
        if multiplier_layer_name:
            multiplier = self._extract_layer(multiplier_layer_name)
            option_cost *= multiplier
            option_li_cost *= multiplier
            option_untracked_cost *= multiplier

        self.costs[option] = option_cost
        self.li_costs[option] = option_li_cost
        self._build_final_routing_layer_from_option(
            option,
            config,
            option_cost + option_li_cost + option_untracked_cost,
        )

    def _build_final_routing_layer_from_option(
        self, option, config, option_layer
    ):
        frictions = da.zeros(self.full_shape, dtype=np.float32)
        for layer_info in config.get("friction_layers", []):
            layer_name = layer_info.get("multiplier_layer")
            friction_layer = self._extract_and_scale_friction_layer(
                layer_name, layer_info
            )
            if layer_info.get("include_in_report", False):
                self.tracked_layers.append(
                    CharacterizedLayer(
                        f"{layer_name}_{option}", friction_layer, option=option
                    )
                )
            frictions += friction_layer

        frictions = da.where(frictions <= -1, -1.0 + 1e-7, frictions)
        option_layer *= 1 + frictions

        barrier_mask = da.zeros(self.full_shape, dtype=bool)
        for layer_info in config.get("barrier_layers", []):
            if layer_info.get("barrier_importance") is not None:
                continue
            barrier_mask |= self._extract_barrier_layer(
                BarrierLayer(**layer_info).to_routing_dict()
            )

        option_layer.values = da.where(
            option_layer <= 0,
            -1 if self.routing_scenario.invalid_costs_block_routing else 1e10,
            option_layer,
        )
        option_layer.values = da.where(
            barrier_mask,
            da.nan,
            option_layer.values,
        )
        self.final_routing_layers[option] = option_layer

    def _extract_and_scale_layer(self, layer_info):
        """Extract layer based on name and scale according to input"""
        cost = self._extract_layer(layer_info["layer_name"])

        multiplier_layer_name = layer_info.get("multiplier_layer")
        if multiplier_layer_name:
            cost *= self._extract_layer(multiplier_layer_name)

        cost *= layer_info.get("multiplier_scalar", 1)
        return cost

    def _extract_and_scale_friction_layer(self, mask_layer_name, layer_info):
        """Extract layer based on name and scale according to input"""
        if not mask_layer_name:
            msg = (
                "Friction layers must specify a 'mask' or "
                "'multiplier_layer' key!"
            )
            raise revrtKeyError(msg)

        cost = self._extract_layer(mask_layer_name)
        cost *= layer_info.get("multiplier_scalar", 1)
        return cost

    def _extract_barrier_layer(self, layer_info):
        """Extract one barrier layer mask from the layered file"""
        layer = self._extract_layer(layer_info["layer_name"])
        layer_data = getattr(layer, "data", layer)
        threshold = layer_info["barrier_threshold"]
        operator = layer_info["barrier_operator"]

        if operator == "gt":
            return layer_data > threshold
        if operator == "ge":
            return layer_data >= threshold
        if operator == "lt":
            return layer_data < threshold
        if operator == "le":
            return layer_data <= threshold
        if operator == "ne":
            return layer_data != threshold
        if operator == "eq":
            return layer_data == threshold

        msg = (
            "Did not recognize barrier operator "
            f"{operator!r} for layer {layer_info['layer_name']!r}"
        )
        raise revrtKeyError(msg)

    def _extract_layer(self, layer_name):
        """Extract layer based on name"""
        self._verify_layer_exists(layer_name)
        return self._layer_fh[layer_name].isel(band=0).astype(np.float32)

    def _build_tracked_layers(self):
        """Build out a dictionary of tracked layers"""
        for tracked_layer_info in self.routing_scenario.tracked_layers:
            tracked_layer = tracked_layer_info["layer_name"]
            method = tracked_layer_info.get("agg_method")

            if method is None:
                msg = (
                    f"Tracked layer {tracked_layer!r} must specify an "
                    "'agg_method' key! Skipping..."
                )
                warn(msg, revrtWarning)
                continue

            if getattr(da, method, None) is None:
                msg = (
                    f"Did not find method {method!r} in dask.array! "
                    f"Skipping tracked layer {tracked_layer!r}"
                )
                warn(msg, revrtWarning)
                continue

            if tracked_layer not in self._layers:
                msg = (
                    f"Did not find layer {tracked_layer!r} in cost file "
                    f"{str(self.routing_scenario.cost_fpath)!r}. "
                    "Skipping..."
                )
                warn(msg, revrtWarning)
                continue

            layer = self._extract_and_scale_layer(tracked_layer_info)
            is_li = tracked_layer_info.get("is_invariant", False)
            for option in self.routing_scenario.routing_option_names:
                self.tracked_layers.append(
                    CharacterizedLayer(
                        f"{tracked_layer}_{option}",
                        layer,
                        option=option,
                        is_length_invariant=is_li,
                        agg_method=method,
                    )
                )


class CharacterizedLayer:
    """Encapsulate tracked routing layer metadata"""

    def __init__(
        self,
        name,
        layer,
        option,
        is_length_invariant=False,
        agg_method=None,
    ):
        """

        Parameters
        ----------
        name : str
            Identifier used when reporting layer-derived metrics.
        layer : xarray.DataArray or dask.array.Array
            Data values sampled from the routing stack.
        option : str
            Routing option this layer should characterize. When set,
            metrics only use matching route cells.
        is_length_invariant : bool, default=False
            Flag signaling cost values should ignore segment length.
            By default, ``False``.
        agg_method : str, optional
            Name of dask aggregation used when tracking statistics.
            By default, ``None``.
        """
        self.name = name
        self.layer = layer
        self.option = option
        self.is_length_invariant = is_length_invariant
        self.agg_method = agg_method

    def __repr__(self):
        return (
            f"CharacterizedLayer(name={self.name!r}, "
            f"option={self.option!r}, "
            f"is_length_invariant={self.is_length_invariant}, "
            f"agg_method={self.agg_method!r})"
        )

    def compute(self, route, cell_size, point_lens):
        """Compute layer metrics along a route

        Parameters
        ----------
        route : sequence
            Ordered ``(row, col, option)`` indices describing the path.
        cell_size : float
            Raster cell size in meters for distance calculations.
        point_lens : array-like
            Per-route-cell distances aligned with ``route``.

        Returns
        -------
        dict
            Mapping of metric names to aggregated layer values.
        """
        route_points = []
        filtered_point_lens = []
        for p, point_len in zip(route, point_lens, strict=True):
            *point, point_option = p
            if point_option != self.option:
                continue
            route_points.append(point)
            filtered_point_lens.append(point_len)

        if not route_points:
            return self._empty_metrics()

        route_points = np.asarray(route_points)
        filtered_point_lens = np.asarray(filtered_point_lens)

        rows, cols = np.array(route_points).T
        layer_values = self.layer.isel(
            y=xr.DataArray(rows, dims="points"),
            x=xr.DataArray(cols, dims="points"),
        )

        if self.agg_method is None:
            return self._compute_total_and_length(
                layer_values,
                route_points,
                cell_size,
                filtered_point_lens,
            )

        return self._compute_agg(layer_values)

    def _empty_metrics(self):
        """dict: Empty metrics for routes that skip this option"""
        if self.agg_method is None:
            return {
                f"{self.name}_cost": 0,
                f"{self.name}_length_km": 0,
            }

        return {f"{self.name}_{self.agg_method}": np.float32(np.nan)}

    def _compute_total_and_length(
        self, layer_values, route, cell_size, point_lens
    ):
        """Compute total cost and length metrics for the layer"""
        if len(route) == 0:
            return {
                f"{self.name}_cost": 0,
                f"{self.name}_length_km": 0,
            }

        layer_data = getattr(layer_values, "data", layer_values)
        if not isinstance(layer_data, da.Array):  # pragma: no cover
            layer_data = da.asarray(layer_data)
        point_lens = da.asarray(point_lens)

        if self.is_length_invariant:
            layer_cost = da.sum(layer_data)
        else:
            layer_cost = da.sum(layer_data * point_lens)

        layer_length = da.sum(point_lens[layer_data > 0]) * cell_size / 1000

        return {
            f"{self.name}_cost": layer_cost.astype(np.float32).compute(),
            f"{self.name}_length_km": (
                layer_length.astype(np.float32).compute()
            ),
        }

    def _compute_agg(self, layer_values):
        """Compute aggregated statistic for tracked layer"""
        aggregate = getattr(da, self.agg_method)(layer_values).astype(
            np.float32
        )
        return {f"{self.name}_{self.agg_method}": aggregate.compute()}


class RouteMetrics:
    """Class to compute route characteristics given layer cost maps"""

    def __init__(
        self,
        routing_layers,
        route,
        optimized_objective,
        add_geom=False,
        attrs=None,
    ):
        """

        Parameters
        ----------
        routing_layers : RoutingLayerManager
            Routing layer manager containing cost and tracker arrays.
        route : list
            Ordered row and column indices defining the path.
        optimized_objective : float
            Objective value returned by the routing solver.
        add_geom : bool, default=False
            Include shapely geometry in the output when ``True``.
            By default, ``False``.
        attrs : dict, optional
            Additional metadata merged into the result dictionary.
            By default, ``None``.
        """
        self._routing_layers = routing_layers
        self._route = route
        self._optimized_objective = optimized_objective
        self.__lens = self._total_path_length = None
        self._by_layer_results = {}
        self._add_geom = add_geom
        self._attrs = attrs or {}

    @property
    def cell_size(self):
        """float: Raster cell size in meters"""
        return abs(self._routing_layers.transform.a)

    @cached_property
    def _route_options(self):
        """list | None: Route option ids aligned with route points"""
        return np.asarray([p[-1] for p in self._route])

    @cached_property
    def _route_row_col(self):
        """list: List of (row, col) tuples defining the path"""
        return [x[:2] for x in self._route]

    @property
    def _lens(self):
        """array-like: Cached per-cell travel distances"""
        if self.__lens is None:
            self._compute_path_length()
        return self.__lens

    @property
    def total_path_length(self):
        """float: Total path length in kilometers"""
        if self._total_path_length is None:
            self._compute_path_length()
        return self._total_path_length

    @property
    def cost(self):
        """float: Optimized objective evaluated over the route"""
        rows, cols = np.array(self._route_row_col).T
        point_lens = xr.DataArray(self._lens, dims="points")
        total_cost = da.zeros((), dtype=np.float32)

        for option in np.unique(self._route_options):
            mask = self._route_options == option
            option_rows = xr.DataArray(rows[mask], dims="points")
            option_cols = xr.DataArray(cols[mask], dims="points")

            cell_costs = self._routing_layers.costs[option].isel(
                y=option_rows, x=option_cols
            )
            total_cost += da.sum(cell_costs * point_lens[mask])

            inv_cell_costs = self._routing_layers.li_costs[option].isel(
                y=option_rows, x=option_cols
            )
            total_cost += da.sum(inv_cell_costs)

        return total_cost.compute() + self._transition_cost()

    def _transition_cost(self):
        """float: Total transition cost implied by option changes"""
        if len(self._route) < MIN_ROUTE_POINTS_FOR_TRANSITION_COST:
            return 0.0

        default_cost, pairwise_costs = _transition_cost_lookup(
            self._routing_layers.routing_scenario.transition_costs
        )
        return sum(
            pairwise_costs.get((src, dst), default_cost)
            for src, dst in pairwise(self._route_options)
        )

    @property
    def end_lat(self):
        """float: Latitude of the terminal path cell"""
        row, col = self._route_row_col[-1]
        return (
            self._routing_layers.latitudes.isel(y=row, x=col)
            .astype(np.float32)
            .compute()
            .item()
        )

    @property
    def end_lon(self):
        """float: Longitude of the terminal path cell"""
        row, col = self._route_row_col[-1]
        return (
            self._routing_layers.longitudes.isel(y=row, x=col)
            .astype(np.float32)
            .compute()
            .item()
        )

    @property
    def geom(self):
        """shapely.GeometryType: Geometry for the computed path"""
        rows, cols = np.array(self._route_row_col).T
        x, y = rasterio.transform.xy(
            self._routing_layers.transform, rows, cols
        )
        if len(self._route) == 1:
            return Point(x, y)

        return LineString(simplify_using_slopes(list(zip(x, y, strict=True))))

    def compute(self):
        """Assemble route metrics and optional geometry payload"""
        results = {
            "length_km": self.total_path_length,
            "cost": self.cost,
            "poi_lat": self.end_lat,
            "poi_lon": self.end_lon,
            "start_row": self._route[0][0],
            "start_col": self._route[0][1],
            "end_row": self._route[-1][0],
            "end_col": self._route[-1][1],
            "optimized_objective": self._optimized_objective,
        }

        results.update(self._attrs)
        for layer in self._routing_layers.tracked_layers:
            layer_result = layer.compute(
                self._route, self.cell_size, self._lens
            )
            results.update(layer_result)

        if self._add_geom:
            results["geometry"] = self.geom

        return results

    def _compute_path_length(self):
        """Compute the total length and cell by cell length of LCP"""
        self.__lens, self._total_path_length = compute_lens(
            self._route_row_col, self.cell_size
        )


def _transition_cost_lookup(transition_costs):
    """Build the directed transition-cost lookup used in reports"""
    if not transition_costs:
        return 0.0, {}

    default_cost = transition_costs.get("default", 0) or 0
    pairwise_costs = {}
    for rule in transition_costs.get("pairwise", []):
        src, dst = rule["between"]
        pairwise_costs[(src, src)] = 0
        pairwise_costs[(dst, dst)] = 0
        pairwise_costs[(src, dst)] = rule["cost"]
        pairwise_costs[(dst, src)] = rule["cost"]

    return default_cost, pairwise_costs


def _cost_layers_for_rust(layers):
    """Cost layers formatted for Rust ingestion"""
    for layer in layers:
        out_layer = layer.copy()
        out_layer.pop("include_in_report", None)
        out_layer.pop("include_in_final_cost", None)
        yield out_layer


def _friction_layers_for_rust(layers):
    """Friction layers formatted for Rust ingestion"""
    for layer in layers:
        out_layer = layer.copy()
        if "layer_name" in out_layer:
            msg = (
                "Specifying `layer_name` for a friction layer is "
                "deprecated! The default behavior of friction layers is "
                "to be multiplied to the aggregated cost layer. Please "
                "remove this key in order to silence this warning."
            )
            warn(msg, revrtDeprecationWarning)
            out_layer.pop("layer_name")

        if "mask" in out_layer:
            out_layer["multiplier_layer"] = out_layer.pop("mask")

        out_layer.pop("include_in_report", None)
        yield out_layer


def _barrier_layers_for_rust(layers):
    """Barrier layers formatted for Rust ingestion"""
    for layer in layers:
        out_layer = BarrierLayer(**layer).to_routing_dict()
        if out_layer.get("barrier_importance") is None:
            out_layer.pop("barrier_importance")
        yield out_layer


def _drivers_for_rust(drivers):
    """Driver rules formatted for Rust ingestion"""
    out_drivers = drivers.copy()
    if "zones" in drivers:
        out_drivers["zones"] = list(
            _driver_zones_for_rust(drivers.get("zones", []))
        )
    return out_drivers


def _driver_zones_for_rust(zones):
    """Driver zones formatted for Rust ingestion"""
    for zone in zones:
        out_zone = zone.copy()
        where = out_zone.pop("where", None)
        if where is not None:
            mask_operator, mask_threshold = parse_comparison_values(where)
            out_zone["mask_operator"] = mask_operator
            out_zone["mask_threshold"] = mask_threshold
        yield out_zone
