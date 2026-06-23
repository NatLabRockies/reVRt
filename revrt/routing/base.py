"""reVRt routing from a point to many points"""

import json
import time
import logging
from pathlib import Path
from warnings import warn
from functools import cached_property
from itertools import pairwise

import rasterio
import numpy as np
import xarray as xr
import pandas as pd
import dask.array as da
import geopandas as gpd
from shapely.geometry import Point
from shapely.geometry import MultiLineString
from shapely.geometry.linestring import LineString

from revrt import RouteFinder, simplify_using_slopes
from revrt.models.cost_layers import BarrierLayer
from revrt.utilities.handlers import IncrementalWriter
from revrt.utilities.monitoring import log_runtime
from revrt.exceptions import (
    revrtKeyError,
    revrtLeastCostPathNotFoundError,
    revrtRustError,
)
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
        ignore_invalid_costs=True,
        algorithm="bidirectional_long_range_dijkstra",
    ):
        """

        Parameters
        ----------
        cost_fpath : path-like
            Path to the cost layer Zarr store used for routing.
        tracked_layers : list, optional
            List of dictionaries defining layers to summarize along the
            route after applying optional multiplier inputs. Omitting
            ``"agg_method"`` reports per-cell cost and length for the
            layer; setting it to a dask method name (e.g., ``"mean"``)
            reports that aggregate statistic instead. Each dictionary
            may also include
            ``"multiplier_layer"`` and ``"multiplier_scalar"``.
        routing_options : dict
            Mapping of routing-option names to dictionaries containing
            ``cost_layers``, ``friction_layers``, and
            ``barrier_layers`` entries. This structure is serialized
            for the Rust routing core.
        drivers : dict, optional
            Optional driver-rule configuration keyed by routing option.
        transition_costs : dict, optional
            Optional transition-cost configuration between routing
            options.
        ignore_invalid_costs : bool, optional
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
        self.routing_options = routing_options
        self.drivers = drivers
        self.transition_costs = transition_costs
        self.tracked_layers = tracked_layers or []
        self.ignore_invalid_costs = ignore_invalid_costs
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
            "ignore_invalid_costs": self.ignore_invalid_costs,
            "routing_options": self._routing_options_for_rust(),
        }
        if self.drivers is not None:
            payload["drivers"] = self.drivers
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

    def _build_cost_layer_from_option(self, option, config):
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
                        layer_name, cost, is_length_invariant=is_li
                    )
                )

        option_cost *= config.get("cost_multiplier_scalar", 1) or 1

        self.costs[option] = option_cost
        self.li_costs[option] = option_li_cost
        self._build_final_routing_layer_from_option(
            option,
            config,
            option_cost + option_li_cost + option_untracked_cost,
        )

    def _empty_cost_layer_data_array(self):
        """xarray.DataArray: Empty routing-cost layer template"""
        template = self.latitudes.astype(np.float32).copy(deep=False)
        template.values = da.zeros(self.full_shape, dtype=np.float32)
        return template

    def _build_final_routing_layer_from_option(
        self, option, config, option_layer
    ):
        frictions = da.zeros(self.full_shape, dtype=np.float32)
        for layer_info in config.get("friction_layers", []):
            layer_name = (
                layer_info["mask"]
                if "mask" in layer_info
                else layer_info.get("multiplier_layer")
            )
            friction_layer = self._extract_and_scale_friction_layer(
                layer_name, layer_info
            )
            if layer_info.get("include_in_report", False):
                self.tracked_layers.append(
                    CharacterizedLayer(
                        f"{layer_name}_{option}", friction_layer
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
            -1 if self.routing_scenario.ignore_invalid_costs else 1e10,
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

        multiplier_layer_name = layer_info.get(
            "mask", layer_info.get("multiplier_layer")
        )
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

    # def _build_barrier_mask(self):
    #     """Build a mask for always-active explicit barriers"""
    #     barrier_mask = da.zeros(self._full_shape, dtype=bool)
    #     for layer_info in self._iter_hard_barrier_layers():
    #         barrier_mask |= self._extract_barrier_layer(layer_info)
    #     return barrier_mask

    # def _iter_hard_barrier_layers(self):
    #     """Yield barrier layers without retry importance"""
    #     for layer_info in self.routing_scenario.barrier_layers:
    #         if layer_info.get("barrier_importance") is None:
    #             yield BarrierLayer(**layer_info).to_routing_dict()

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
            self.tracked_layers.append(
                CharacterizedLayer(
                    tracked_layer,
                    layer,
                    is_length_invariant=is_li,
                    agg_method=method,
                )
            )


class CharacterizedLayer:
    """Encapsulate tracked routing layer metadata"""

    def __init__(
        self, name, layer, is_length_invariant=False, agg_method=None
    ):
        """

        Parameters
        ----------
        name : str
            Identifier used when reporting layer-derived metrics.
        layer : xarray.DataArray or dask.array.Array
            Data values sampled from the routing stack.
        is_length_invariant : bool, default=False
            Flag signaling cost values should ignore segment length.
            By default, ``False``.
        agg_method : str, optional
            Name of dask aggregation used when tracking statistics.
            By default, ``None``.
        """
        self.name = name
        self.layer = layer
        self.is_length_invariant = is_length_invariant
        self.agg_method = agg_method

    def __repr__(self):
        return (
            f"CharacterizedLayer(name={self.name!r}, "
            f"is_length_invariant={self.is_length_invariant}, "
            f"agg_method={self.agg_method!r})"
        )

    def compute(self, route, cell_size):
        """Compute layer metrics along a route

        Parameters
        ----------
        route : sequence
            Ordered ``(row, col)`` indices describing the path.
        cell_size : float
            Raster cell size in meters for distance calculations.

        Returns
        -------
        dict
            Mapping of metric names to aggregated layer values.
        """
        rows, cols = np.array(route).T
        layer_values = self.layer.isel(
            y=xr.DataArray(rows, dims="points"),
            x=xr.DataArray(cols, dims="points"),
        )

        if self.agg_method is None:
            return self._compute_total_and_length(
                layer_values, route, cell_size
            )

        return self._compute_agg(layer_values)

    def _compute_total_and_length(self, layer_values, route, cell_size):
        """Compute total cost and length metrics for the layer"""
        if len(route) == 1:
            return {
                f"{self.name}_cost": 0,
                f"{self.name}_length_km": 0,
            }

        lens, __ = _compute_lens(route, cell_size)

        layer_data = getattr(layer_values, "data", layer_values)
        if not isinstance(layer_data, da.Array):  # pragma: no cover
            layer_data = da.asarray(layer_data)

        if self.is_length_invariant:
            layer_cost = da.sum(layer_data)
        else:
            layer_cost = da.sum(layer_data * lens)

        layer_length = da.sum(lens[layer_data > 0]) * cell_size / 1000

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
            layer_result = layer.compute(self._route_row_col, self.cell_size)
            results.update(layer_result)

        if self._add_geom:
            results["geometry"] = self.geom

        return results

    def _compute_path_length(self):
        """Compute the total length and cell by cell length of LCP"""
        self.__lens, self._total_path_length = _compute_lens(
            self._route_row_col, self.cell_size
        )


class IncrementalRouteWriter(IncrementalWriter):
    """Stream results to disk by appending each new result to a file

    A new file is created if one does not exist.
    """

    def __init__(self, out_fp, crs=None):
        """

        Parameters
        ----------
        out_fp : path-like
            Path to output file.
        crs : rasterio.crs.CRS or dict, optional
            Coordinate reference system for geometries when saving to
            GeoPackage. By default, ``None``.
        """
        super().__init__(out_fp)
        self.crs = crs

    def preprocess_chunk(self, result):
        """Turn result into a dataframe chunk

        Parameters
        ----------
        result : dict
            Route result dictionary as built by
            ``RouteMetrics.compute()``.

        Returns
        -------
        pandas.DataFrame or geopandas.GeoDataFrame
            A dataframe holding the route result.
        """
        if "geometry" in result:
            return gpd.GeoDataFrame(
                [result], geometry="geometry", crs=self.crs
            )
        return pd.DataFrame([result])


class BatchRouteProcessor:
    """Class to manage batches of route computations"""

    def __init__(
        self,
        routing_scenario,
        route_definitions,
        route_attrs=None,
        mem_limit_gb=4,
    ):
        """

        Parameters
        ----------
        routing_scenario : RoutingScenario
            Scenario describing the cost layers and routing options.
        route_definitions : Iterable
            Sequence of ``(start_points, end_points, option)`` tuples
            defining which points to route between. Each of
            ``start_points`` and ``end_points`` should be a list of
            ``(row, col, option_name)`` tuples.
        route_attrs : dict, optional
            Mapping of tuples of the form (int, (int, int, option))
            where the first integer represents the route ID and the
            tuple of integers + str represents the starting index to
            additional attributes to include in the output for that
            route. By default, ``None``.
        mem_limit_gb : int or float, default=4
            Memory limit in gigabytes for routing computations.
            By default, ``4``.
        """
        self.routing_scenario = routing_scenario
        self._route_definitions = route_definitions
        self._route_attrs = route_attrs or {}
        self.mem_limit_gb = mem_limit_gb

    @cached_property
    def default_attrs(self):
        """dict: Default attributes for all routes"""
        keys = set().union(*[set(x) for x in self._route_attrs.values()])
        return dict.fromkeys(keys)

    @cached_property
    def route_attrs(self):
        """dict: Mapping of frozen route node pair sets to attributes"""
        return {
            k: {**self.default_attrs, **v}
            for k, v in self._route_attrs.items()
        }

    @cached_property
    def route_definitions(self):
        """list: Validated route definitions for computation"""
        return self._compile_valid_route_definitions()

    @cached_property
    def routing_layers(self):
        """RoutingLayerManager: Built routing layers for the scenario"""
        return RoutingLayerManager(self.routing_scenario).build()

    def process(self, out_fp, save_paths=False, routing_layer_out_fp=None):
        """Compute all routes and save to disk

        Parameters
        ----------
        out_fp : path-like
            Path to output file. If ``save_paths=True``, a GeoPackage
            will be created (recommend to pass in a filepath ending in
            ".gpkg"). Otherwise, a CSV file will be created (recommend
            to pass in a filepath ending in ".csv").
        save_paths : bool, default=False
            Include shapely geometries in the output when ``True``.
            By default, ``False``.
        routing_layer_out_fp : path-like, optional
            Optional output path for Rust routing-layer cache data.
            By default, ``None``.
        """
        if not self.route_definitions:
            return

        with log_runtime(
            f"Routing for {len(self.route_definitions)} route definitions"
        ):
            try:
                self._compute_routes(
                    out_fp, save_paths=save_paths, rl=routing_layer_out_fp
                )
            finally:
                self._reset_routing_layers()

    def _compute_routes(self, out_fp, save_paths, rl=None):
        """Evaluate route definitions and build result records"""

        out_fp = _validate_out_fp(out_fp, save_paths)
        writer = IncrementalRouteWriter(
            out_fp, crs=self.routing_layers.cost_crs
        )
        option_writer = None
        if save_paths and self.routing_scenario.routing_options is not None:
            option_writer = IncrementalRouteWriter(
                _routing_options_output_fp(out_fp),
                crs=self.routing_layers.cost_crs,
            )

        for indices, optimized_objective, attrs in self._route_results(rl):
            metrics = RouteMetrics(
                self.routing_layers,
                indices,
                optimized_objective,
                add_geom=save_paths,
                attrs=attrs,
            )
            route_result = metrics.compute()
            writer.save(route_result)
            if option_writer is not None:
                for option_result in self._routing_option_results(
                    indices, route_result
                ):
                    option_writer.save(option_result)

    def _routing_option_results(self, indices, route_result):
        """Yield aggregated geometries for each routing option used"""

        segments_by_option = {}
        current_option = None
        current_segment = []
        for start_p, end_p in pairwise(indices):
            if start_p == end_p:
                continue

            start = tuple(start_p[:2])
            end = tuple(end_p[:2])
            start_point_option = start_p[-1]
            if (
                current_option == start_point_option
                and current_segment[-1] == start
            ):
                current_segment.append(end)
                continue

            if current_segment:
                segments_by_option.setdefault(current_option, []).append(
                    current_segment
                )
            current_option = start_point_option
            current_segment = [start, end]

        if current_segment:
            segments_by_option.setdefault(current_option, []).append(
                current_segment
            )

        cell_size = abs(self.routing_layers.transform.a)
        results = []
        for option, segments in segments_by_option.items():
            geoms = [self._component_geometry(segment) for segment in segments]
            if not geoms:
                continue

            geometry = (
                geoms[0]
                if len(geoms) == 1
                else MultiLineString([list(geom.coords) for geom in geoms])
            )
            length_km = sum(
                _compute_lens(segment, cell_size)[1] for segment in segments
            )
            results.append(
                {
                    **{
                        key: value
                        for key, value in route_result.items()
                        if key
                        not in {
                            "geometry",
                            "cost",
                            "optimized_objective",
                            "length_km",
                        }
                    },
                    "routing_option": option,
                    "length_km": length_km,
                    "geometry": geometry,
                }
            )

        return results

    def _component_geometry(self, route):
        """Build geometry for one contiguous routing-option segment"""
        rows, cols = np.array(route).T
        x, y = rasterio.transform.xy(self.routing_layers.transform, rows, cols)
        if len(route) == 1:
            return Point(x, y)

        return LineString(simplify_using_slopes(list(zip(x, y, strict=True))))

    def _route_results(self, routing_layer_out_fp=None):
        """Generator yielding route results from Rust computations"""
        if not self.route_definitions:
            return

        logger.debug(
            "Setting memory limit to %.2f GB for Rust computations",
            self.mem_limit_gb,
        )
        route_results = RouteFinder(
            zarr_fp=self.routing_scenario.cost_fpath,
            cost_function=self.routing_scenario.cost_function_json,
            route_definitions=[
                (rid, sp, ep)
                for rid, (sp, ep) in self.route_definitions.items()
            ],
            mem_limit_bytes=int(self.mem_limit_gb * 1_000_000_000),
            algorithm=self.routing_scenario.algorithm,
            log_level=logging.getLogger("revrt").level or None,
            routing_layer_out_fp=routing_layer_out_fp,
        )
        yield from self._skip_failed_routes(route_results)

    def _compile_valid_route_definitions(self):
        """Filter route definitions to those with valid route nodes"""
        if not self._route_definitions:
            return {}

        sample_definition = self._route_definitions[0]
        if len(sample_definition) == 2:  # noqa: PLR2004
            self._route_definitions = _add_route_ids(self._route_definitions)

        routes_to_compute = {}
        for route_id, start_points, end_points in self._route_definitions:
            filtered_start_points = self._validate_start_points(start_points)
            if not filtered_start_points:
                msg = (
                    f"All start points are invalid for route with ID "
                    f"{route_id}: {start_points}\nSkipping..."
                )
                warn(msg, revrtWarning)
                continue

            try:
                filtered_end_points = self._validate_end_points(end_points)
            except revrtLeastCostPathNotFoundError:
                continue

            if not filtered_end_points:
                msg = (
                    f"All end points are invalid for route with ID "
                    f"{route_id}: {end_points}\nSkipping..."
                )
                warn(msg, revrtWarning)
                continue

            routes_to_compute[route_id] = (
                filtered_start_points,
                filtered_end_points,
            )

        return routes_to_compute

    def _skip_failed_routes(self, routing_results):
        """Yield only successfully computed routes from Rust results"""

        results_iter = iter(routing_results)
        num_complete = 0
        ts = time.monotonic()
        while True:
            num_complete += 1
            try:
                route_id, solutions = next(results_iter)
                yield from self._formatted_solutions(solutions, route_id)
                time_elapsed = f"{(time.monotonic() - ts) / 60:.2f} minute(s)"
                logger.info(
                    "%d/%d (%.2f%%) route definitions processed in %s",
                    num_complete,
                    len(self.route_definitions),
                    (num_complete / len(self.route_definitions)) * 100,
                    time_elapsed,
                )
            except revrtRustError:  # pragma: no cover
                logger.exception("Rust error when computing route")
                continue
            except StopIteration:
                logger.info("Routing complete")
                break

    def _formatted_solutions(self, solutions, route_id):
        """Format reVRt output solutions and log any failures"""
        start_points, end_points = self.route_definitions[route_id]
        if not solutions:
            msg = (
                f"Unable to find route from {start_points} to any of "
                f"{end_points} (route ID: {route_id}). Please verify "
                "that the start and end points are not separated by "
                "hard barriers or invalid cost cells."
            )
            logger.error(msg)
            return

        logger.debug(
            "Got result from Rust for route_id %d. Processing..."
            "\n\t- Start points: %r\n\t- End points: %r",
            route_id,
            start_points,
            end_points,
        )
        for solution in solutions:
            if len(solution) == ROUTE_SOLUTION_LEN:
                indices, optimized_objective, dbl = solution
            else:  # pragma: no cover
                msg = f"Unexpected route solution payload: {solution!r}"
                raise revrtKeyError(msg)

            attrs_key = (route_id, indices[0])
            attrs = {
                **self.route_attrs.get(attrs_key, self.default_attrs),
                "dropped_barrier_layers": json.dumps(dbl),
            }
            yield indices, optimized_objective, attrs

    def _validate_start_points(self, points):
        """Validate start points by removing cells invalid cost"""
        points = _get_valid_points(
            points, self.routing_layers.full_shape, point_type="start"
        )
        if not points or not self.routing_scenario.ignore_invalid_costs:
            return points

        routing_options = {point[-1] for point in points}
        bad_point_inds = set()
        for r_o in routing_options:
            rows, cols = np.array(
                [point[:2] for point in points if point[-1] == r_o]
            ).T
            costs = self.routing_layers.costs[r_o].isel(
                y=xr.DataArray(rows, dims="points"),
                x=xr.DataArray(cols, dims="points"),
            )

            cost_values = costs.compute()
            bad_point_inds |= set(
                np.where(np.isnan(cost_values) | (cost_values <= 0))[0]
            )

        if not bad_point_inds:
            return points

        invalid_points = {points[i] for i in bad_point_inds}
        msg = (
            f"One or more of the start points have an invalid cost "
            f"(must be > 0): {invalid_points}\n"
            "Dropping these from consideration..."
        )
        warn(msg, revrtWarning)

        if not points:
            all_invalid_points_msg = (
                "None of the start points have a valid cost (must be > 0): "
                f"{points}"
            )
            raise revrtLeastCostPathNotFoundError(all_invalid_points_msg)

        return [p for p in points if p not in invalid_points]

    def _validate_end_points(self, points):
        """Filter out invalid endpoints; raise if all are invalid"""
        points = _get_valid_points(
            points, self.routing_layers.full_shape, point_type="end"
        )
        if not points or not self.routing_scenario.ignore_invalid_costs:
            return points

        routing_options = {point[-1] for point in points}
        bad_point_inds = set()
        for r_o in routing_options:
            rows, cols = np.array(
                [point[:2] for point in points if point[-1] == r_o]
            ).T
            costs = self.routing_layers.costs[r_o].isel(
                y=xr.DataArray(rows, dims="points"),
                x=xr.DataArray(cols, dims="points"),
            )

            cost_values = costs.compute()
            bad_point_inds |= set(
                np.where(np.isnan(cost_values) | (cost_values <= 0))[0]
            )
        if not bad_point_inds:
            return points

        invalid_points = {points[i] for i in bad_point_inds}
        msg = (
            f"One or more of the end points have an invalid cost "
            f"(must be > 0): {invalid_points}\n"
            "Dropping these from consideration..."
        )
        warn(msg, revrtWarning)
        points = [p for p in points if p not in invalid_points]

        if not points:
            all_invalid_points_msg = (
                "None of the end points have a valid cost (must be > 0): "
                f"{sorted(invalid_points)}"
            )
            raise revrtLeastCostPathNotFoundError(all_invalid_points_msg)

        return points

    def _reset_routing_layers(self):
        """Close handler and remove built routing layers from memory"""
        self.routing_layers.close()
        del self.routing_layers


def _validate_out_fp(out_fp, save_paths):
    """Validate output filepath extension"""
    out_fp = Path(out_fp)

    if save_paths and out_fp.suffix.lower() != ".gpkg":
        msg = (
            "When saving paths, the output file should have a '.gpkg' "
            f"extension to ensure proper format! Got input file: '{out_fp}'. "
            "Adding '.gpkg' extension... "
        )
        warn(msg, revrtWarning)
        out_fp = out_fp.with_suffix(".gpkg")
    elif not save_paths and out_fp.suffix.lower() != ".csv":
        msg = (
            "When not saving paths, the output file should have a '.csv' "
            f"extension to ensure proper format! Got input file: '{out_fp}'. "
            "Adding '.csv' extension... "
        )
        warn(msg, revrtWarning)
        out_fp = out_fp.with_suffix(".csv")

    logger.debug("Validated output filepath: %s", out_fp)
    return out_fp


def _get_valid_points(points, arr_shape, point_type):
    """Get only points that are within array bounds"""
    valid_points = []
    invalid_points = []
    for point in points:
        if _is_valid_point(point, arr_shape):
            valid_points.append(point)
        else:
            invalid_points.append(point)

    if invalid_points:
        msg = (
            f"One or more of the {point_type} points are out of bounds for an "
            f"array of shape {arr_shape}: {invalid_points}\n"
            "Dropping these from consideration..."
        )
        warn(msg, revrtWarning)

    return valid_points


def _is_valid_point(point, arr_shape):
    """Check if point is within array bounds"""
    row, col, *__ = point
    return 0 <= row < arr_shape[0] and 0 <= col < arr_shape[1]


def _add_route_ids(route_definitions):
    """Add route IDs to route definitions missing them"""
    logger.info(
        "Route ID's missing from route definitions - adding definition "
        "index as route ID..."
    )
    return [
        (ind, start_points, end_points)
        for ind, (start_points, end_points) in enumerate(route_definitions)
    ]


def _compute_lens(route, cell_size):
    """Compute the total length and cell by cell length of LCP"""
    # Use Pythagorean theorem to calculate length between cells (km)
    # Use c**2 = a**2 + b**2 to determine length of individual paths
    lens = np.sqrt(np.sum(np.diff(route, axis=0) ** 2, axis=1))
    total_path_length = np.sum(lens) * cell_size / 1000

    # Need to determine distance coming into and out of any cell.
    # Assume paths start and end at the center of a cell. Therefore,
    # distance traveled in the cell is half the distance entering it
    # and half the distance exiting it. Duplicate all lengths,
    # pad 0s on ends for start  and end cells, and divide all
    # distance by half.
    lens = np.repeat(lens, 2)
    lens = np.insert(np.append(lens, 0), 0, 0)
    lens /= 2

    # Group entrance and exits distance together, and add them
    lens = lens.reshape((int(lens.shape[0] / 2), 2))
    lens = np.sum(lens, axis=1)
    return lens, total_path_length


def _transition_cost_lookup(transition_costs):
    """Build the directed transition-cost lookup used in reports"""
    if not transition_costs:
        return 0.0, {}

    default_cost = transition_costs.get("default", 0) or 0
    pairwise_costs = {}
    for rule in transition_costs.get("pairwise", []):
        pairwise_costs[(rule["from"], rule["from"])] = 0
        pairwise_costs[(rule["to"], rule["to"])] = 0
        pairwise_costs[(rule["from"], rule["to"])] = rule["cost"]

    return default_cost, pairwise_costs


def _routing_options_output_fp(out_fp):
    """pathlib.Path: Companion output path for routing-option pieces"""
    return out_fp.with_name(f"{out_fp.stem}_routing_options{out_fp.suffix}")


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
